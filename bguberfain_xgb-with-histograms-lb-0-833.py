from multiprocessing import Pool, cpu_count

from os.path import join



import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.metrics import fbeta_score



import cv2



# Definitions

image_format = 'jpg' # one of 'jpg' or 'tif'

histogram_size = 64

xgb_tune_train_subsample = 0.35 # Percent of train used for tune (just to avoid Kaggle's limit)

input_folder = join('..', 'input')

train_folder = join(input_folder, 'train-' + image_format)

test_folder = join(input_folder, 'test-' + image_format)
# Load data

df_train = pd.read_csv(join(input_folder, 'train.csv'))

df_sub = pd.read_csv(join(input_folder, 'sample_submission.csv'))



# One-hot encoding of 'tags'

df_train = pd.concat([df_train['image_name'], df_train.tags.str.get_dummies(sep=' ')], axis=1)

all_labels = df_train.columns[1:].tolist()



df_train.sample(10, random_state=42)
def load_train_im(f):

    return cv2.imread(join(train_folder, f + '.jpg'))



def load_test_im(f):

    return cv2.imread(join(test_folder, f + '.jpg'))



def im2hist(im):

    im_size = np.prod(im.shape[:2])



    hist_b = cv2.calcHist([im], [0], None, [histogram_size], [0, 256]).ravel() / im_size

    hist_g = cv2.calcHist([im], [1], None, [histogram_size], [0, 256]).ravel() / im_size

    hist_r = cv2.calcHist([im], [2], None, [histogram_size], [0, 256]).ravel() / im_size



    return np.r_[hist_b, hist_g, hist_r]



def train_file2hist(f):

    return im2hist(load_train_im(f))



def test_file2hist(f):

    return im2hist(load_test_im(f))



# From here: https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/fixed-f2-score-in-python/code

def f2_score(y_true, y_pred):

    y_true, y_pred, = np.array(y_true), np.array(y_pred)

    return fbeta_score(y_true, y_pred, beta=2, average='samples')
# For each sample, we create its histogram

pool = Pool(cpu_count())

try:

    hist_train = pool.map(train_file2hist, df_train['image_name'])

    X_train = np.array(hist_train)



    hist_test = pool.map(test_file2hist, df_sub['image_name'])

    X_test = np.array(hist_test)

finally:

    pool.terminate()
# XGB hyperparameters

xgb_params = {

    'objective': 'binary:logistic',

    'eta': 0.3,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'silent': 1

}
class XGBCVHolder:

    """

    This is a hack to XGBoost, which does not provide an API to access 

    the models trained over xgb.cv

    """

    def __init__(self):

        self.models = []

        self.dtests = []

        self.called = False



    def __call__(self, env):

        if not self.called:

            self.called = True

            for cvpack in env.cvfolds:

                self.models.append(cvpack.bst)

                self.dtests.append(cvpack.dtest)



    def predict_oof(self, ntree_limit=0):

        y = []

        y_hat = []

        for model, dtest in zip(self.models, self.dtests):

            y.extend(dtest.get_label())

            y_hat.extend(model.predict(dtest, ntree_limit=ntree_limit))



        return np.array(y), np.array(y_hat)
# Generate dtest (submission)

dtest = xgb.DMatrix(X_test)

df_sub['tags'] = ''



train_subsample = np.random.choice(df_train.shape[0], int(df_train.shape[0] * xgb_tune_train_subsample))



# Begin CV to find best num_boost_trees

y_true, y_pred = [], []

for label in all_labels:

    print("Tunning label '{}'...".format(label))



    # Perform cross-validation on train data

    xgb_model = XGBCVHolder()

    dtrain = xgb.DMatrix(X_train[train_subsample], df_train.iloc[train_subsample][label].values)

    result = xgb.cv(xgb_params, dtrain, num_boost_round=200, nfold=2, metrics=['error', 'auc'],

        early_stopping_rounds=20, verbose_eval=False, show_stdv=False, callbacks=[xgb_model])



    print(result.iloc[-1][['test-auc-mean', 'test-error-mean']])



    # Result of tunning

    num_boost_round = result.shape[0]





    # Get OOF predictions based on models trained on CV

    y, y_hat = xgb_model.predict_oof(ntree_limit=num_boost_round)

    y_true.append(y)

    y_pred.append(y_hat)



    # Train main model

    print("Training model...")

    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)

    ytest_hat = model.predict(dtest) > 0.5



    df_sub.loc[ytest_hat, 'tags'] = df_sub.loc[ytest_hat, 'tags'] + ' ' + label
# Join all OOF predictions and y_true

y_true = np.c_[y_true].T

y_pred = np.c_[y_pred].T



# Calculate OOF f2 score

cv_f2score = f2_score(y_true > 0.5, y_pred > 0.5)

print("Expected f2 in CV is %.4f" % cv_f2score)
df_sub.to_csv('sub_%.4f.csv' % cv_f2score, index=False)