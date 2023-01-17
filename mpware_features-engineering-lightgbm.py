import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, math, shutil

from scipy.ndimage import gaussian_filter

from scipy.stats import kurtosis, skew

from scipy.ndimage import laplace, sobel

from skimage import img_as_float

from sklearn.externals import joblib

from skimage.morphology import reconstruction

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import StratifiedKFold #for K-fold cross validation

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

import lightgbm as lgb

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

ANGLE = "inc_angle"

ICEBERG = "is_iceberg"

BAND1 = "band_1"

BAND2 = "band_2"

ID = "id"

initial_model_path = "lgbm"

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
# Load data

train = pd.read_json('../input/train.json')

print("rows, cols = " + str(train.shape))

train.head()
# Any missing data?

train[ANGLE] = pd.to_numeric(train[ANGLE],errors='coerce')

nullPD = pd.DataFrame(train.isnull().sum(), columns=['TotalNull'])

nullPD
f, ax = plt.subplots(1,2,figsize=(16,5))

r = train.plot(kind="kde", y = ANGLE, ax=ax[0], label="Any", grid=True, title="Angle of incidence KDE")

r = train[train[ICEBERG] == 1].plot(kind="kde", label="Iceberg", grid=True, y = ANGLE, ax=ax[1], title="Angle of incidence KDE")

r = train[train[ICEBERG] == 0].plot(kind="kde", label="Not Iceberg", grid=True, y = ANGLE, ax=r)
P = train.groupby(ICEBERG)[ID].count().reset_index()

P['Percentage'] = 100*P[ID]/P[ID].sum()

P
# Features

MIN="min"

MAX="max"

MEAN="mean"

STD="std"

LAPLACE="laplacestd"

SOBEL0 = "sobelstd_x"

SOBEL1 = "sobelstd_y"

KURTOSIS = "kurtosis"

SKEW = "skew"

CORR = "pearson"

HYP = "hypstd"



AGG_COLS = [

    "%s_%s"%(BAND1,MAX), "%s_%s"%(BAND2,MAX),

    "%s_%s"%(BAND1,MIN), "%s_%s"%(BAND2,MIN),

    "%s_%s"%(BAND1,MEAN), "%s_%s"%(BAND2,MEAN),

    "%s_%s"%(BAND1,STD), "%s_%s"%(BAND2,STD),

    "%s_%s"%(BAND1,KURTOSIS), "%s_%s"%(BAND2,KURTOSIS),

    "%s_%s"%(BAND1,SKEW), "%s_%s"%(BAND2,SKEW),

    "%s_%s"%(BAND1,SOBEL0), "%s_%s"%(BAND2,SOBEL0),

    "%s_%s"%(BAND1,SOBEL1), "%s_%s"%(BAND2,SOBEL1),

    "%s_%s"%(BAND1,LAPLACE), "%s_%s"%(BAND2,LAPLACE),

    CORR,

    HYP,

]



# Volume

ISO = "iso"

ISO_COLS = [

        "%s_%s"%(BAND1,ISO), "%s_%s"%(BAND2,ISO)

]

VOLUME = "vol"



# Final features

FEATURES = [ANGLE, VOLUME] + AGG_COLS



# Isolation function.

def iso(arr):

    image = img_as_float(np.reshape(np.array(arr), [75,75]))

    image = gaussian_filter(image,2.5)

    seed = np.copy(image)

    seed[1:-1, 1:-1] = image.min()

    mask = image 

    dilated = reconstruction(seed, mask, method='dilation')

    return image-dilated



# Standard deviation for sobel filter

def sobelstd(arr, axis=0):

    image = img_as_float(np.reshape(np.array(arr), [75,75]))

    sobelstd = sobel(image, axis=axis, mode='reflect', cval=0.0).ravel()

    return [sobelstd.std(), sobelstd.max(), sobelstd.mean()]



# Standard deviation for laplace filter

def lapacestd(arr):

    image = img_as_float(np.reshape(np.array(arr), [75,75]))

    lapacestd = laplace(image, mode='reflect', cval=0.0).ravel()

    return [lapacestd.std(), lapacestd.max(), lapacestd.mean()]



def volume(arr):

    return np.sum(arr)



def hypot(arr1, arr2):

    hyp = np.hypot(arr1, arr2)

    return [np.std(hyp), np.max(hyp), np.median(hyp)]



def computeAdditionalFeatures(df):

    # Aggregation on raw signal

    df["%s_%s"%(BAND1,MAX)] = df[BAND1].apply(lambda x: np.max(x))

    df["%s_%s"%(BAND2,MAX)] = df[BAND2].apply(lambda x: np.max(x))

    df["%s_%s"%(BAND1,MIN)] = df[BAND1].apply(lambda x: np.min(x))

    df["%s_%s"%(BAND2,MIN)] = df[BAND2].apply(lambda x: np.min(x))

    df["%s_%s"%(BAND1,MEAN)] = df[BAND1].apply(lambda x: np.mean(x))

    df["%s_%s"%(BAND2,MEAN)] = df[BAND2].apply(lambda x: np.mean(x))

    df["%s_%s"%(BAND1,STD)] = df[BAND1].apply(lambda x: np.std(x))

    df["%s_%s"%(BAND2,STD)] = df[BAND2].apply(lambda x: np.std(x))

    df["%s_%s"%(BAND1,KURTOSIS)] = df[BAND1].apply(lambda x: kurtosis(x))

    df["%s_%s"%(BAND2,KURTOSIS)] = df[BAND2].apply(lambda x: kurtosis(x))    

    df["%s_%s"%(BAND1,SKEW)] = df[BAND1].apply(lambda x: skew(x))

    df["%s_%s"%(BAND2,SKEW)] = df[BAND2].apply(lambda x: skew(x))     

    df["%s_%s"%(BAND1,SOBEL0)] = df[BAND1].apply(lambda x: sobelstd(x, axis=0)[0])

    df["%s_%s"%(BAND1,SOBEL1)] = df[BAND1].apply(lambda x: sobelstd(x, axis=1)[0])    

    df["%s_%s"%(BAND2,SOBEL0)] = df[BAND2].apply(lambda x: sobelstd(x, axis=0)[0])

    df["%s_%s"%(BAND2,SOBEL1)] = df[BAND2].apply(lambda x: sobelstd(x, axis=1)[0])   

    df["%s_%s"%(BAND1,LAPLACE)] = df[BAND1].apply(lambda x: lapacestd(x)[0])

    df["%s_%s"%(BAND2,LAPLACE)] = df[BAND2].apply(lambda x: lapacestd(x)[0])    

    df[CORR] = df.apply(lambda row: np.corrcoef(x=row[BAND1], y=row[BAND2])[1,0], axis=1)

    df[HYP] = df.apply(lambda row: hypot(row[BAND1], row[BAND2])[0], axis=1)

    

    # Volume

    df["%s_%s"%(BAND1,ISO)] = df[BAND1].apply(lambda x: iso(x))

    df["%s_%s"%(BAND2,ISO)] = df[BAND2].apply(lambda x: iso(x))

    df[VOLUME] = (df["%s_%s"%(BAND1,ISO)] + df["%s_%s"%(BAND2,ISO)]).apply(volume)



    cleanDF = df.dropna()

    ret = cleanDF[FEATURES].copy(deep=True)

    ret_labels = None

    if ICEBERG in cleanDF.columns:

        ret_labels = cleanDF[[ICEBERG]]

    ret_ids = cleanDF[[ID]]

    ret_cols = ret.columns

            

    return ret, ret_labels, ret_ids
def read_and_normalize_train_data(train_df):

    featuresDF, labelsDF, idsDF = computeAdditionalFeatures(train_df.copy(deep=True))

    train_features = featuresDF.as_matrix()

    train_target = labelsDF.as_matrix()

    train_id = idsDF.as_matrix()

    print("Features size: %s/%s"%(str(train_features.shape), str(train_target.shape)))

    return train_features, train_target, train_id, featuresDF[FEATURES].columns
def fit_evaluate_model_lgbm(X_train, Y_train, X_valid, Y_valid, train_data_columns, model_path, num_fold, importance=False):

    X_trainDF = pd.DataFrame(X_train, columns=train_data_columns)

    X_validDF = pd.DataFrame(X_valid, columns=train_data_columns)

    train_dataset = lgb.Dataset(X_trainDF, Y_train.reshape(Y_train.shape[0]))

    test_dataset = lgb.Dataset(X_validDF, Y_valid.reshape(Y_valid.shape[0]))

    # Fit

    evals_result = {}

    params = {

            'objective': 'binary',

            'metric': 'binary_logloss',

            'boosting': 'gbdt',

            'learning_rate': 0.1,

            'num_rounds': 150,

            'early_stopping_rounds': 100

    }

    gbm = lgb.train(params, train_dataset, 

                    valid_sets=test_dataset, 

                    evals_result=evals_result,

                    verbose_eval=50)

    # Evaluate

    predict_y_proba_gbm = gbm.predict(X_valid, num_iteration=gbm.best_iteration) # Proba of class 1

    predict_y_gbm = np.where(predict_y_proba_gbm.reshape((predict_y_proba_gbm.shape[0])) > 0.5, 1, 0)



    score_ll = metrics.log_loss(Y_valid, predict_y_proba_gbm)

    score_ac = metrics.accuracy_score(Y_valid, predict_y_gbm)

    score_pr = metrics.precision_score(Y_valid, predict_y_gbm)

    score_re = metrics.recall_score(Y_valid, predict_y_gbm)

    score = [score_ll, score_ac, score_pr, score_re]

    

    if (importance == True):

        ax = lgb.plot_importance(gbm, max_num_features=20, figsize=(16, 5))

        plt.show()

    

    gbmDF = pd.DataFrame([tuple(gbm.feature_importance())], columns= gbm.feature_name())

    gbmDF.sort_index(axis=1, inplace=True)



    return score, gbmDF, gbm
# With KFolder stratified CV.

def run_cross_validation_create_models(train_df, model_path, nfolds=4, break_fold=-1, importance=False):

    train_data, train_target, train_id, train_data_columns = read_and_normalize_train_data(train_df)

    train_target = train_target.ravel()

    kf = StratifiedKFold(n_splits=nfolds, random_state=None, shuffle=True)

    num_fold = 0

    sum_score_ll = 0

    sum_score_ac = 0

    sum_score_pr = 0

    sum_score_re = 0

    scores = []

    models = []

    importanceDF = pd.DataFrame()

    for train_index, test_index in kf.split(train_data, train_target):

        num_fold += 1

        print('\n==> Start KFold number {} from {}'.format(num_fold, nfolds))

        X_train = train_data[train_index]

        Y_train = train_target[train_index]

        X_valid = train_data[test_index]

        Y_valid = train_target[test_index]

        print("Train size: %s/%s"%(str(X_train.shape), str(Y_train.shape)))

        print("Valid size: %s/%s"%(str(X_valid.shape), str(Y_valid.shape)))

        score, impDF, m = fit_evaluate_model_lgbm(X_train,Y_train, X_valid, Y_valid, train_data_columns, 

                                   model_path, num_fold, importance=False)

        models.append(m)

        if len(importanceDF) == 0:

            importanceDF = impDF

        else:

            importanceDF = pd.concat([importanceDF, impDF])

        print('Test loss:', score[0])

        print('Test accuracy:', score[1])

        sum_score_ll += score[0]*len(test_index)

        sum_score_ac += score[1]*len(test_index)

        sum_score_pr += score[2]*len(test_index)

        sum_score_re += score[3]*len(test_index)

        scores.append(score)

        # Break KFold loop

        if (break_fold > 0) & (break_fold == num_fold):

            break

    score_ll = sum_score_ll/len(train_data)

    score_ac = sum_score_ac/len(train_data)

    score_pr = sum_score_pr/len(train_data)

    score_re = sum_score_re/len(train_data)

    print('\nCV average scores:')

    print('log_loss: %s\naccuracy: %s\nprecision: %s\nrecall: %s\n'%(score_ll, score_ac, score_pr, score_re))

    return scores, importanceDF, models
# Training

NFOLDS = 10

seed = 1337

np.random.seed(seed)

print("-----------   Seed %d   ----------------"%seed)

model_path = "%s.cv%d.%d"%(initial_model_path, NFOLDS, seed)

os.makedirs(model_path, exist_ok=True)



scores, importanceDF, models = run_cross_validation_create_models(train, model_path, nfolds=NFOLDS, break_fold=-1, importance = True)

scores = np.array(scores)

scores_loss = scores[:,0]

scores_other = scores[:,1:4]



box_loss = pd.DataFrame(scores_loss, columns=["log loss"])

box_other = pd.DataFrame(scores_other*100.0, columns=["accuracy", "precision", "recall"])

f, ax = plt.subplots(1, 2, figsize=(16,3))

box_other.boxplot(ax=ax[0], showmeans=True)

ax[0].set_title("Accuracy, Precision, Recall")

box_loss.boxplot(ax=ax[1], showmeans=True)

ax[1].set_title("Log Loss")

plt.show()

print("CV val Log loss: %s"%(np.mean(scores_loss)))



importanceDF.head()
mgbmDF = pd.DataFrame(importanceDF.mean(), columns=["Importance"])

mgbmDF = mgbmDF.apply(lambda x: 100.0 * x / float(x.sum()))

mgbmDF.sort_values(by=["Importance"], ascending=[True]).plot(kind="barh", legend=False, grid=True, figsize=(16,8))

a = plt.title("Features Importance CV mean")
# Testing

test = pd.read_json('../input/test.json')

print("rows, cols = " + str(test.shape))
def read_and_normalize_test_data(test_df):

    featuresDF, labelsDF, idsDF = computeAdditionalFeatures(test_df.copy(deep=True))

    test_features = featuresDF.as_matrix()

    test_id = idsDF.as_matrix()

    print("Features size: %s/%s"%(str(test_features.shape), str(test_id.shape)))

    return test_features, test_id, featuresDF[FEATURES].columns
X_test, X_test_id, X_test_columns = read_and_normalize_test_data(test)
yfull_proba_train = []

yfull_proba_test = []

yfull_label_test = []

num_fold = 0

# Run predictions on each fold

for model in models:

    num_fold += 1

    print('==> Start KFold number {} from {}'.format(num_fold, NFOLDS))

    # Testing

    predicted_test = model.predict(X_test, num_iteration=model.best_iteration) # Proba of class 1

    predicted_test_label = np.where(predicted_test.reshape((predicted_test.shape[0])) > 0.5, 1, 0)

    yfull_proba_test.append(predicted_test)

    yfull_label_test.append(predicted_test_label)
def merge_several_folds_mean(data, nfolds):

    a = np.array(data[0])

    aPD = pd.DataFrame(a, columns = ["%s_%d"%(ICEBERG, 1)])

    for i in range(1, nfolds):

        a = a + np.array(data[i])

        aPD["%s_%d"%(ICEBERG, i+1)] = np.array(data[i])

    a = a / nfolds*1.0

    return a, aPD
kfold_cols = ["%s_%d"%(ICEBERG, i) for i in range(1, NFOLDS + 1) ]

predicted_test_mean, predicted_test_pd = merge_several_folds_mean(yfull_proba_test, NFOLDS)

predicted_test_pd.head()
# Submission file

submission = pd.DataFrame()

submission[ID]=test[ID]

submission[ICEBERG]=predicted_test_mean

submission.to_csv("submissionlgbmv1.csv", index=False, sep=",", decimal=".")