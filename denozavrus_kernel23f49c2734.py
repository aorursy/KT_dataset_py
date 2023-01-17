import pydicom
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
import gc
from keras.applications.densenet import preprocess_input, DenseNet169
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

sns.set()

%config InlineBackend.figure_format = 'retina'
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
train_df.isnull().sum()
missing_ids_sex = train_df[train_df['sex'].isnull()]['patient_id']
missing_ids_sex.unique()
train_df.loc[train_df['patient_id'].isin(missing_ids_sex.unique()), ['sex']]['sex'].value_counts()
missing_ids_age = train_df[train_df['age_approx'].isnull()]['patient_id']
missing_ids_age.unique()
train_df.loc[train_df['patient_id'].isin(missing_ids_age.unique()), ['age_approx']]['age_approx'].value_counts()
train_df.loc[train_df['patient_id'].isin(missing_ids_sex.unique()), ['sex']]['sex'].value_counts()
missing_ids_age = train_df[train_df['age_approx'].isnull()]['patient_id']
missing_ids_age.unique()
train_df.loc[train_df['patient_id'].isin(missing_ids_age.unique()), ['age_approx']]['age_approx'].value_counts()
set(missing_ids_age) - set(missing_ids_sex)

missing_ids_site = train_df[train_df['anatom_site_general_challenge'].isnull()]['patient_id']
missing_ids_site.unique() 
ind_to_drop = train_df[train_df['patient_id'].isin(missing_ids_sex.unique())].index
train_df.drop(index = ind_to_drop, inplace = True)
train_df.isnull().sum()
id_w_zero = train_df[train_df['age_approx'] < 5]['patient_id']
id_w_zero.values
ind_zero = train_df.loc[train_df['patient_id'].isin(id_w_zero.values)].index
train_df.loc[train_df['patient_id'].isin(id_w_zero.values)]
train_df.loc[ind_zero, 'age_approx'] = 10.0
train_df.loc[ind_zero]
val = {'age_approx' : train_df['age_approx'].mean()}
train_df.fillna(val, inplace = True)
train_df.isnull().sum()
mapping = {'male' : 1, 'female' : 0}
train_df['sex'] = train_df['sex'].map(mapping)
test_df['sex'] = test_df['sex'].map(mapping)
train_df.head()

train_df[train_df['anatom_site_general_challenge'].isna()]['target'].value_counts()
ind_to_drop = train_df[(train_df['anatom_site_general_challenge'].isna()) & (train_df['target'] == 0)].index
ind_to_drop
train_df.drop(ind_to_drop, inplace = True)
train_df.isnull().sum() ,train_df.shape
train_df = train_df.fillna('torso')
train_df.isnull().sum()
train_df = train_df.drop(['diagnosis', 'benign_malignant'], axis = 1)
train_df.head()
X_train, y_train = train_df.drop('target', axis = 1), train_df['target']
X_train
y_train

test_df.isnull().sum()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imp.fit(X_train)
ind, col = test_df.index, test_df.columns
X_test = pd.DataFrame(imp.transform(test_df), index = ind, columns = col)
X_test.head()
X_test.isnull().sum()
X_train.head()
cat_features = ["anatom_site_general_challenge"]
encoded = pd.get_dummies(X_train[cat_features])
encoded.set_index(X_train.index)
X_train.drop(cat_features, inplace = True, axis = 1)
X_train_encoded = pd.concat([X_train,encoded], axis = 1)
X_train_encoded.head()
encoded = pd.get_dummies(X_test[cat_features])
encoded.set_index(X_test.index)
X_test.drop(cat_features, inplace = True, axis = 1)
X_test_encoded = pd.concat([X_test,encoded], axis = 1)
X_test_encoded.head()
train_df_clean = pd.concat([X_train_encoded, y_train], axis = 1)
test_df_clean = X_test_encoded
train_df_clean.head()
train_df_clean.to_csv('train_clean.csv', index = False)
test_df_clean.to_csv('test_clean.csv', index = False)

train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

img_size = 256

def resize_image(img):
    old_size = img.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1],new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0,0,0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def load_image(path, img_id):
    path = os.path.join(path,img_id+'.jpg')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img = resize_image(img)
    new_img = preprocess_input(new_img)
    return new_img
fig = plt.figure(figsize=(16, 16))
for i,image_id in enumerate(np.random.choice(train_df[train_df['target']== 0].image_name,5)):
    image = load_image(train_img_path,image_id)
    fig.add_subplot(1,5,i+1)
    plt.imshow(image)
img_size = 256
batch_size = 16 #16 images per batch

train_img_ids = train_df.image_name.values
n_batches = len(train_img_ids) // batch_size + 1

#Model to extract image features
inp = Input((256,256,3))
backbone = DenseNet169(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)
features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = train_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = load_image(train_img_path,img_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,img_id in enumerate(batch_ids):
        features[img_id] = batch_preds[i]
train_feats = pd.DataFrame.from_dict(features, orient = 'index')
train_feats.to_csv('train_img_features.csv')
train_feats.head()
test_img_ids = test_df.image_name.values
n_batches = len(test_img_ids) // batch_size + 1
features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = test_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = load_image(test_img_path,img_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,img_id in enumerate(batch_ids):
        features[img_id] = batch_preds[i]
test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.to_csv('test_img_features.csv')
test_feats.head()
train_feat_img = pd.read_csv ('../input/melanoma-dataset-for-images/train_img_features.csv')
test_feat_img = pd.read_csv ('../input/melanoma-dataset-for-images/test_img_features.csv')
test_feat_img.head()
train_feat_img = train_feat_img.set_index('Unnamed: 0')
test_feat_img = test_feat_img.set_index('Unnamed: 0')
train_feat_img.head()
test_feat_img.head()
train_data = pd.read_csv('./train_clean.csv')
test_data = pd.read_csv('./test_clean.csv')
train_data.head()
test_data.head()
X_train_encoded = train_data.drop('target', axis = 1)
y_train = train_data['target']
X_train_encoded.head()
y_train.head()
X_train_full =  X_train_encoded.merge (train_feat_img, 
                       how = 'inner',
                      left_on = 'image_name', 
                      right_index = True,
                      )
X_train_full.head()
X_test_full = test_data.merge (test_feat_img, 
                      how = 'inner',
                      left_on = 'image_name', 
                      right_index = True,
                      )
X_test_full.head()
X_train_full.drop(['image_name', 'patient_id'], inplace = True, axis = 1)
X_train_full.head()
X_train.shape
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
boosting = xgb.XGBClassifier(max_depth = 5, 
                            reg_lambda = 1.3,
                            subsample = 0.8, 
                            n_estimators = 800, 
                            min_child_weight = 3, 
                            learning_rate = 0.3)
scorer = make_scorer(roc_auc_score)
skf = StratifiedKFold(n_splits = 5)
score_cv = cross_val_score(boosting, X_train_full, y_train, cv = skf)
X_train_full.head()
X_test_full.head()
score_cv
boost = xgb.XGBClassifier ()

scorer = make_scorer(roc_auc_score)

param_grid = {
    'n_estimators': [ 500, 800, 1000],
    'max_depth': [5, 10, 20, 30],
    'reg_lambda': [1.1, 1.2, 1.3],
    "min_child_weight": [1,3,5]
}

skf = StratifiedKFold(n_splits = 5)

gs = RandomizedSearchCV(
        estimator = boost,
        param_distributions = param_grid, 
        cv = skf, 
        n_jobs = -1, 
        scoring = scorer,
        verbose = True, 
        n_iter = 10
    )
best_est = gs.fit(X_train_full, y_train)
best_est.best_score_
best_est.best_params_
boosting = xgb.XGBClassifier(max_depth = 5, 
                            reg_lambda = 1.3,
                            subsample = 0.8, 
                            n_estimators = 800, 
                            min_child_weight = 3, 
                            learning_rate = 0.3)
boosting.fit(X_train_full, y_train)
X_test_full.drop('patient_id', axis = 1, inplace = True)

X_test_full.head()
X_test_full = X_test_full.set_index('image_name')
X_test_full.head()
y_test = boosting.predict_proba(X_test_full)[:, 1]
y_test
X_test_full.index
submission = pd.DataFrame({
    'image_name': X_test_full.index,  
    'target' : y_test
})
submission.head()
submission.to_csv('submission.csv', index = False)
