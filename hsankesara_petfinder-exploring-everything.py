# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import gc
gc.collect()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train/train.csv')
df_breed = pd.read_csv('../input/breed_labels.csv')
df_color = pd.read_csv('../input/color_labels.csv')
df_state = pd.read_csv('../input/state_labels.csv')
df_test = pd.read_csv('../input/test/test.csv')
df_train.head()

df_breed.head()
df_color.head()
df_state.head()
sns.countplot(df_train.AdoptionSpeed)
sns.countplot(df_train.Type)
# Checking Age Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Age, ax=ax)
# Checking Age,Fee and AdoptionnSpeed correlation
sns.heatmap(df_train[['Age', 'Fee', 'AdoptionSpeed']].corr(), annot=True)
df_train.Age.describe()
# Since Age is skewed, We can try logarithmic transformation
sns.distplot(np.log(df_train.Age + 0.5))
# Checking put distribution of Breed1 and Breed2
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.Breed1, ax=ax[0])
sns.distplot(df_train.Breed2, ax=ax[1])
df_train.Name = df_train.Name.fillna('')
df_test.Name = df_test.Name.fillna('')
df_train['Name'] = df_train['Name'].replace('No Name Yet', '')
df_test['Name'] = df_test['Name'].replace('No Name Yet', '')
df_train['name_len'] = df_train.Name.str.len()
df_test['name_len'] = df_test.Name.str.len()
df_train.name_len.head()
sns.distplot(df_train.name_len)
sns.heatmap(df_train[['name_len', 'AdoptionSpeed']].corr(), annot=True)
sns.distplot(np.log(df_train.name_len + 1 - df_train.name_len.min()))
# Gender distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.Gender, ax=ax)
# Quantity Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Quantity, ax=ax)
f, ax = plt.subplots(figsize=(21, 6))
quant_gender1 = df_train[df_train['Gender'] == 1]
quant_gender2 = df_train[df_train['Gender'] == 2]
quant_gender3= df_train[df_train['Gender'] == 3]
sns.distplot(quant_gender1.Quantity, ax=ax , hist=False, rug=True)
sns.distplot(quant_gender2.Quantity, ax=ax,  hist=False, rug=True)
sns.distplot(quant_gender3.Quantity, ax=ax,  hist=False, rug=True)
plt.show()
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot('Quantity',data=df_train,hue='Gender', ax=ax)
# Color distribution
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(18,10))
sns.countplot(df_train.Color1, ax=ax[0])
sns.countplot(df_train.Color2, ax=ax[1])
sns.countplot(df_train.Color3, ax=ax[2])
# Maturity Size Distribution
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.MaturitySize, ax=ax)
# Furlength
f, ax = plt.subplots(figsize=(21, 6))
sns.countplot(df_train.FurLength, ax=ax)
# Vaccination
f, ax = plt.subplots(figsize=(21, 6))
sns.distplot(df_train.Vaccinated, ax=ax)
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(18,10))
sns.distplot(df_train.Sterilized, ax=ax[0])
sns.distplot(df_train.Health, ax=ax[1])
sns.distplot(df_train.Dewormed, ax=ax[2])
sns.distplot(df_train.Fee, ax=ax[3])
f, ax = plt.subplots(figsize=(12, 8))
sns.countplot(df_train.State, ax=ax)
f, ax = plt.subplots(figsize=(12, 8))
sns.distplot(df_train.Description.fillna('').str.len(), ax=ax)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.PhotoAmt, ax=ax[0])
sns.distplot(df_train.VideoAmt, ax=ax[1])
f, ax = plt.subplots(figsize=(24, 18))
sns.heatmap(df_train.corr(), annot=True, ax=ax)
import json
train_sentiment_path = '../input/train_sentiment/'
test_sentiment_path = '../input/test_sentiment/'
train_meta_path = '../input/train_metadata/'
test_meta_path = '../input/test_metadata/'
def get_sentiment(pet_id, json_dir):
    try:
        with open(json_dir + pet_id + '.json') as f:
            data = json.load(f)
        return pd.Series((data['documentSentiment']['magnitude'], data['documentSentiment']['score']))
    except FileNotFoundError:
        return pd.Series((np.nan, np.nan))
df_train[['desc_magnitude', 'desc_score']] = df_train['PetID'].progress_apply(lambda x: get_sentiment(x, train_sentiment_path))
df_test[['desc_magnitude', 'desc_score']] = df_test['PetID'].progress_apply(lambda x: get_sentiment(x, test_sentiment_path))
df_train.head()
sns.heatmap(df_train[['desc_magnitude', 'desc_score', 'AdoptionSpeed']].corr(), annot=True)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.desc_magnitude.dropna(), ax=ax[0])
sns.distplot(df_train.desc_score.dropna(), ax=ax[1])
df_train.desc_magnitude.count() / df_train.shape[0]
sns.distplot(np.log(df_train.desc_magnitude.dropna() + 0.5))
sns.heatmap(np.corrcoef(df_train.Description.fillna('').str.len(), df_train.AdoptionSpeed), annot=True)
target = df_train['AdoptionSpeed']
train_id = df_train['PetID']
test_id = df_test['PetID']
df_train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
df_test.drop(['PetID'], axis=1, inplace=True)
%%time
vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
df_train.loc[:, 'vertex_x'] = vertex_xs
df_train.loc[:, 'vertex_y'] = vertex_ys
df_train.loc[:, 'bounding_confidence'] = bounding_confidences
df_train.loc[:, 'bounding_importance'] = bounding_importance_fracs
df_train.loc[:, 'dominant_blue'] = dominant_blues
df_train.loc[:, 'dominant_green'] = dominant_greens
df_train.loc[:, 'dominant_red'] = dominant_reds
df_train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
df_train.loc[:, 'dominant_score'] = dominant_scores
df_train.loc[:, 'label_description'] = label_descriptions
df_train.loc[:, 'label_score'] = label_scores


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
df_test.loc[:, 'vertex_x'] = vertex_xs
df_test.loc[:, 'vertex_y'] = vertex_ys
df_test.loc[:, 'bounding_confidence'] = bounding_confidences
df_test.loc[:, 'bounding_importance'] = bounding_importance_fracs
df_test.loc[:, 'dominant_blue'] = dominant_blues
df_test.loc[:, 'dominant_green'] = dominant_greens
df_test.loc[:, 'dominant_red'] = dominant_reds
df_test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
df_test.loc[:, 'dominant_score'] = dominant_scores
df_test.loc[:, 'label_description'] = label_descriptions
df_test.loc[:, 'label_score'] = label_scores
image_meta_col = ['vertex_x', 'vertex_y', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score']
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df_train[image_meta_col].corr(), annot=True)
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(18,10))
sns.distplot(df_train.vertex_x, ax=ax[0])
sns.distplot(df_train.vertex_y, ax=ax[1])
sns.distplot(np.log(df_train.vertex_x + 1 - df_train.vertex_x.min()), ax=ax[2])
sns.distplot(np.log(df_train.vertex_y + 1 - df_train.vertex_x.min()), ax=ax[3])
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,10))
sns.distplot(df_train.vertex_x + df_train.vertex_y, ax=ax[0])
sns.distplot(np.log(df_train.vertex_x + 1 - df_train.vertex_x.min()) +np.log(df_train.vertex_y + 1 - df_train.vertex_x.min()) , ax=ax[1])
sns.distplot(df_train.bounding_importance)
sns.distplot(df_train.bounding_confidence)
sns.distplot((df_train.bounding_confidence + df_train.bounding_importance) / 2)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(df_train.dominant_red, ax=ax, color='red', hist=False)
sns.distplot(df_train.dominant_green, ax=ax, color='green',  hist=False)
sns.distplot(df_train.dominant_blue, ax=ax, color='blue', hist=False)
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(np.log(df_train.dominant_red), ax=ax, color='red', hist=False)
sns.distplot(np.log(df_train.dominant_green), ax=ax, color='green',  hist=False)
sns.distplot(np.log(df_train.dominant_blue), ax=ax, color='blue', hist=False)
sns.distplot(np.log((df_train.dominant_blue + df_train.dominant_green + df_train.dominant_red ) / 3 + 3))
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(df_train.dominant_pixel_frac, ax=ax, color='red', hist=False)
sns.distplot(df_train.dominant_score, ax=ax, color='green',  hist=False)
sns.distplot(df_train.label_score, ax=ax, color='blue', hist=False)
sns.distplot((df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3)
sns.heatmap(np.corrcoef((df_train.bounding_confidence + df_train.bounding_importance) / 2, np.log((df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3 + 3)), annot=True)
df_train.isna().sum()
def log_transform(feature, df_train, df_test):
    min_feature = min(df_train[feature].min(), df_test[feature].min())
    df_train[feature] = np.log(df_train[feature] + 1 - min_feature)
    df_test[feature] = np.log(df_test[feature] + 1 - min_feature)
    return df_train, df_test
df_train, df_test = log_transform('vertex_x', df_train, df_test)
df_train, df_test = log_transform('vertex_y', df_train, df_test)
df_train['bounding_agg'] = (df_train.bounding_confidence + df_train.bounding_importance) / 2
df_test['bounding_agg'] = (df_test.bounding_confidence + df_test.bounding_importance) / 2
df_train['dominant_color'] = (df_train.dominant_blue + df_train.dominant_green + df_train.dominant_red ) / 3
df_test['dominant_color'] = (df_test.dominant_blue + df_test.dominant_green + df_test.dominant_red ) / 3
df_train, df_test = log_transform('dominant_color', df_train, df_test)
df_train['dominant_frac_agg'] = (df_train.dominant_pixel_frac + df_train.dominant_score + df_train.label_score) / 3
df_test['dominant_frac_agg'] = (df_test.dominant_pixel_frac + df_test.dominant_score + df_test.label_score) / 3
df_train.info()
df_train.drop(['Name', 'Description', 'RescuerID', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score', 'label_description'], axis=1, inplace=True)
df_test.drop(['Name', 'Description', 'RescuerID', 'bounding_confidence', 'bounding_importance', 'dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score', 'label_description'], axis=1, inplace=True)
df_train.info()
magnitude_std = df_train.desc_magnitude.std()
magnitude_mean = df_train.desc_magnitude.mean()
score_std = df_train.desc_score.std()
score_mean = df_train.desc_score.mean()
df_train['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_train['desc_score'].fillna( np.random.normal(score_mean, score_std), inplace=True)
df_test['desc_magnitude'].fillna(np.random.normal(magnitude_mean, magnitude_std), inplace=True)
df_test['desc_score'].fillna(np.random.normal(score_mean, score_std), inplace=True)
category_columns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
numerical_columns = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'name_len', 'desc_magnitude', 'desc_score', 'bounding_agg', 'dominant_color', 'dominant_frac_agg']
df_train[category_columns] = df_train[category_columns].astype('category')
df_test[category_columns] = df_test[category_columns].astype('category')
min_age = min(df_train.Age.min(), df_test.Age.min())
df_train.Age = np.log(df_train.Age + 1 - min_age)
df_test.Age = np.log(df_test.Age + 1 - min_age)
min_magn = min(df_train.desc_magnitude.min(), df_test.desc_magnitude.min())
df_train.desc_magnitude = np.log(df_train.desc_magnitude + 1 - min_magn)
df_test.desc_magnitude = np.log(df_test.desc_magnitude + 1 - min_magn)
df_train.name_len = np.log(df_train.name_len + 1)
df_test.name_len = np.log(df_test.name_len + 1)
df_train.info()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df_train, target, train_size=0.8, random_state=1234)
import lightgbm as lgbm
params_lgbm = {'num_leaves': 38,
         'min_data_in_leaf': 146, 
         'objective':'multiclass',
         'num_class': 5,
         'max_depth': 4,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9980062052116254,
         "bagging_freq": 1,
         "bagging_fraction": 0.844212672233457,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.12757257166471625,
         "random_state": 133,
         "verbosity": -1
              }
lgbm_train = lgbm.Dataset(X_train, y_train, categorical_feature=category_columns)
lgbm_valid = lgbm.Dataset(X_val, y_val, categorical_feature=category_columns)
model_lgbm = lgbm.train(params_lgbm, lgbm_train, 10000, valid_sets=[lgbm_valid],  verbose_eval= 500, categorical_feature=category_columns, early_stopping_rounds = 200)
(np.argmax(model_lgbm.predict(X_val), axis=1) == y_val).sum() / y_val.shape[0]
f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_lgbm.feature_importance()
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

from sklearn.ensemble import RandomForestClassifier as Rf
from sklearn.ensemble import GradientBoostingClassifier as Gb
model_rf = Rf()
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train)
model_rf.score(X_val,y_val)
f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

model_gb = Gb()
model_gb.fit(X_train,y_train)
model_gb.score(X_train,y_train)
model_gb.score(X_val,y_val)
f, ax = plt.subplots(figsize=(12, 8))
features = X_train.columns
importances = model_gb.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Checking result of two best models
val_lgbm = model_lgbm.predict(X_val)
val_gb = model_gb.predict_log_proba(X_val)
val_mixed = (val_gb + val_lgbm) / 2

(np.argmax(val_mixed, axis=1) == y_val).sum() / y_val.shape[0]
test_lgbm = np.argmax(model_lgbm.predict(df_test), axis=1)
test_id = pd.DataFrame(test_id)
submission = test_id.join(pd.DataFrame(test_lgbm, columns=['AdoptionSpeed']))
submission.to_csv('submission.csv', index=False)

