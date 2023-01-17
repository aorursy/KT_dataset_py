# import required libs

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# load data

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
# first look at the data

train.head()
test.head()
# Missing values

fig, ax = plt.subplots(1, 2, figsize= (12,6))



# missing values in train

sns.barplot(train.isnull().sum().index, train.isnull().sum()*100/ train.shape[0],\

            palette= 'Blues', ax= ax[0])

#ax[0].xaxis(rotation= 90)

ax[0].set_ylabel('% Missing values')

ax[0].set_title('Missing values in train set')



# missing values in test

sns.barplot(test.isnull().sum().index, test.isnull().sum()*100/ test.shape[0],\

            palette= 'BrBG', ax= ax[1])

#ax[1].set_xticks(rotation= 90)

ax[1].set_ylabel('% Missing values')

ax[1].set_title('Missing values in test set')



fig.autofmt_xdate(rotation=45)
# impute missing values

# first check the dirtribution of these variables

fig, ax = plt.subplots(1, 2, figsize= (12, 4))





# plot of gender

sns.countplot(train['sex'].sort_values(ignore_index= True), 

             color= 'green',

             label= 'Train', 

             ax= ax[0],

             alpha= 0.3)





sns.countplot(test['sex'].sort_values(ignore_index= True), 

             color= 'red',

             label= 'Test', 

             ax= ax[0],

             alpha= 0.3)

ax[0].margins(0.2)

ax[0].legend()





# plot of location

sns.countplot(train['anatom_site_general_challenge'],

             color= 'red',

             label= 'Train', 

             ax= ax[1],

             alpha= 0.3)



sns.countplot(test['anatom_site_general_challenge'],

             color= 'blue',

             label= 'Test', 

             ax= ax[1],

             alpha= 0.3)

ax[1].margins(0.2)

ax[1].legend()



fig.autofmt_xdate(rotation=45)
# distribution of age

fig, ax = plt.subplots(figsize= (12, 4))

sns.distplot(train['age_approx'], color= 'red', ax= ax, label= 'Train')

sns.distplot(test['age_approx'], color= 'black', ax= ax, label= 'Test')

ax.legend()
# Imputing missing values

train['age_approx'].fillna(train['age_approx'].mean(), inplace= True)

train[train['age_approx'] == 0]['age_approx'] == train['age_approx'].mean()

train['sex'].fillna('male', inplace= True)

train['anatom_site_general_challenge'].fillna('unknown', inplace= True)
# sanity check

train.isnull().sum()
# checking duplicates

# duplicate images

uni_img = len(train['image_name'].unique())*100/train.shape[0]

print('Fraction of unique images: {}%'.format(uni_img))



# repeating patients

uni_pat = len(train['patient_id'].unique())*100/train.shape[0]

print('Fraction of unique patients: {}%'.format(uni_pat))
# Remove duplicate images

df_dup = pd.read_csv('../input/siim-list-of-duplicates/2020_Challenge_duplicates.csv')

dups = df_dup[df_dup['partition'] == 'train']['ISIC_id']



train.drop(train[train['image_name'].isin(dups)].index, inplace= True)
# Gender and target

group = train.groupby(['sex', 'benign_malignant']).size().unstack()

group['total'] = group.sum(axis= 1)

group['benign'] = group['benign']/ group['total']

group['malignant'] = group['malignant']/ group['total']

group.drop('total', inplace= True, axis= 1)



group.plot(kind= 'bar')

plt.xlabel('Gender')

plt.ylabel('% of Cancer in each Gender')
# rename columns for convenience

train.columns = ['image_name', 'patient_id', 'sex', 'age', 'location', 'diagnosis', 'target_string', 'target']

test.columns = ['image_name', 'patient_id', 'sex', 'age', 'location']
# Age and target

fig, ax = plt.subplots(1, 2, figsize= (12, 4))



sns.distplot(train[train['target'] == 0].age, color= 'red', label= 'Benign', ax= ax[0],)

sns.distplot(train[train['target'] == 1].age, color= 'black', label= 'Malignant', ax= ax[0])

ax[0].legend()





sns.boxplot(x= 'target_string', y= 'age', data= train, palette= 'cubehelix_r', ax= ax[1])

ax[1].set_xlabel('')

ax[1].set_ylabel('Age')



fig.suptitle('Distribution of Age with Target')

plt.margins(0.2)
# location and target

fig, ax = plt.subplots(figsize= (12, 4))

sns.countplot(x= 'location', data= train, hue= 'target_string', ax= ax, palette= 'gist_heat_r')

ax.set_xlabel('Anatom Site')

ax.legend(('Malignant', 'Benign'))

fig.autofmt_xdate()

plt.margins(0.2)

# diagnosis and target

fig, ax = plt.subplots(figsize= (12, 4))

sns.countplot(x= 'diagnosis', data= train, hue= 'target_string', ax= ax, palette= 'gist_ncar_r')

ax.set_xlabel('Diagnosis')

ax.legend(('Malignant', 'Benign'))

fig.autofmt_xdate()

plt.margins(0.2)
# sex and age

fig, ax = plt.subplots(figsize= (10, 4))

sns.boxplot(x= 'sex', y= 'age', data= train, hue= 'target_string', ax= ax, palette= 'gist_rainbow')

plt.margins(0.2)
fig, ax = plt.subplots(figsize= (12, 6))

group = train.groupby(['location', 'diagnosis']).size().unstack().fillna(0).T

group.plot( kind= 'bar',ax= ax, color= sns.color_palette('rocket'))

fig.autofmt_xdate()

plt.xlabel('Diagnosis')

plt.ylabel('Count')
# import libraries

import os

import gc

import cv2

from tqdm import tqdm

from skimage import measure, color

from skimage.filters import threshold_otsu
img_dir_train = '../input/siim-isic-melanoma-classification/jpeg/train'

img_dir_test = '../input/siim-isic-melanoma-classification/jpeg/test'



train_imgs = os.listdir(img_dir_train)

test_imgs = os.listdir(img_dir_test)
"""# Extract features from train images

train['image_pixels'] = np.zeros(train.shape[0])

train['width'] = np.zeros(train.shape[0])

train['height'] = np.zeros(train.shape[0])

train['red'] = np.zeros(train.shape[0])

train['green'] = np.zeros(train.shape[0])

train['blue'] = np.zeros(train.shape[0])

train['mean_color'] = np.zeros(train.shape[0])

#train['contours'] = np.zeros(train.shape[0])



for i in tqdm(range(train.shape[0])):

    row = train.iloc[i, :]

    image_name = row['image_name'] + '.jpg'

    

    # read image

    img_path = img_dir_train + '/' + image_name

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

    

    # total pixels

    row['image_pixels'] = img.size

    

    # image dimensions

    row['width'] = img.shape[1]

    row['height'] = img.shape[0]

    

    # RGB channels

    row['red'] = img[:, : , 0].mean()

    row['green'] = img[:, :, 1].mean()

    row['blue'] = img[:, :, 2].mean()

    row['mean_color'] = img.mean()

    

    # number of contours

    img_gray = color.rgb2gray(img)

    thresh = threshold_otsu(img_gray)

    thresholded_image = img_gray > thresh

    contours = measure.find_contours(thresholded_image, 0.8)#too slow

    

    train.iloc[i, :] = row

    del(img)

    del(row)

    gc.collect()

    #del(imgage_gray)

    #del(thresholded_image)"""
"""# Extract image features in test set

test['image_pixels'] = np.zeros(test.shape[0])

test['width'] = np.zeros(test.shape[0])

test['height'] = np.zeros(test.shape[0])

test['red'] = np.zeros(test.shape[0])

test['green'] = np.zeros(test.shape[0])

test['blue'] = np.zeros(test.shape[0])

test['mean_color'] = np.zeros(test.shape[0])

#test['contours'] = np.zeros(test.shape[0])



for i in tqdm(range(test.shape[0])):

    row = test.iloc[i, :]

    image_name = row['image_name'] + '.jpg'

    

    # read image

    img_path = img_dir_test + '/' + image_name

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

    

    # total pixels

    row['image_pixels'] = img.size

    

    # image dimensions

    row['width'] = img.shape[1]

    row['height'] = img.shape[0]

    

    # RGB channels

    row['red'] = img[:, : , 0].mean()

    row['green'] = img[:, :, 1].mean()

    row['blue'] = img[:, :, 2].mean()

    row['mean_color'] = img.mean()

    

    # number of contours

    img_gray = color.rgb2gray(img)

    thresh = threshold_otsu(img_gray)

    thresholded_image = img_gray > thresh

    contours = measure.find_contours(thresholded_image, 0.8)#too slow

    

    test.iloc[i, :] = row

    del(img)

    del(row)

    gc.collect()

    #del(imgage_gray)

    #del(thresholded_image)"""
test = pd.read_csv('../input/modified-data/modified_test.csv')

train = pd.read_csv('../input/modified-train/modified_train.csv')

test.drop(test.columns[0], axis= 1, inplace= True)

train.drop(train.columns[15], axis= 1, inplace= True)
# import req. libraries

import xgboost as xgb

from xgboost import plot_importance



from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

from sklearn.metrics import roc_auc_score
# prepare data for model



# dummy vairables for categorical features

# sex

dummies = pd.get_dummies(train['sex'], prefix= 'sex')

train = pd.concat([train, dummies], axis= 1)



dummies = pd.get_dummies(test['sex'], prefix= 'sex')

test = pd.concat([test, dummies], axis= 1)



# location

dummies = pd.get_dummies(train['location'], prefix= 'anatom')

train = pd.concat([train, dummies], axis= 1)



dummies = pd.get_dummies(test['location'], prefix= 'anatom')

test = pd.concat([test, dummies], axis= 1)





# remvoe redundant columns

X = train.drop(['target_string', 'target', 'image_name', 'patient_id', 'sex', 'location', 'diagnosis'], axis= 1)

y = train['target']



test.drop(['image_name', 'patient_id', 'sex', 'location'],axis= 1, inplace= True)
# cross validation setup

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size= 0.2, random_state= 20)



cv = StratifiedKFold(5, random_state= 20, shuffle= True)
# 5 folds cross-validation

cv_result = []

for i,( train_idx, test_idx) in tqdm(enumerate(cv.split(X_train, y_train))):

    X_train_cv, y_train_cv = X_train.iloc[train_idx], y_train.iloc[train_idx]

    X_test_cv, y_test_cv = X_train.iloc[test_idx], y_train.iloc[test_idx]

    

    

    # model setup

    clf = xgb.XGBClassifier(n_estimators= 100000,

                           max_depth= 2,

                           learning_rate= 0.001,

                           n_jobs= -1,

                           subsample= 0.6,

                           colsample_bytree= 0.8,

                           colsample_bynode= 0.8,

                            random_state= 20,

                            gamma=0.10

                           )

    # fit model

    clf.fit(X_train_cv, y_train_cv, eval_set= [(X_test_cv, y_test_cv)],

           eval_metric= 'auc', early_stopping_rounds= 100)

    

    # predict with validation set

    roc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    cv_result.append(roc_score)

    

    # save model

    clf.save_model('XGB_{}.txt'.format(i))
# cv_results for test set

print('Folds roc: ', cv_result)

print('Mean roc: ', np.mean(cv_result))
test['anatom_unknown'] = np.zeros(len(test))
X_train.head()
# predict on test set

img_name = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')['image_name']

predictions = pd.DataFrame({'image_name': img_name})

for i in tqdm(range(5)):

    # load model

    clf = xgb.XGBClassifier(n_estimators= 10000,

                           max_depth= 2,

                           learning_rate= 0.001,

                           n_jobs= -1,

                           subsample= 0.6,

                           colsample_bytree= 0.8,

                           colsample_bynode= 0.8,

                            random_state= 20,

                            gamma=0.10

                           )

    clf.load_model('XGB_{}.txt'.format(i))

    

    predictions['pred_{}'.format(i)] = clf.predict_proba(test)[:, 1]
# submission

predictions['target'] = predictions.filter(regex='^pred').sum(axis= 1)/ 5

submission_xgb = predictions[['image_name', 'target']]



submission_xgb.to_csv('submission_xgb.csv', index= False)