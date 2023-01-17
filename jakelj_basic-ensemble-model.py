

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import re

import sklearn

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.experimental import enable_hist_gradient_boosting

# Going to use these  base models for the stacking

from sklearn.ensemble import (  BaggingClassifier, 

                              ExtraTreesClassifier,  HistGradientBoostingClassifier)



from tqdm import tqdm

from mlxtend.classifier import StackingCVClassifier

from lightgbm import LGBMClassifier

from itertools import combinations, chain

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')

ID = test['Id'] # save this for later 
train.head()
train.columns
train.dtypes.unique()
#equal amounts of each variable in the train set

sns.distplot(train.Cover_Type, kde = False)
fig, axs = plt.subplots(ncols = 2,figsize=(15,5))



sns.boxplot(train.Cover_Type,train.Elevation, whis = 4, hue = train.Cover_Type, ax=axs[0])

sns.boxplot(train.Cover_Type,train.Slope, whis = 4, hue = train.Cover_Type, ax=axs[1])

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=False, cmap=colormap, linecolor='white', annot=False)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(test.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=False, cmap=colormap, linecolor='white', annot=False)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=5)

sns.heatmap(train.corr()[['Cover_Type']].sort_values(by =['Cover_Type'], ascending=False),vmin=-1, 

            square=False, cmap=colormap, linecolor='white', annot=False)
target = train.Cover_Type

train = train.drop(['Cover_Type'], axis = 1)




train = train.drop(['Soil_Type7','Soil_Type15','Id',], axis =1)

test = test.drop(['Soil_Type7','Soil_Type15','Id'], axis =1)

train['EV_DTH'] = (train.Elevation - train.Vertical_Distance_To_Hydrology)

test['EV_DTH'] = (test.Elevation - test.Vertical_Distance_To_Hydrology)





train['EH_DTH'] = (train.Elevation -  (train.Horizontal_Distance_To_Hydrology *0.2))

test['EH_DTH'] = (test.Elevation -  (test.Horizontal_Distance_To_Hydrology *0.2))



train['Dis_To_Hy'] = (((train.Horizontal_Distance_To_Hydrology **2) + (train.Vertical_Distance_To_Hydrology **2))**0.5)

test['Dis_To_Hy'] = (((test.Horizontal_Distance_To_Hydrology **2) + (test.Vertical_Distance_To_Hydrology **2))**0.5)



train['HyF_1'] = (train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points)

test['HyF_1'] = (test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Fire_Points)



train['HyF_2'] = (train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)

test['HyF_2'] = (test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Fire_Points)



train['HyR_1'] = (train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)

test['HyR_1'] = (test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)



train['HyR_2'] = (train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)

test['HyR_2'] = (test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Roadways)





train['FiR_1'] = (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)

test['FiR_1'] = (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Roadways)



train['FiR_1'] = (train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)

test['FiR_1'] = (test.Horizontal_Distance_To_Fire_Points - test.Horizontal_Distance_To_Roadways)



train['Avg_shade'] = ((train.Hillshade_9am + train.Hillshade_Noon + train.Hillshade_3pm) /3)

test['Avg_shade'] = ((test.Hillshade_9am + test.Hillshade_Noon + test.Hillshade_3pm) /3)



train['Morn_noon_int'] = ((train.Hillshade_9am + train.Hillshade_Noon) / 2)

test['Morn_noon_int'] = ((test.Hillshade_9am + test.Hillshade_Noon) / 2)



train['noon_eve_int'] = ((train.Hillshade_3pm + train.Hillshade_Noon) / 2)

test['noon_eve_int'] = ((test.Hillshade_3pm + test.Hillshade_Noon) / 2)



train['Slope2'] = np.sqrt(train.Horizontal_Distance_To_Hydrology**2 + train.Vertical_Distance_To_Hydrology**2)

test['Slope2'] = np.sqrt(test.Horizontal_Distance_To_Hydrology**2 + test.Vertical_Distance_To_Hydrology**2)



#Specify the model and parameters and then use the training data to fit the model

gm = GaussianMixture(n_components  = 15)

gm.fit(train)
train['g_mixture'] = gm.predict(train)

test['g_mixture'] = gm.predict(test)


y = target

X = train

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=1)
clf_lgbm = LGBMClassifier(n_estimators=400,num_leaves=100,verbosity=0)

clf_knc = KNeighborsClassifier(n_jobs = -1, n_neighbors =1)

clf_etc = ExtraTreesClassifier(random_state = 1, n_estimators = 900, max_depth =50,max_features = 30)

clf_hbc = HistGradientBoostingClassifier(random_state = 1, max_iter = 500, max_depth =25)
clf_knc.fit(x_train,y_train)

testing_predictions = clf_knc.predict(x_test)



print(round(accuracy_score(y_test, testing_predictions),4))
ensemble = [

            ('clf_knc', clf_knc),

            ('clf_hbc', clf_hbc),

            ('clf_etc', clf_etc),

            ('clf_lgbm', clf_lgbm)

           

           ]



stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],

                             meta_classifier=clf_lgbm,

                             cv=3,

                             use_probas=True, 

                             use_features_in_secondary=True,

                             verbose=-1,

                             n_jobs=-1)



stack = stack.fit(X,y)

predictions = stack.predict(test)
stack.predict_proba(test)
pd.Series(predictions).value_counts()
submission = pd.DataFrame({ 'Id': ID,

                            'Cover_Type': predictions })

submission.to_csv("submission_example.csv", index=False)