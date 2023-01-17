import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from  sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, QuantileTransformer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import optuna

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv', index_col='Id')

test=pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv', index_col='Id')

submission=pd.read_csv('/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv', index_col='Id')
Ytrain=train['Cover_Type']

train=train[list(test)]
train.isnull().sum()
test.isnull().sum()
for col in ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',

            'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 

            'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 

            'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 

            'Soil_Type40']:

    if train[col].nunique() != test[col].nunique():

        print(f'Different Values train-test : {col}')
train=train.drop(columns=['Soil_Type7','Soil_Type15'])

test=test.drop(columns=['Soil_Type7','Soil_Type15'])
sns.countplot(Ytrain)

Ytrain.value_counts()
train.head()
train['Distance_To_Hydrology']=np.sqrt((train['Horizontal_Distance_To_Hydrology'] **2)  + (train['Vertical_Distance_To_Hydrology'] **2))

test['Distance_To_Hydrology']=np.sqrt((test['Horizontal_Distance_To_Hydrology'] **2)  + (test['Vertical_Distance_To_Hydrology'] **2))





figure, (ax1,  ax2) = plt.subplots(nrows=1, ncols=2)

figure.set_size_inches(14,6)



sns.pointplot(data=train, x=Ytrain, y='Distance_To_Hydrology', ax=ax1)

sns.distplot(train['Distance_To_Hydrology'], ax=ax2)
print(f"Num_Kind of plants living in water? : {Ytrain.loc[train.loc[train['Distance_To_Hydrology']==0].index].nunique()}")
sns.distplot(np.log(train['Distance_To_Hydrology']+1), fit=norm)

train['Distance_To_Hydrology']=np.log(train['Distance_To_Hydrology']+1)

test['Distance_To_Hydrology']=np.log(test['Distance_To_Hydrology']+1)





train.loc[train['Distance_To_Hydrology']==0,'Distance_To_Hydrology']=np.nan

test.loc[test['Distance_To_Hydrology']==0,'Distance_To_Hydrology']=np.nan
qut=QuantileTransformer(output_distribution='normal')

pipeline=make_pipeline(qut, PCA(n_components=1))
train.corr()
sns.scatterplot(data=train, x='Elevation', y='Horizontal_Distance_To_Roadways', hue=Ytrain)
train['Elevation_Roadways']=pipeline.fit_transform(train[['Elevation','Horizontal_Distance_To_Roadways']])

test['Elevation_Roadways']=pipeline.transform(test[['Elevation','Horizontal_Distance_To_Roadways']])
alpha=train[['Hillshade_9am','Hillshade_3pm']]

alpha2=test[['Hillshade_9am','Hillshade_3pm']]
train['Hillshade9am_Hillshade_3pm']=pipeline.fit_transform(alpha)

test['Hillshade9am_Hillshade_3pm']=pipeline.transform(alpha2)
sns.lineplot(data=train, x=Ytrain, y='Hillshade9am_Hillshade_3pm')
alpha=train[['Aspect','Hillshade9am_Hillshade_3pm']]

alpha2=test[['Aspect','Hillshade9am_Hillshade_3pm']]
train['Aspect_Hillshade']=pipeline.fit_transform(alpha)

test['Aspect_Hillshade']=pipeline.transform(alpha2)
alpha=train[['Slope','Hillshade_Noon']]

alpha2=test[['Slope','Hillshade_Noon']]
train['Slope_Hillshade']=pipeline.fit_transform(alpha)

test['Slope_Hillshade']=pipeline.transform(alpha2)
Xtrain=train

Xtest=test
rf=RandomForestClassifier(n_estimators=5000, random_state=18, n_jobs=-1)

rf.fit(Xtrain.fillna(-999), Ytrain)

feature_importance_df=pd.DataFrame(data=None,  columns=['feature','importances'])

feature_importance_df['importances']=rf.feature_importances_

feature_importance_df['feature']=Xtrain.columns



feature_importance_df=feature_importance_df.sort_values(by='importances', ascending=False)

feature_importance_df.tail()
feature_names=feature_importance_df.head(56)['feature']

Xtrain=train[feature_names]

Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
params={'n_estimators': 1222, 'learning_rate': 0.07307234151834806, 'num_leaves': 96, 'colsample_bytree': 0.8972376156262298, 

        'subsample': 0.9312856106293543, 'min_child_samples': 1}



lightgbm=LGBMClassifier(random_state=18, subsample_freq=1, silent=False, **params)
lightgbm.fit(Xtrain, Ytrain)

predictions=lightgbm.predict(Xtest)

submission['Cover_Type']=predictions

submission.to_csv('LGBSingleModel.csv')

submission.head()