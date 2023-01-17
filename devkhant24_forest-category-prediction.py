# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import xgboost

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import  f_classif



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
df.drop('Id',axis=1,inplace=True)
dft = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
dft.drop('Id',axis=1,inplace=True)
xg = xgboost.XGBClassifier()

lg = LogisticRegression()

dt = DecisionTreeClassifier(random_state=1)

rf = RandomForestClassifier(random_state=1, criterion='entropy')

svm = SVC(kernel='linear')

nb = GaussianNB()

ss = StandardScaler()

knn = KNeighborsClassifier()
x = df.drop('Cover_Type',axis=1)

y = df['Cover_Type']
bestfeatures = SelectKBest(score_func=f_classif,k=10)

fit = bestfeatures.fit(x,y)
dfscore = pd.DataFrame(fit.scores_)

dfcolums = pd.DataFrame(x.columns)
features = pd.concat([dfcolums,dfscore],axis=1)

features.columns = ['Specs','Score']

print(features.nlargest(20,'Score'))
x = df[['Soil_Type29','Soil_Type30','Horizontal_Distance_To_Hydrology','Hillshade_9am','Soil_Type40','Wilderness_Area3','Soil_Type39'

       ,'Soil_Type38','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Soil_Type3','Soil_Type10','Horizontal_Distance_To_Roadways'

       ,'Wilderness_Area4','Elevation','Slope','Soil_Type4','Soil_Type22','Soil_Type17']]
test = dft[['Soil_Type29','Soil_Type30','Horizontal_Distance_To_Hydrology','Hillshade_9am','Soil_Type40','Wilderness_Area3','Soil_Type39'

       ,'Soil_Type38','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Soil_Type3','Soil_Type10','Horizontal_Distance_To_Roadways'

       ,'Wilderness_Area4','Elevation','Slope','Soil_Type4','Soil_Type22','Soil_Type17']]
x = ss.fit_transform(x)

test = ss.fit_transform(test)
score = cross_val_score(rf,x,y,cv=5)

score.mean()
xg.fit(x,y)
yp = xg.predict(test)

pred = pd.DataFrame(yp)

pred.columns = ['Cover_Type']
c = pd.read_csv('/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv')
submission = pd.concat([c['Id'],pred],axis=1)
submission.to_csv('sampleSubmission.csv',index=False)