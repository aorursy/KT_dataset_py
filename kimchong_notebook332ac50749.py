# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import folium

import squarify

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Data = pd.read_csv('/kaggle/input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')
Data.info()
Columns = Data.columns
Data.head()
plt.plot( Data['X'], Data['Y'])

plt.show()
Data.info()
Data['Category'].unique()
Data['Resolution'].unique()
RL_None = Data[Data['Resolution'] == 'NONE'] # Resolution이 None 인 상태 즉 처벌 받지 않거나 해결되지 않은 미제 사건 

RL_NN   = Data[Data['Resolution'] != 'NONE'] # Resolution이 처리된 사건 
RL_None.info()
RL_None.head()
RL_None['Category'].unique()
plt.figure(figsize = (20,20)) 

plt.scatter( RL_None['X'], RL_None['Y'], s = 1)

plt.show()
Data['Category'].unique()
RL_NN.info()
RL_NN['Category'].unique()
plt.figure(figsize = (20,20)) 

plt.scatter( RL_NN['X'], RL_NN['Y'], s = 1)

plt.show()
plt.figure(figsize = (20,20)) 

plt.scatter( RL_None['X'], RL_None['Y'], s = 1)

plt.scatter( RL_NN['X'], RL_NN['Y'], s = 1, c = 'r')

plt.show()
#중요 범죄 여부에 따라서 시간때 별 피해가야할 장소, 목적의식 생각 
Data.isnull().sum()
Data['PdDistrict'].fillna(Data['PdDistrict'].mode()[0], inplace = True)
CateG = Data['Category'].value_counts()

    

plt.rcParams['figure.figsize'] = (20, 20)

plt.style.use('fivethirtyeight')



color = plt.cm.magma(np.linspace(0, 1, 20))

squarify.plot(sizes = CateG.values, label = CateG.index, alpha=.8, color = color)

plt.title('Tree Map for Crimes', fontsize = 15)



plt.axis('off')

plt.show()
RS_D = Data['Resolution'].value_counts()

    

plt.rcParams['figure.figsize'] = (20, 20)

plt.style.use('fivethirtyeight')



color = plt.cm.magma(np.linspace(0, 1, 20))

squarify.plot(sizes = RS_D.values, label = RS_D.index, alpha=.8, color = color)

plt.title('Tree Map for Resolution', fontsize = 15)



plt.axis('off')

plt.show()
TrainData = Data

TrainData['Resolution'] = Data.apply(lambda x : 0 if x['Resolution'] == 'NONE' else 1, axis = 1)
TrainData.head()
#location, Pdld제거 진행

TrainData.drop(['Location', 'PdId'] , axis = 1, inplace = True)
# TrainData_One = pd.get_dummies(TrainData, columns = ['PdDistrict']) # 의미가 크게 없다고 본다. x,y 의 데이터가 유사하기에 
TrainData.drop(['PdDistrict'] , axis = 1, inplace = True)
TrainData.drop(['IncidntNum', 'Descript', 'Date'] , axis = 1, inplace = True)
TrainData.drop(['Address'] , axis = 1, inplace = True)
TrainData.head()
TrainData_One = pd.get_dummies(TrainData, columns = ['Category','DayOfWeek'])
TrainData_One.drop(['Time' ] , axis = 1, inplace = True)
TrainData_One.head()
X = TrainData_One.drop(['Resolution' ] , axis = 1, inplace = False)

Y = TrainData_One['Resolution' ]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, random_state = 0)
def get_model_predict(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    pred = model.score(X_test, y_test)

    print('###',model.__class__.__name__,'###')

    print('예상 정확도 : ' ,pred)
KN_MD = KNeighborsClassifier()

RF_MD = RandomForestClassifier()

GB_MD = GradientBoostingClassifier()

XG_MD = XGBClassifier()

# LG_MD = LGBMClassifier()

for _MD in [KN_MD, RF_MD, GB_MD, XG_MD]:

    get_model_predict ( _MD, trainX, testX, trainY, testY)