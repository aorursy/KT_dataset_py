# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1 = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv',index_col='Serial No.')

data2 = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col='Serial No.')
df = pd.concat([data1, data2])

df.head()
df.describe()
df.info()
#Import Seaborn for visualization

import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df['GRE Score'],kde=False)

plt.title('GRE Score Distribution')

plt.show()
sns.distplot(df['TOEFL Score'],kde=False)

plt.title('TOEFL Score Distribution')

plt.show()
sns.distplot(df['University Rating'],kde=False)

plt.title('University Ranking Distribution')

plt.show()
sns.distplot(df['SOP'],kde=False)

plt.title('SOP Distribution')

plt.show()
sns.distplot(df['LOR '],kde=False)

plt.title('LOR Distribution')

plt.show()
sns.distplot(df['CGPA'],kde=False)

plt.title('CGPA Distribution')

plt.show()
sns.distplot(df['Research'],kde=False)

plt.title('Research Distribution')

plt.show()
sns.distplot(df['Chance of Admit '],kde=False)

plt.title('Chances Distribution')

plt.show()
plt.figure(figsize=(10,10))

sns.regplot(x='GRE Score',y='Chance of Admit ',data=df)

plt.title('GRE Score vs Change of Admit')
plt.figure(figsize=(10,10))

sns.regplot(x='TOEFL Score',y='Chance of Admit ',data=df)

plt.title('TOEFL Score vs Change of Admit')
plt.figure(figsize=(10,10))

sns.regplot(x='CGPA',y='Chance of Admit ',data=df)

plt.title('CGPA vs Change of Admit')
plt.figure(figsize=(20,10))

sns.swarmplot(y='Chance of Admit ',x='University Rating',data=df)

plt.yticks(rotation=90)

plt.title('University Ranking vs Chance of Admit')

plt.show()
plt.figure(figsize=(20,10))

sns.swarmplot(y='Chance of Admit ',x='SOP',data=df)

plt.yticks(rotation=90)

plt.title('SOP vs Chance of Admit')

plt.show()
plt.figure(figsize=(20,10))

sns.swarmplot(y='Chance of Admit ',x='LOR ',data=df)

plt.yticks(rotation=90)

plt.title('LOR vs Chance of Admit')

plt.show()
plt.figure(figsize=(20,10))

sns.swarmplot(y='Chance of Admit ',x='Research',data=df)

plt.yticks(rotation=90)

plt.title('Research vs Chance of Admit')

plt.show()
plt.figure(figsize=(8,12))



heatmap = sns.heatmap(df.corr()[['Chance of Admit ']].sort_values(by='Chance of Admit ',ascending=False),

                                     vmin=-1,vmax=1,annot=True,cmap='BrBG')

heatmap.set_title('Features coorelating with Chance of Admit')
from sklearn.model_selection import train_test_split



X = df.drop(['Chance of Admit '],axis=1)

y = df['Chance of Admit ']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.metrics import accuracy_score,mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LogisticRegression,SGDRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor
models = [['Decision Tree:',DecisionTreeRegressor()],

         ['Linear Regression:',LinearRegression()],

         ['Random Forest:',RandomForestRegressor()],

         ['KNeighbors:',KNeighborsRegressor(n_neighbors=2)],

         ['SVM:',SVR()],

         ['Ada Boost Classifier:',AdaBoostRegressor()],

         ['Gradient Boost Classifier:',GradientBoostingRegressor()],

         ['Xgboost:',XGBRegressor()],

         ['Cat Boost:',CatBoostRegressor(logging_level='Silent')],

         ['Lasso:',Lasso()],

         ['Ridge:',Ridge()],

         ['Elastic Net:',ElasticNet()],

         ['SGD Regressor:',SGDRegressor()]]



for name,model in models:

    model = model

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print(name,(np.sqrt(mean_squared_error(y_test,pred))))
classifier = RandomForestRegressor()

classifier.fit(X,y)

feature_name = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'],ascending=True)
plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()