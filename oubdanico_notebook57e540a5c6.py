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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
dataset.head(5)
df = dataset.copy()
df.describe()
df.shape
df.isna().sum()
df.dtypes.value_counts().plot.pie()
df.drop('Serial No.', axis=1)
for col in df.columns:
    plt.figure()
    sns.distplot(df[col])
plt.figure()
sns.distplot(df['Chance of Admit '])
plt.figure()
sns.relplot(x="TOEFL Score",y="Chance of Admit ",data=df)
sns.pairplot(df)
df1 = dataset.copy()
df1 = df1.drop('Serial No.',axis=1)
trainset, testset = train_test_split(df1, test_size=0.2)
trainset.shape
testset.shape
from sklearn.preprocessing import StandardScaler

def normalisation(df):
    
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
  
    return df
def preprocessing(df):
    
    X = df.drop('Chance of Admit ', axis=1)
    y = df['Chance of Admit ']
    X = normalisation(X)
    
    return X,y
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)
from sklearn.metrics import mean_squared_error
def evaluation(model):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    score = model.score(X_test, y_test)
    
    print('La performance du modèle sur la base de test')
    print('--------------------------------------')
    print('Mean quadratic error {}'.format(rmse))
    print('Score R2  {}'.format(score))
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

LinearRegression = LinearRegression()
DecisionTreeRegressor = DecisionTreeRegressor(random_state=0)
SVR = SVR()
RandomForestRegressor = RandomForestRegressor()

dict_models = {'LinearRegression': LinearRegression,
               'DecisionTreeRegressor': DecisionTreeRegressor, 
               'SVR':SVR,
               'RandomForestRegressor': RandomForestRegressor}
for name, model in dict_models.items():
    
    print("========================================")
    print(name)
    print("========================================")
    evaluation(model)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

LinearRegression = LinearRegression()
hyper_params = {'normalize': [True,False],
               'fit_intercept': [True,False],
               'n_jobs': range(-1,1)}
              
grid = GridSearchCV(LinearRegression,hyper_params, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)


rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
score = grid.score(X_test, y_test)

print('FINAL MODEL=====>  RANDOM FOREST REGRESSOR')
print('Mean quadratic error {}'.format(rmse))
print('Score R2  {}'.format(score))
    
print('Your chances are {}%'.format(round(model.predict([[305, 108, 4, 4.5, 4.5, 8.35, 0]])[0]*100, 1)))