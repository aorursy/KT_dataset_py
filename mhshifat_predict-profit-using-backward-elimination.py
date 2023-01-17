# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/startup-logistic-regression/50_Startups.csv")
dataset.head()
dataset.isna().any()
plt.figure(figsize=(10,8))
g=sns.distplot(dataset['R&D Spend'],label='R&D Spend')
plt.title('R&D Spend \n Median:{0:.1f} \n Mean:{1:.1f}'.format(dataset['R&D Spend'].median(),
                                                                            dataset['R&D Spend'].mean()),size=25)
l1=plt.axvline(dataset['R&D Spend'].median(),color='black',label='Median score')
plt.legend()
plt.show()

g=sns.lmplot('R&D Spend','Profit',data=dataset,order=1,line_kws={'color':'red','linewidth':2.5},
           height=5,aspect=2,scatter_kws={'s':50,'alpha':0.4})
plt.title('R&D Spend Vs Profit',size=30)

plt.axhline(0.8,color='black',alpha=0.2)
plt.axvline(324,color='black',alpha=0.2)
plt.xticks(np.arange(280,365,5))
plt.show()
plt.figure(figsize=(10,8))
g=sns.distplot(dataset['Administration'],label='Administration')
plt.title('Administration \n Median:{0:.1f} \n Mean:{1:.1f}'.format(dataset['Administration'].median(),
                                                                            dataset['Administration'].mean()),size=25)
l1=plt.axvline(dataset['Administration'].median(),color='black',label='Median score')
plt.legend()
plt.show()
g=sns.lmplot('Administration','Profit',data=dataset,order=1,line_kws={'color':'red','linewidth':2.5},
           height=5,aspect=2,scatter_kws={'s':50,'alpha':0.4})
plt.title('Administration Vs Profit',size=30)

plt.axhline(0.8,color='black',alpha=0.2)
plt.axvline(324,color='black',alpha=0.2)
plt.xticks(np.arange(280,365,5))
plt.show()
plt.figure(figsize=(10,8))
g=sns.distplot(dataset['Marketing Spend'],label='Marketing Spend')
plt.title('Marketing Spend \n Median:{0:.1f} \n Mean:{1:.1f}'.format(dataset['Marketing Spend'].median(),
                                                                            dataset['Marketing Spend'].mean()),size=25)
l1=plt.axvline(dataset['Marketing Spend'].median(),color='black',label='Median score')
plt.legend()
plt.show()
g=sns.lmplot('Marketing Spend','Profit',data=dataset,order=1,line_kws={'color':'red','linewidth':2.5},
           height=5,aspect=2,scatter_kws={'s':50,'alpha':0.4})
plt.title('Marketing Spend Vs Profit',size=30)

plt.axhline(0.8,color='black',alpha=0.2)
plt.axvline(324,color='black',alpha=0.2)
plt.xticks(np.arange(280,365,5))
plt.show()
x = dataset.drop('Profit',axis=1).values
x
y = dataset['Profit'].values
y
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers = [
        ("asda",
        OneHotEncoder(),
        [3]
        )
    ],
    remainder = 'passthrough'
)

x = transformer.fit_transform(x)

x
x=x[:,1:]
x
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)
x_train
y_train
x_test
y_test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
 
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))  
x = np.append(arr = np.ones((50,1)).astype(int),values=x,axis=1)
x
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
X_Modeled = backwardElimination(X_opt, SL)
X_Modeled
dataset
x_train,x_test,y_train,y_test = tts(X_Modeled[:,1:],y,test_size=0.2,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test)) 