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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_red = pd.read_csv('../input/wine-quality-dataset/winequality-red.csv', sep=';')

df_red.head()
df_white = pd.read_csv('../input/wine-quality-dataset/winequality-white.csv', sep=';')

df_white.head()
df_red['color'] = 'red'
df_white['color'] = 'white'
plt.figure(figsize=(12,6)) #let's have a look at the relationship between variables in red wine data

sns.heatmap(df_red.corr(),cmap='inferno', annot=True) #it seems there's no variable have a strong relationship with dependant 

                                                      #variable-y (quality)but there are some between independants
sns.scatterplot(data=df_red, x='fixed acidity', y='pH')
sns.scatterplot(data=df_red, x='fixed acidity', y='density')
sns.scatterplot(data=df_red, x='fixed acidity', y='citric acid')
plt.figure(figsize=(12,6)) #same as we did for red wine data

sns.heatmap(df_white.corr(),cmap='rainbow', annot=True)
sns.scatterplot(data=df_white.drop(df_white.index[df_white['density'] > 1.005]), x='density', y='alcohol')
sns.scatterplot(data=df_white.drop(df_white.index[df_white['density'] > 1.01]), x='density', y='residual sugar')
diff_corr = df_red.corr() - df_white.corr() #we see the biggest difference on the relationship between 

                                            #residual sugar&alcohol
plt.figure(figsize=(12,6))

sns.heatmap(diff_corr, cmap='viridis', annot=True)
df = pd.concat([df_red, df_white]) #combining red&white datasets

df
df.info() #there are no missing variables and all of them are float or integers except for color
df.describe().transpose()
df_miss = df.drop('color', axis=1)
for i in df_miss.columns: #let's have a look at the distribution of variables

    print(sns.distplot(df_miss[i]))

    plt.show()
df['quality'].value_counts().plot.bar() 
df[df['color'] == 'red']['quality'].value_counts().plot.bar()
df[df['color'] == 'white']['quality'].value_counts().plot.bar()
df[(df['color'] == 'white') & (df['quality'] == 9)] #wondered the ones have highest score
for i in df.drop('color', axis=1).columns: #for checking outliers, whiskers and median

    plt.figure(figsize=(8,4))              #after that we may remove outliers if we need

    sns.boxplot(df.drop('color', axis=1)[i])

    plt.show()
df = df.drop(df.index[df['free sulfur dioxide'] > 250], axis=0)
df = df.drop(df.index[df['density'] > 1.03], axis=0)
df = df.drop(df.index[df['sulphates'] > 1.40], axis=0)
sns.boxplot(x=df['sulphates'], data=df)
sns.boxplot(x=df['free sulfur dioxide'], data=df)
sns.boxplot(x=df['density'], data=df)
df = df.drop(df.index[df['density'] > 1.005], axis=0)
df = df.drop_duplicates() #removing duplicates
len(df[df['color'] == 'red'])
len(df[df['color'] == 'white'])
plt.figure(figsize=(10,6)) #looking for the relationship between variables

sns.heatmap(data=df.corr(), cmap='coolwarm', annot=True)
plt.figure(figsize=(10,6))

sns.scatterplot(data=df, x='alcohol', y='density', hue='color', alpha=0.15)
plt.figure(figsize=(8,6))

sns.scatterplot(data=df, x='free sulfur dioxide', y='total sulfur dioxide', hue='color', alpha=0.1)
for i in df.drop('color', axis=1).columns:

    plt.figure(figsize=(8,4))

    sns.boxplot(df.drop('color', axis=1)[i])

    plt.show()
df['color2'] = df['color'].map({'white':1, 'red':2}) #changed into numerical variable
df.columns
df = df.drop(df.index[df['volatile acidity'] > 1.4], axis=0) #removing outliers again
df = df.drop(df.index[df['citric acid'] > 1.00], axis=0)
df = df.drop(df.index[df['residual sugar'] > 21], axis=0)
df = df.drop(df.index[df['chlorides'] > 0.45], axis=0)
df = df.drop(df.index[df['alcohol'] > 14.5], axis=0)
for i in df.drop('color', axis=1).columns:

    plt.figure(figsize=(8,4))

    sns.boxplot(df.drop('color', axis=1)[i])

    plt.show()
df = df.drop('color', axis=1)
X = df.drop('quality', axis=1)

y = df['quality']
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import RobustScaler #we still have outliers 
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression #There will be two different models; linear regression and svm
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
lm.coef_
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test (True Values)')

plt.ylabel('Predicted Values')
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test,predictions))
print('MSE: ', metrics.mean_squared_error(y_test,predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
from sklearn.metrics import r2_score
metrics.explained_variance_score(y_test, predictions) #too low 
from sklearn.model_selection import GridSearchCV #beginning of svm 

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
model = SVC()
model.fit(X_train, y_train)
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=5)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions)) 

print('\n')                                      

print(classification_report(y_test,grid_predictions))