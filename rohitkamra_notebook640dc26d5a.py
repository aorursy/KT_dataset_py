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
csv_path = '/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(csv_path)
df
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
df.drop(['Serial No.'],axis=1,  inplace=True)
df.info()
df.isnull().sum()
%matplotlib inline
matrix = df.corr()
g = sns.heatmap(matrix, cmap='RdYlGn_r', annot=True)
df.columns
plotColumns = ['GRE Score', 'TOEFL Score','CGPA','Chance of Admit ']
sns.pairplot(df[plotColumns])
plt.figure(figsize=(25,25))
sns.barplot(y = 'GRE Score', x = 'Chance of Admit ', data = df)
plt.ylim([300,350])
plt.figure(figsize=(10,10))
sns.boxplot(x = 'LOR ', y = 'Chance of Admit ', data = df)
plt.figure(figsize=(15,10))
sns.lineplot(x="SOP",y="Chance of Admit ",data=df, label="SOP")
sns.lineplot(x="LOR ",y="Chance of Admit ",data=df, label="LOR")
sns.lineplot(x="University Rating",y="Chance of Admit ",data=df, label="Research")

df.hist(color = 'green', figsize=(15,15), bins = 30)
from sklearn.preprocessing import LabelEncoder
cat_features = ['LOR ', 'SOP', 'University Rating']
lblEncoder = {}
for feature in cat_features:
    lblEncoder[feature] = LabelEncoder()
    df[feature] = lblEncoder[feature].fit_transform(df[feature])
df
X = df.drop(['Chance of Admit '], axis =1)
y = df['Chance of Admit ']


from sklearn import model_selection
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size = 0.3)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(xtrain, ytrain)
y_pred = lin_reg.predict(xtest)

from sklearn import metrics
rmse = metrics.mean_squared_error(y_pred, ytest, squared=False)
rmse
r2 = metrics.r2_score(y_pred, ytest)
r2
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(xtrain)
X_test = sc.transform(xtest)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_train, ytrain)
y_pred2 = lin_reg2.predict(X_test)
lin_reg2.score(X_test, ytest)
plt.plot(ytest,y_pred2, 'o', color = 'g')
plt.xlabel('test')
plt.ylabel('prediciton')
