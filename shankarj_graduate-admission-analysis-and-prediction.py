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
data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv',index_col = 'Serial No.')
data.head()
data.describe()
data.info()
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax = plt.subplots(4,2,figsize = (15,18))
features = list(data.columns)
x = 0
y = 0
sns.set_style('whitegrid')
for feature in features:
    sns.distplot(a = data[feature],kde = False,ax = ax[x,y])
    if y==1:
        x +=1
        y = 0
    else:y+=1
print(features)
features = ['GRE Score', 'TOEFL Score', 'University Rating','CGPA']
fig,ax = plt.subplots(3,2,figsize = (15,18))
x,y = 0,0
sns.set_style('darkgrid')
for i in range(len(features)-1):
    for j in range(i+1,len(features)):
        sns.regplot(x = data[features[i]],y = data[features[j]],ax =ax[x,y] )
        if y==1:
            x+=1
            y=0
        else:y+=1
admit_percent = [">=50%" if x>=0.5 else "<50%" for x in data['Chance of Admit ']]
admit_percent[:10]
data['Admission Percentage'] = admit_percent
fig,ax = plt.subplots(3,2,figsize = (15,18))
x,y = 0,0
sns.set_style('darkgrid')
for i in range(len(features)-1):
    for j in range(i+1,len(features)):
        sns.scatterplot(x = data[features[i]],y = data[features[j]],hue = data['Admission Percentage'],ax = ax[x,y])
        if y==1:
            x+=1
            y=0
        else:y+=1
fig,ax = plt.subplots(3,1,figsize = (25,25))
#plt.figure(figsize = (25,10))
sns.barplot(x = data['Chance of Admit '],y = data['CGPA'],ax = ax[0] )
sns.barplot(x = data['Chance of Admit '],y = data['GRE Score'],ax = ax[1] )
sns.barplot(x = data['Chance of Admit '],y = data['TOEFL Score'],ax = ax[2] )
sns.set_style('white')
data['Admission Percentage'].value_counts().plot(kind = 'bar',alpha = 0.7)
plt.title("Number of people admitted and not admitted")
f,ax = plt.subplots(1,3,figsize = (20,4))
sns.scatterplot(y = data['Chance of Admit '],x = data['CGPA'],hue = data['Admission Percentage'],ax = ax[0])
ax[0].axvline(data['CGPA'].mean())
sns.scatterplot(y = data['Chance of Admit '],x = data['GRE Score'],hue = data['Admission Percentage'],ax = ax[1])
ax[1].axvline(data['GRE Score'].mean())
sns.scatterplot(y = data['Chance of Admit '],x = data['TOEFL Score'],hue = data['Admission Percentage'],ax = ax[2])
ax[2].axvline(data['TOEFL Score'].mean())
plt.figure(figsize = (10,10))
sns.heatmap(data.corr(),annot = True)
plt.title('Correlation')
correlations = data.corr().sort_values(by = 'Chance of Admit ')
correlations['Chance of Admit '].plot(kind = 'bar')
from sklearn.model_selection import train_test_split
data.drop('Admission Percentage',axis = 1,inplace = True)
X = data.drop('Chance of Admit ',1)
Y = data['Chance of Admit ']
X, xtest, Y, ytest = train_test_split(X,Y,test_size = 0.1)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X,Y)
linear_model.score(xtest,ytest)
best_features = ['CGPA','TOEFL Score','GRE Score','University Rating']
X = data[best_features]
Y = data['Chance of Admit ']
X, xtest, Y, ytest = train_test_split(X,Y,test_size = 0.1)
best_model = ''
best_score = 0
for i in range(5):
    linear_model.fit(X,Y)
    score = linear_model.score(xtest,ytest)
    if score > best_score:best_score,best_model = score,linear_model
print(best_score)
ypreds = linear_model.predict(xtest)
plt.scatter(ytest,ypreds)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.plot([0,1],[0,1])

best_features = ['CGPA','TOEFL Score','GRE Score','University Rating']
X = data[best_features]
Y = data['Chance of Admit ']
X, xtest, Y, ytest = train_test_split(X,Y,test_size = 0.2)
import tensorflow as tf
from tensorflow import keras
neural_model = keras.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(12,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1,activation = 'sigmoid')
])
neural_model.compile(loss='mse',optimizer = 'sgd')
history = neural_model.fit(X,Y,epochs = 15,batch_size = 10)
loss = history.history['loss']
plt.plot(loss,label = 'Loss Function')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()