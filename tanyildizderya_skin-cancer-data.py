# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HAM10000_metadata.csv')
data.head()
data.tail()
data.describe()
data.sex = [1 if each == 'female'else 0 for each in data.sex]
data.tail()
image = pd.read_csv('../input/hmnist_28_28_L.csv')
image.head()
data.dx = [1 if each == 'bkl' or each == 'nv' or each == 'df' else 0 for each in data.dx]
data.dx
data.head()
#data['localization'] = pd.Categorical(data['localization'])
#new_categ_df = pd.get_dummies(data['localization'],prefix ='local')
#new_data_frame = pd.concat([data,new_categ_df],axis=1)
#new_data_frame.head()
#new_data_frame['dx_type'] = pd.Categorical(new_data_frame['dx_type'])
#new_categ_df = pd.get_dummies(new_data_frame['dx_type'],prefix ='dx_type')
#new_data =  pd.concat([new_data_frame,new_categ_df],axis=1)
#new_data.head()
#data = new_data.drop(columns=['image_id', 'lesion_id','dx_type','localization'],axis=1)
data.head()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['age'].dropna()
data = data.dropna()
sns.set_style('whitegrid')
sns.countplot(x='dx',data=data)
data.head()
sns.countplot(x='dx',hue='age',data=data)

plt.legend(loc='best')
sns.distplot(data['age'].dropna(),kde=False)
data['age'].plot.hist(bins=35)
import cufflinks as cf
cf.go_offline()
#data['age'].iplot(kind='hist',bins=10)
sns.boxplot(x='dx_type', y='age', data=data)
data['localization'] = pd.Categorical(data['localization'])
new_categ_df = pd.get_dummies(data['localization'],prefix ='local')
new_data_frame = pd.concat([data,new_categ_df],axis=1)
new_data_frame.head()
data = new_data_frame
data.head()
new_data_frame['dx_type'] = pd.Categorical(new_data_frame['dx_type'])
new_categ_df = pd.get_dummies(new_data_frame['dx_type'],prefix ='dx_type')
new_data =  pd.concat([new_data_frame,new_categ_df],axis=1)
new_data.head()
new_data_frame = pd.concat([data,new_categ_df],axis=1)
data = new_data_frame
data.head()
data = data.drop(['localization','lesion_id','image_id','dx_type'],axis=1)
data.head()
x = data.drop('dx',axis=1)

y = data['dx']
#Logistic Regression

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
sns.countplot(x='dx_type_histo',hue='age',data=data)

plt.legend(loc='best')
#KNN

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('dx',axis=1))
scaled_features = scaler.transform(data.drop('dx',axis=1))
scaled_features
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()
from sklearn.model_selection import train_test_split
x = df_feat

y = data['dx']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
pred
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)

plt.title('error rate vs K value')

plt.xlabel('k')

plt.ylabel('error rate')
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

pred = knn.predict(x_test)

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(x_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(x_test)
grid_predictions
print(confusion_matrix(y_test,grid_predictions))

print('\n')

print(classification_report(y_test,grid_predictions))
data.head()
sns.scatterplot(x='dx',y='age',data=data)

plt.plot()
sns.pairplot(data,hue='dx')