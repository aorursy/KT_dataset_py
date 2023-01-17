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
import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.utils import shuffle
dataset=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

dataset.head()
dataset.columns
# now check the shape of data set

dataset.info()
# lets describe the dataset



print('Unique value available in columns:')

print('SEX has',dataset['sex'].unique())

print('CP has',dataset['cp'].unique())

print('Fbs has',dataset['fbs'].unique())

print('EXANG has',dataset['exang'].unique())

print('Slope has',dataset['slope'].unique())

print('ca has',dataset['ca'].unique())

print('thal has',dataset['thal'].unique())

print('target has',dataset['target'].unique())

print('\t')

print(dataset.describe().T)
dataset.isnull().any()

plt.style.use('ggplot')

fig,ax=plt.subplots(1,2,figsize=(20,5))

ax[0].set_title('Age Distribution')

ax[1].set_title('Resting Blood Pressure ')

sns.distplot(dataset['age'],ax=ax[0])

sns.distplot(dataset['trestbps'],ax=ax[1])
plt.style.use('ggplot')

plt.figure(figsize=(6,4))

sns.countplot(dataset['target'])
corr_matrix=dataset.corr()

plt.figure(figsize=(15,6))

sns.heatmap(corr_matrix)
print(dataset.head(2))

print(dataset.tail(2))



dataset=shuffle(dataset)
dataset.head()
x_dataset=dataset.drop('target',axis=1)

y_dataset=dataset['target']

print(x_dataset.shape,y_dataset.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# transform data

x_dataset = scaler.fit_transform(x_dataset)

print(x_dataset)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.model_selection import RandomizedSearchCV

#splitting trainx,test_x,train_y,tets_y of size 20 percent

X_train,X_test,y_train,y_test=train_test_split(x_dataset,y_dataset,test_size=0.2,random_state=0)

print(X_train.shape,y_train.shape)# validating shape
model1=KNeighborsClassifier()

model2= SVC()

model3= XGBClassifier()

model4= RandomForestClassifier()

model5= LogisticRegression()

models=[model1,model2,model3,model4,model5]
for model in models:

    model.fit(X_train,y_train)

    print(model)

    print('score',model.score(X_test,y_test))

    y_pred=model.predict(X_test)

    print('F1-score=',f1_score(y_test,y_pred))
# creating model as RandomizedSearchCV

model=RandomForestClassifier()
# defining param grid

n_estimators = [100, 200, 300, 400, 500]

max_features = ['auto', 'sqrt']

max_depth = [5, 10, 20, 30, 40, 50]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

        

random_grid = {'n_estimators': n_estimators,

               'criterion':['gini','entropy'],

                       'max_features': max_features,

                       'max_depth': max_depth,

                       'min_samples_split': min_samples_split,

                       'min_samples_leaf': min_samples_leaf,

                       'bootstrap': bootstrap}
#randomSearch=RandomizedSearchCV(estimator =model, param_distributions = random_grid, cv = 10)

#randomSearch.fit(X_train,y_train)
# lets print the best param and best score of after tuning

#print('best_params',randomSearch.best_params_)

#print('best-score',randomSearch.best_score_)
model=RandomForestClassifier(n_estimators=500,min_samples_split=2,min_samples_leaf=4,max_features='sqrt',max_depth=10,bootstrap=False,criterion='gini')

model.fit(X_train,y_train)

print(model)

print('score',model.score(X_test,y_test))

y_pred=model.predict(X_test)

print('F1-score=',f1_score(y_test,y_pred))
y_pred=model.predict(X_test)

y_pred
import pickle

# save the model to disk

filename = 'finalized_model.pkl'

pickle.dump(model, open(filename, 'wb'))