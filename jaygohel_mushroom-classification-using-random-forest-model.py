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

3 # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# Importing data

data = pd.read_csv(r'/kaggle/input/mushroom-classification/mushrooms.csv')

data.head()
# checking number of rows and columns in data

print('Number of rows in dataset is {} and number of columns is {}'.format(data.shape[0],data.shape[1]))
data.info()
# Checking the balance of target variable

data['class'].value_counts()
# checking distinct values in the columns

total_col_list=list(data.columns)

for col_name in total_col_list:

#     print((data[col_name].value_counts()/data.shape[0])*100)

    data[col_name].value_counts().plot.pie(autopct='%.2f')

    plt.title('Pie chart for {}'.format(col_name))

    plt.show()
# column gill-attachment,viel_type,veil_color and ring_number do not convey any significant information. 

# Hence dropping those columns where percentage of a particular value is greater than or equal to 90

drop_list=[]

for col_name in total_col_list:

    if ((data[col_name].value_counts()/data.shape[0])*100).sort_values(ascending=False)[0]>=90:

        drop_list.append(col_name)

drop_list
# dropping columns present in drop_list

print('Number of columns before dropping is {}'.format(data.shape[1]))

data.drop(columns=drop_list,inplace=True)

print('Number of columns after dropping is {}'.format(data.shape[1]))
# since the data is categorical, we will use one-hot enconding and then split the data and build the model

total_col_list=list(data.columns)

data_num=pd.get_dummies(data[total_col_list],drop_first=True,prefix=total_col_list,prefix_sep='_')

data_num.head()
# let us check if there is any multicollinearity present

Y=pd.DataFrame(data_num.pop('class_p'))

X=pd.DataFrame(data_num)

# Splitting the data into train and test data set



X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,stratify=Y)
# Building Random Forest Model

param_grid={'max_depth': [4,8,12,16,20],

           'n_estimators':[50,100,150,200,250,300],

           'min_samples_leaf':range(50,200,50),

           'min_samples_split':range(50,200,50),

           'max_features':range(1,19,3)}



RF=RandomForestClassifier(random_state=42)



grid_search=GridSearchCV(estimator=RF,

                         param_grid=param_grid,

                         n_jobs=-1,

                         cv=4,

                         verbose=2,

                         scoring='accuracy'

                        )



grid_search.fit(X_train,y_train)
best_params=grid_search.best_estimator_.get_params()

print('We get accuracy of {} using parameters {}'.format(round(grid_search.best_score_,4)*100,grid_search.best_estimator_.get_params()))
RF_FINAL=RandomForestClassifier(bootstrap=best_params['bootstrap'],

                               criterion=best_params['criterion'],

                               max_depth=best_params['max_depth'],

                               max_features=best_params['max_features'],

                               min_samples_leaf=best_params['min_samples_leaf'],

                               min_samples_split=best_params['min_samples_split'],

                               n_estimators=best_params['n_estimators'],

                               n_jobs=-1)



RF_FINAL.fit(X_train,y_train)
y_train_pred=RF_FINAL.predict(X_train)



print('Accuracy on training dataset is {}'.format(round(accuracy_score(y_train,y_train_pred),4)*100))
y_test_pred=RF_FINAL.predict(X_test)



print('Accuracy on test dataset is {}'.format(round(accuracy_score(y_test,y_test_pred),4)*100))
# Let us make prediction on complete dataset

y_pred=RF_FINAL.predict(X)



print('Accuracy on complete dataset is {}'.format(round(accuracy_score(Y,y_pred),4)*100))