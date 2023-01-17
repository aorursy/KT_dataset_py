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
dataset = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

init_notebook_mode()
dataset.head()
dataset.describe()
#to visually check for missing data in the dataset, we will make use of the missigno module

import missingno as msn



msn.matrix(dataset)
from sklearn.impute import SimpleImputer

impute_salary = SimpleImputer(missing_values=np.nan, strategy='mean')

impute_salary.fit(dataset[['salary']])

dataset[['salary']] = impute_salary.transform(dataset[['salary']])

#before we continue, lets drop the sl_no column as it is redundant and contains no information



dataset = dataset.drop(columns=['sl_no'])
msn.matrix(dataset)
#we then proceed to check for outliers using the boxplot





plt.figure(figsize = (15,15))





plt.subplot(3, 2, 1)

sns.boxplot(x=dataset.ssc_p)



plt.subplot(3, 2, 2)

sns.boxplot(x=dataset.hsc_p)



plt.subplot(3, 2, 3)

sns.boxplot(x=dataset.degree_p)



plt.subplot(3, 2, 4)

sns.boxplot(x=dataset.etest_p)



plt.subplot(3, 2, 5)

sns.boxplot(x=dataset.mba_p)



plt.subplot(3, 2, 6)

sns.boxplot(x=dataset.salary)



plt.show()
#drop gender, ssc_b and hsc_b column as they intuitively have no effect in making inferences on if the data point will get placed 

#or the salary of the each data point



columns_drp = ['gender', 'ssc_b', 'hsc_b']



dataset = dataset.drop(columns=columns_drp)
dataset.head()
#checking distribution of datapoints on status feature



dataset.groupby('status').count()
#label encode the status column



from sklearn.preprocessing import LabelEncoder

encode_status = LabelEncoder()

dataset['status'] = encode_status.fit_transform(dataset['status'])
#encoding categorical features 

# encode using target encoder and status as the target



from category_encoders import TargetEncoder



target_encoder_hsc_s = TargetEncoder()

dataset[['hsc_s']] = target_encoder_hsc_s.fit_transform(dataset[['hsc_s']], dataset[['status']])
# encode using degree_t target encoder and status as the target



target_encoder_degree = TargetEncoder()

dataset[['degree_t']] = target_encoder_degree.fit_transform(dataset[['degree_t']], dataset[['status']] )
# encoding using workex target encoder and status as the target



target_encoder_workex = TargetEncoder()

dataset[['workex']] = target_encoder_workex.fit_transform(dataset[['workex']], dataset[['status']] )
# encoding using specialisation target encoder and status as the target



target_encoder_spec = TargetEncoder()

dataset[['specialisation']] = target_encoder_spec.fit_transform(dataset[['specialisation']], dataset[['status']] )
dataset
#divide into features of train and target



X = dataset[['ssc_p','hsc_p','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','mba_p','salary']]

y = dataset[["status"]]
X.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)
#create an xgboostclassifier model



from xgboost import XGBClassifier



clf = XGBClassifier()



#train the model on train set

clf.fit(X_train, y_train)
#test the trained model on the test set

y_pred = clf.predict(X_test)
#check performance metrics

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
#check accuracy score

from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(y_test, y_pred)

Accuracy
#divide into features of train and target



X_reg = dataset[['ssc_p','hsc_p','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','mba_p','status']]

y_reg = dataset[["salary"]]
# scale the matrix of features X



scaler_reg = StandardScaler()

X_reg = scaler_reg.fit_transform(X_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size = 0.3, random_state = 42)
#create an Xgboostregressor



from xgboost import XGBRegressor



reg = XGBRegressor()



#train the regression model

reg.fit(X_train_reg, y_train_reg)
#test the regression model

y_pred_reg = reg.predict(X_test_reg)
#checking for performance of regression model

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt



met1 = r2_score(y_test_reg, y_pred_reg)

met1