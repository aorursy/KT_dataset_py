import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import csv as csv
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import catboost as cb
#Read CSV file
df_train=pd.read_csv('../input/zs-ml-challenge/train.csv')
data_test=pd.read_csv('../input/zs-ml-challenge/test.csv')

#for i in df_train['intake_datetime']:
    
df_train.isnull().sum()
sns.countplot(x="outcome_type",data=df_train)
plt.figure(figsize=(17,6))
sns.countplot(x="animal_type",hue="outcome_type",data=df_train)
sns.countplot(data = df_train, x='sex_upon_outcome', hue='outcome_type')
#impute the missing values & type conversion
df_train["sex_upon_intake"].fillna("Unknown", inplace = True)
df_train["sex_upon_outcome"].fillna("Unknown", inplace = True)
df_train["sex_upon_intake"].unique()
df_train.sex_upon_intake = df_train.sex_upon_intake.astype(object)
df_train.sex_upon_outcome = df_train.sex_upon_outcome.astype(object)
df_train["sex_upon_outcome"].unique()

data_test["sex_upon_intake"].fillna("Unknown", inplace = True)
data_test["sex_upon_outcome"].fillna("Unknown", inplace = True)
data_test["sex_upon_intake"].unique()
data_test.sex_upon_intake = data_test.sex_upon_intake.astype(object)
data_test.sex_upon_outcome = data_test.sex_upon_outcome.astype(object)
data_test["sex_upon_outcome"].unique()
#removing some redundant features & Target value
X_train1 = df_train.drop(["outcome_type",'count','age_upon_intake','age_upon_intake_(years)','intake_monthyear','animal_id_outcome','age_upon_intake_age_group','intake_monthyear','time_in_shelter','age_upon_outcome','age_upon_outcome_(years)','age_upon_outcome_age_group','outcome_datetime','outcome_monthyear'],axis=1)
ytrain1 = df_train["outcome_type"]

X_test = data_test.drop(['count','age_upon_intake','age_upon_intake_(years)','intake_monthyear','animal_id_outcome','age_upon_intake_age_group','intake_monthyear','time_in_shelter','age_upon_outcome','age_upon_outcome_(years)','age_upon_outcome_age_group','outcome_datetime','outcome_monthyear'],axis=1)
#Label encoding features having large varity of attributes
X_train1.head()
#le_breed = preprocessing.LabelEncoder()
#X_train1.breed = le_breed.fit_transform(X_train1.breed)
le_intake_condition = preprocessing.LabelEncoder()
X_train1.intake_condition = le_intake_condition.fit_transform(X_train1.intake_condition)
le_color = preprocessing.LabelEncoder()
X_train1.color = le_color.fit_transform(X_train1.color)

le_out = preprocessing.LabelEncoder()
y_train = le_out.fit_transform(ytrain1)

X_train1.head()
le_breed = preprocessing.LabelEncoder()
X_test.breed = le_breed.fit_transform(X_test.breed)
le_intake_condition = preprocessing.LabelEncoder()
X_test.intake_condition = le_intake_condition.fit_transform(X_test.intake_condition)
le_color = preprocessing.LabelEncoder()
X_test.color = le_color.fit_transform(X_test.color)
X_train, X_test1, ytrain, y_test = train_test_split(X_train1, ytrain1, test_size=0.2)
#implementing Cataboost
cat_features_index=['intake_datetime','breed','animal_type','intake_condition','intake_datetime','intake_type','date_of_birth','sex_upon_intake','intake_weekday','sex_upon_outcome','intake_weekday','outcome_weekday']
clf2 = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=15,depth=6, iterations= 800, l2_leaf_reg= 5, learning_rate= 0.15,random_seed=0)
clf2.fit(X_train1,ytrain1, cat_features= cat_features_index)
ytest=clf2.predict(X_test1)
accuracy_score(ytest,  y_test)
#mapping
target_mapping = {index: label for index, label in 
                  enumerate(le_out.classes_)}
target_mapping
y_pred=clf2.predict(X_test1)
Y_pred=clf2.predict(X_test)
Y_pred_outcome=pd.DataFrame(Y_pred)
accuracy_score(y_pred,  y_test)
Y_pred.shape
ID=pd.DataFrame(data_test['animal_id_outcome'])
ID.size
predicted_outcome=pd.DataFrame(np.hstack([ID,Y_pred]),columns=['animal_id_outcome','outcome_type'])
predicted_outcome
features=X_train1.columns
importances = clf2.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show
predicted_outcome.to_csv('predicted_outcome.csv')