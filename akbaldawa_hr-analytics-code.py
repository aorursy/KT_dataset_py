# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/Train.csv')
train.head()
test = pd.read_csv('/kaggle/input/Test.csv')
train['type'] = 'Train'
test['type']  = 'Test'
print(train.shape,test.shape)
train.info()
train.describe()
import seaborn as sns

sns.countplot(x = 'is_promoted',data=train)
chart = sns.countplot(x = 'age',data=train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
import matplotlib.pyplot as plt

#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);
train.select_dtypes(include=['object'])

##Categorical columns are ['department 	region 	education 	gender 	recruitment_channel']
chart = sns.countplot('department',data = train[train['education'].isna()])
chart.set_xticklabels(chart.get_xticklabels(), rotation=40)
train.groupby("is_promoted")["education"].count()
train.groupby("department")["education"].count()
##unique values of the categorical values

train['department'].unique()

##array(['Sales & Marketing', 'Operations', 'Technology', 'Analytics',
##       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], dtype=object)

train['region'].unique()

# array(['region_7', 'region_22', 'region_19', 'region_23', 'region_26',
#        'region_2', 'region_20', 'region_34', 'region_1', 'region_4',
#        'region_29', 'region_31', 'region_15', 'region_14', 'region_11',
#        'region_5', 'region_28', 'region_17', 'region_13', 'region_16',
#        'region_25', 'region_10', 'region_27', 'region_30', 'region_12',
#        'region_21', 'region_8', 'region_32', 'region_6', 'region_33',
#        'region_24', 'region_3', 'region_9', 'region_18'], dtype=object)

train['education'].unique()

# array(["Master's & above", "Bachelor's", nan, 'Below Secondary'],
#       dtype=object)

train['recruitment_channel'].unique()

#array(['sourcing', 'other', 'referred']
train['education'].where(train['department'].isna())
Train_Target = train.is_promoted
data = pd.concat([train,test],axis=0)
data.head()
data.shape
data.isnull().sum()
Target = data['is_promoted']
data = data.drop(['is_promoted'],axis = 1)
Missing_prev_year = data[data.previous_year_rating.isna()]
Missing_prev_year.head()
Missing_prev_year['length_of_service'].value_counts()
Missing_prev_year['KPIs_met >80%'].value_counts()
Missing_prev_year['no_of_trainings'].value_counts()
Missing_prev_year['awards_won?'].value_counts()
Missing_prev_year['awards_won?'].value_counts()
Missing_prev_year['avg_training_score'].plot(kind='hist')
data['education'].replace("Master's & above",3,inplace=True)
data['education'].replace("Bachelor's",2,inplace=True)
data['education'].replace("Below Secondary",1,inplace=True)
data['education'] = data['education'].fillna(2)
data['education'] = data['education'].astype(int) 
data.isna().sum()
data['previous_year_rating'] = data['previous_year_rating'].fillna(3)
data.head()
data['previous_year_rating'] = data['previous_year_rating'].astype(int) 
data = data.drop(['employee_id','recruitment_channel'],axis = 1)
data.columns
data['tot_score'] = data['no_of_trainings'] * data['avg_training_score']
data['tot_score'].describe()
tot_val = data['tot_score'].value_counts()
#data.drop(['tot_score'],axis = 1)
plt.figure(figsize=(16, 10))
chart= sns.countplot("tot_score",data = data)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.show()
#data = data.drop(['region'],axis = 1)
plt.figure(figsize=(16, 6))
chart= sns.countplot("avg_training_score",data = data)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.show()
data.groupby('department')['avg_training_score'].mean()
#data.groupby('department')['tot_score'].mean()
data['tot_score'].mean()
data['tot_score'] = np.where((data['department'] == 'Analytics'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Finance'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'HR'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Legal'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Operations'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Procurement'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'R&D'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Sales & Marketing'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'] = np.where((data['department'] == 'Technology'),
                                       data['tot_score']/83,
                                       data['tot_score'])
data['tot_score'].describe()
data['avg_training_score'] = round(data['avg_training_score'],2) 
data['tot_score'] = round(data['tot_score'],2) 
data['avg_training_score'].describe()
# #for rows,columns in data:
    
# for col in data.department.unique():
#     data[data['department'] == col] = data[data['department'] == col].groupby('department')['avg_training_score'].mean()
#     #data[data['department'] == 'Technology'].groupby('department')['avg_training_score'].mean()

#     #data[data['department']== 'Technology']['avg_training_score'] = data.groupby('department')['avg_training_score'].mean()
bins = [0.78,0.90,0.96,1.01,1.09,1.97]
labels = ["0.78-0.89","0.90-0.95","0.96-1.0","1.01-1.08","1.09-1.97"]
data["TrainGroup"] = pd.cut(data["avg_training_score"],bins, labels = labels, include_lowest = True)
Train_score_map = {"0.78-0.89":0,"0.90-0.95":1,"0.96-1.0":2,"1.01-1.08":3,"1.09-1.97":4}
data["TrainGroup"] = data["TrainGroup"].map(Train_score_map)
data.drop("avg_training_score", axis=1, inplace=True)
bins = [20,30,40,50,60]
labels = ["20-29","30-39","40-49","50+"]
data["AgeGroup"] = pd.cut(data["age"],bins, labels = labels, include_lowest = True)
age_mapping = {"20-29":0,"30-39":1,"40-49":2,"50+":3}
data["AgeGroup"] = data["AgeGroup"].map(age_mapping)
data.drop("age", axis=1, inplace=True)
# bins = [0.01,0.79,0.90,1.11,1.97]
# labels = ["0-0.01","0.02-0.78","0.79-0.89","0.90-0.10","0.11-1.97"]
# data["TotScoreGp"] = pd.cut(data["tot_score"],bins, labels = labels, include_lowest = True)
# score_mapping = {"1-50":0,"51-100":1,"101-199":2,"200-499":3,"500-999":4,"1000-3198":5}
# data["TotScoreGp"] = data["TotScoreGp"].map(score_mapping)
# data.drop("tot_score", axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

data_final = pd.DataFrame()
label = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data_final[col] = label.fit_transform(data[col])
    else:
        data_final[col] = data[col]
# scaler = StandardScaler()
# scaler.fit(data_final)
# data_final = scaler.transform(data_final)
train_final = data_final[data_final['type'] == 1]
test_df     = data_final[data_final['type'] == 0]

train_final = train_final.drop(['type'],axis = 1)
test_df     = test_df.drop(['type'],axis = 1)
## Machine learning tools.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
##Splitting the dataset

independent_var = train_final
dependent_var = Train_Target
x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, 
                                                  test_size = 20, random_state = 0)
models = []
models.append(SVC())
models.append(LinearSVC())
models.append(Perceptron())
models.append(GaussianNB())
models.append(SGDClassifier())
models.append(LogisticRegression())
models.append(KNeighborsClassifier())
models.append(RandomForestClassifier())
models.append(DecisionTreeClassifier())
models.append(GradientBoostingClassifier())

i = 0
accuracy_list = []
model_name_list = ["SVM","Linear SVC","Perceptron","Gaussian NB","SGD Classifier","Logistic Regression",
                   "K-Neighbors Classifier","Random Forest Classifier","Decision Tree","Gradient Boosting"]
employee_id = test["employee_id"]

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = (f1_score(y_pred, y_test,average="macro"))
    accuracy_list.append(accuracy)
    pred = model.predict(test_df)
    predictions = pd.DataFrame({ "employee_id" : employee_id, "is_promoted": pred })
    predictions.to_csv('HR_Submission_%s.csv'%(model_name_list[i]),index=False)
    i += 1

best_model = pd.DataFrame({"Model": model_name_list, "Score": accuracy_list})
best_model.sort_values(by="Score", ascending=False)
predictions['is_promoted'].value_counts()
