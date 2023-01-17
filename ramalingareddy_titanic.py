# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/gender_submission.csv")
test = pd.read_csv("../input/test.csv")
from sklearn.model_selection import train_test_split
data_dummies = pd.get_dummies(data.drop(['Ticket','Name','Cabin'],axis =1))
train,validate = train_test_split(data_dummies,test_size = 0.3,random_state = 100)
train.head()
train_y = train['Survived']
validate_y = validate['Survived']
train_x = train.drop('Survived',axis=1)
validate_x = validate.drop('Survived',axis =1)

##standardize data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train_x)

train_x_scaled = pd.DataFrame(scaler.transform(train_x),columns = train_x.columns)
validate_x_scaled = scaler.transform(validate_x)


##imputing the missing values


#method1:impute with column mean
data_dummies['Age_m1']=data_dummies['Age'].fillna(data_dummies['Age'].mean())
#data_dummies['Age'].mean()
#method2:impute with column median(if u have outliers in age)
data_dummies['Age_m1']=data_dummies['Age'].fillna(data_dummies['Age'].median())
pd.isnull(data_dummies['Age_m1'].sum())

avg_age_female = round(data[data['Sex']=='female']['Age'].mean())
avg_age_male = round(data[data['Sex']=='male']['Age'].mean())

print(avg_age_female,avg_age_male)

#method 3 :impute using other columns
#data_dummies[pd.isnull(data_dummies['Age'])]
def method3_impute(row):
    if row.Sex_female == 1:
        row.Age = avg_age_female
    else:
        row.Age = avg_age_male
    return row
data_dummies.apply(method3_impute,axis =1)

###method4:
def method3_impute(row):
    if row.Sex_female == 1 and pd.isnull(row.Age):
        row.Age = avg_age_female
    elif row.Sex_female == 0 and pd.isnull(row.Age):
        row.Age = avg_age_male
    else:
        pass
    return row
data_dummies_m3=data_dummies.apply(method3_impute,axis =1).head()
##data_dummies_m3['Age'].plot.density()

###
        
    





####method 5 using machine learning algorithmn knn

cols_drop = ['Age','Age_m1','Survived']
train_x_impute=data_dummies[~pd.isnull(data_dummies['Age'])]
train_x_impute = train_x_impute.drop(cols_drop , axis =1)


test_x_impute=data_dummies[pd.isnull(data_dummies['Age'])]
test_x_impute = test_x_impute.drop(cols_drop , axis =1)

train_y_impute = data_dummies.loc[train_x_impute.index,'Age']


##standardization
from sklearn.preprocessing import StandardScaler
scaler_impute = StandardScaler()
scaler_impute.fit(train_x_impute)
train_x_impute_scaled = scaler_impute.transform(train_x_impute)
test_x_impute_scaled = scaler_impute.transform(test_x_impute)

###Fit knn model
from sklearn.neighbors import KNeighborsRegressor
model_impute = KNeighborsRegressor(n_neighbors =5)
model_impute.fit(train_x_impute_scaled,train_y_impute)
test_y_impute_pred = model_impute.predict((test_x_impute_scaled))





##impute predictions in original data
data_dummies.loc[test_x_impute.index,'Age']=test_y_impute_pred

#data_dummies = pd.get_dummies(data.drop(['Age_m1'],axis =1))
train,validate = train_test_split(data_dummies,test_size = 0.3,random_state = 100)
train.head()
train_y = train['Survived']
validate_y = validate['Survived']
train_x = train.drop('Survived',axis=1)
validate_x = validate.drop('Survived',axis =1)


pd.isnull(data_dummies).sum()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

data_dummies= data_dummies.drop('Age_m1',axis =1)
train,validate = train_test_split(data_dummies,test_size = 0.3,random_state = 100)
train_y = train['Survived']
validate_y = validate['Survived']
train_x = train.drop('Survived',axis=1)
validate_x = validate.drop('Survived',axis =1)


m1= DecisionTreeClassifier(max_depth=4,random_state = 100)
m1.fit(train_x,train_y)
validate_pred = pd.DataFrame(m1.predict_proba(validate_x),columns = ['Neg_0','Pos_1'])
fpr_m1 ,tpr_m1,cutoffs_1 = roc_curve(validate_y,validate_pred['Pos_1'])
auc_m1 = auc(fpr_m1,tpr_m1)


m2=RandomForestClassifier(max_depth=1,n_estimators = 300)
m2.fit(train_x,train_y)
validate_pred2 = pd.DataFrame(m2.predict_proba(validate_x),columns = ['Neg_0','Pos_1'])
fpr_m2,tpr_m2,cutoffs_2 = roc_curve(validate_y,validate_pred2['Pos_1'])
auc_m2 = auc(fpr_m2,tpr_m2)


m3=AdaBoostClassifier(random_state=100,n_estimators = 300)
m3.fit(train_x,train_y)
validate_pred3 = pd.DataFrame(m3.predict_proba(validate_x),columns = ['Neg_0','Pos_1'])
fpr_m3,tpr_m3,cutoffs_3 = roc_curve(validate_y,validate_pred3['Pos_1'])
auc_m3 = auc(fpr_m3,tpr_m3)


m4= DecisionTreeClassifier(max_depth=10,random_state = 100)
m4.fit(train_x,train_y)
validate_pred4 = pd.DataFrame(m4.predict_proba(validate_x),columns = ['Neg_0','Pos_1'])
fpr_m4 ,tpr_m4,cutoffs_4 = roc_curve(validate_y,validate_pred4['Pos_1'])
auc_m4 = auc(fpr_m4,tpr_m4)
import matplotlib.pyplot as plt

p1 = plt.plot(fpr_m1,tpr_m1,color = 'red')
plt.xlabel('False positive rate(FPR)')
plt.ylabel('True positive rate(Tpr)')
p2 = plt.plot(fpr_m2,tpr_m2,color = 'green')
p3 = plt.plot(fpr_m3,tpr_m3,color = 'steelblue')
p4 = plt.plot(fpr_m4,tpr_m4,color = 'yellow')
plt.legend(['Decision Tree max depth(4);Auc=%.2f'% auc_m1,
            'Random Forest ;Auc=%.2f'% auc_m2,
            'Ada Boost;Auc=%.2f'% auc_m3,
            'Decision Tree max depth(10);Auc=%.2f'% auc_m4])







cols_drop = ['Cabin','Name','Ticket']
test_dummies=pd.get_dummies(test.drop(cols_drop,axis = 1))

test_x_impute = scaler_impute.transform(test_dummies.drop('Age',axis = 1 ))
test_x_impute = pd.DataFrame(test_x_impute,columns = test_dummies.columns.drop('Age'))
test_x = test.drop(cols_drop,axis = 1)

test_age_miss_rows = test[pd.isnull(test['Age'])].index
fare_avg = test_x_impute['Fare'].mean()
test_age_miss = test_x_impute.loc[test_age_miss_rows,:]
test_age_miss_pred = model_impute.predict(test_age_miss)

test_x_impute['Age']=test['Age']
test_x_impute.loc[test_age_miss_rows,'Age']=test_age_miss_pred

test_x_impute['Fare']= test_x_impute['Fare'].fillna(fare_avg)
test_pred = m2.predict(test_x_impute)


submission = pd.DataFrame({'PassengerId': test['PassengerId'].values,'Survived':test_pred})
submission.to_csv('Submission1.csv',index = False)
submission['Survived'].value_counts()
