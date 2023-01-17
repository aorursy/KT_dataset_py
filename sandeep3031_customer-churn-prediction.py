#importing libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import date,datetime,timedelta

import time

import warnings

warnings.filterwarnings("ignore")
#reading the train data and test data

data=pd.read_csv('../input/Train.csv')

test=pd.read_csv('../input/Test.csv')
data.head()
#checking shape of the data

data.shape
#checking NA's  values 

data.isna().sum()
#checking statistics for every column

data.describe(include='all')

#checking data types 

data.dtypes
#droping the unwanted columns

data.drop('CustomerID',axis=1,inplace=True)

data.drop('CustomerName',axis=1,inplace=True)
#creating a new Dateof birth column from yearofbirth,monthofbirth,dayoofbirth

data['Dateofbirth'] = pd.to_datetime(

data[['yearofBirth', 'monthofBirth', 'dayofBirth']].astype(str).agg('-'.join, axis=1))



#creating a year column from dateofbirth

data['year']=data['Dateofbirth'].dt.year



#adding a new column AGE from present year and year of birth

data['age']=2019-data['year']

sns.countplot(data.Churn)

plt.title("distribution of Levels in churn ")
sns.countplot(data.Occupation)

plt.xticks(rotation=45)

plt.title('No of categories in occupation')
#histogram for account balance

plt.hist(data.AccountBalance,bins=10)

plt.xlabel("account balance")

plt.ylabel("frequency")

plt.title("distribution for account balance")

sns.boxplot(x='Churn',y='CreditScore',data=data)

plt.title("churn vs CreditScore")
sns.countplot(x='Churn',hue='Occupation',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
sns.boxplot(x='Churn',y='Salary',data=data)

plt.title("churn vs Salary")
#churn vs age,here we can observe that the churn rate is more between 40-50 age 

sns.boxplot(x=data.Churn,y=data.age)

plt.title("churn vs age")
sns.countplot(x='Churn',hue='Gender',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
#droping year column

data.drop('year',axis=1,inplace=True)
#creating lists of categorical and numerical columns

cat_cols=['Gender','Location','Education','MaritalStatus','Occupation','Ownhouse']

num_cols=['yearofBirth','monthofBirth','dayofBirth','yearofEntry','monthofEntry','dayofEntry','CreditScore','AccountBalance','NumberOfProducts','IsCreditCardCustomer','ActiveMember','Salary']

#separating the independent and dependent column

y=data.Churn

X=data

X.drop('Churn',axis=1,inplace=True)
#spliting the train data into train_X,train_y,valid_X,valid_y

train_X,valid_X,train_y,valid_y=train_test_split(X,y,train_size=0.7,random_state=1)



#printing the shape of train_X,train_y,validation_X,validation_y

print(train_X.shape)

print(valid_X.shape)

print(train_y.shape)

print(valid_y.shape)
#deleting the Dateofbirth column since we added new column called 'AGE'

train_X.drop('Dateofbirth',axis=1,inplace=True)



#CONVERTING THE DATATYPES TO FLOAT AND CATEGORIC

train_X[cat_cols]=train_X[cat_cols].apply(lambda x:x.astype("category"))

train_X[num_cols]=train_X[num_cols].apply(lambda x:x.astype("float"))
train_num_data=train_X.loc[:,num_cols]

train_cat_data=train_X.loc[:,cat_cols]
from sklearn.preprocessing import StandardScaler

stand=StandardScaler()

stand.fit(train_num_data[train_num_data.columns])

train_num_data[train_num_data.columns]=stand.transform(train_num_data[train_num_data.columns])
train_X=pd.concat([train_num_data,train_cat_data],axis=1)



#creating dummies 

train_X=pd.get_dummies(train_X,columns=cat_cols)
#PREPROCESSING ON VALIDATION DATA

#droping dateofbirth column since we calculated age column

valid_X.drop('Dateofbirth',axis=1,inplace=True)



#CONVERTING THE DATATYPES TO FLOAT AND CATEGORIC

valid_X[cat_cols]=valid_X[cat_cols].apply(lambda x:x.astype("category"))

valid_X[num_cols]=valid_X[num_cols].apply(lambda x:x.astype("float"))



valid_num_data=valid_X.loc[:,num_cols]

valid_cat_data=valid_X.loc[:,cat_cols]



valid_num_data[valid_num_data.columns]=stand.transform(valid_num_data[valid_num_data.columns])



valid_X=pd.concat([valid_num_data,valid_cat_data],axis=1)



#creating dummies 

valid_X=pd.get_dummies(valid_X,columns=cat_cols)
#checking the shape of train_X,validation_X

print(valid_X.shape)

print(train_X.shape)
test.head()
#keeping the CustomerID in custid variable for submission along with test predictions

custid=test.CustomerID

#preprocessing on test droping unwanted columns

test.drop('CustomerID',axis=1,inplace=True)

test.drop('CustomerName',axis=1,inplace=True)



#creating a new dateofbirth column from year of birth,month of birth,day of birth

test['Dateofbirth'] = pd.to_datetime(

test[['yearofBirth', 'monthofBirth', 'dayofBirth']].astype(str).agg('-'.join, axis=1))



#creating year column from dateofbirth for calculating age 

test['year']=test['Dateofbirth'].dt.year



#creating new "age" column subrating from present year and yearofbirth

test['age']=2019-test['year']



#and droping year column

test.drop('year',axis=1,inplace=True)



#droping dateofbirth column from test

test.drop('Dateofbirth',axis=1,inplace=True)
#CONVERTING THE DATATYPES TO FLOAT AND CATEGORIC

test[cat_cols]=test[cat_cols].apply(lambda x:x.astype("category"))

test[num_cols]=test[num_cols].apply(lambda x:x.astype("float"))



test_num_data=test.loc[:,num_cols]

test_cat_data=test.loc[:,cat_cols]



test_num_data[test_num_data.columns]=stand.transform(test_num_data[test_num_data.columns])



test=pd.concat([test_num_data,test_cat_data],axis=1)



#creating dummies 

test=pd.get_dummies(test,columns=cat_cols)
#checking shape for train_X,valdiation_X,test data

print(train_X.shape)

print(valid_X.shape)

print(test.shape)
#MODEL1 LOGISTIC MODEL

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,accuracy_score



#fitting the model on train_X,TRAIN_Y

log=LogisticRegression()

log.fit(train_X,train_y)

#prediction on train_x,validation_x using logistic model and storing it in variables 

train_preds1=log.predict(train_X)

valid_preds1=log.predict(valid_X)
#checking model performance using accuaracy and recall

print("accuracy on train data:",classification_report(train_y,train_preds1))

print("accuracy on validation data:",classification_report(valid_y,valid_preds1))

train_score1=accuracy_score(train_y,train_preds1)

valid_score1=accuracy_score(valid_y,valid_preds1)
#creating a function which will plot learning curves for our model

#which will help us get bias and variance

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)#if y limits are given consider the specified limits

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    #getting means and std for train and test for particular train_sizes

    #with specified cv.

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    #creating connections between the training score points since we only get points.

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    #similarly for cross validation

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
plot_learning_curve(log,'logistic regression learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
#MODEL2 DECISION TREE CLASSIFIER AND FITTING ON TRAIN AND VALIDATION

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(train_X,train_y)



#predicting on train_X,valdiation_x and storing in variables

train_preds2=dtc.predict(train_X)

valid_preds2=dtc.predict(valid_X)
#checking accuracy and recall on train and validation

print("accuracy on train data:",classification_report(train_y,train_preds2))

print("accuracy on validation data:",classification_report(valid_y,valid_preds2))

train_score2=accuracy_score(train_y,train_preds2)

valid_score2=accuracy_score(valid_y,valid_preds2)
plot_learning_curve(dtc,'Decision tree learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
#MODEL3 KNN CLASIFIER

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(train_X,train_y)



#predicting on train and validation

train_preds3=knn.predict(train_X)

valid_preds3=knn.predict(valid_X)



#checking accuarcy score on train and validation

print("accuracy_score on train data:",accuracy_score(train_y,train_preds3))

print("accuracy_score on validaion data:",accuracy_score(valid_y,valid_preds3))



train_score3=accuracy_score(train_y,train_preds3)

valid_score3=accuracy_score(valid_y,valid_preds3)
plot_learning_curve(knn,'KNN learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
#MODEL 4

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

svc=SVC()



 

param_grid = {



'C': [0.001, 0.01, 0.1, 1, 10],

'gamma': [0.001, 0.01, 0.1, 1], 

'kernel':['linear','rbf']}



 

svc_cv = GridSearchCV(estimator = svc, param_grid = param_grid, cv = 10,n_jobs=-1)
svc_cv.fit(train_X,train_y)
svc_cv.best_estimator_.fit(train_X,train_y)
#predicting on train and validation

train_preds4=svc_cv.best_estimator_.predict(train_X)

valid_preds4=svc_cv.best_estimator_.predict(valid_X)
#checking accuarcy score on train and validation

print("accuracy_score on train data:",accuracy_score(train_y,train_preds4))

print("accuracy_score on validaion data:",accuracy_score(valid_y,valid_preds4))



train_score4=accuracy_score(train_y,train_preds4)

valid_score4=accuracy_score(valid_y,valid_preds4)
plot_learning_curve(svc_cv.best_estimator_,'SVC learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
#MODEL 5

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()



max_depth=[4,6,8,10]

min_samples_leaf=[0.06,0.08,0.10]

max_features=[0.02,0.04,0.06,0.08]



params={

    "max_depth":max_depth,

    "min_samples_leaf": min_samples_leaf,

    "max_features": max_features,

}



grid=GridSearchCV(estimator=rfc,param_grid=params,cv=5,n_jobs=-1)

grid.fit(train_X,train_y)
grid.best_estimator_
#getting best estimator from grid search and fitting on train_x,train_y

grid.best_estimator_.fit(train_X,train_y)



#prediction on train and validation 

train_preds5=grid.predict(train_X)

valid_preds5=grid.predict(valid_X)

#checking the model performance using accuracy score on train and validation

print("accuracy_score on train data:",accuracy_score(train_y,train_preds5))

print("accuracy_score on validaion data:",accuracy_score(valid_y,valid_preds5))



train_score5=accuracy_score(train_y,train_preds5)

valid_score5=accuracy_score(valid_y,valid_preds5)
plot_learning_curve(grid.best_estimator_,'RandomForest learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
#MODEL 6

#importing Randomsearchcv and creating hyperparameters for XGboost 

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

xgb2=XGBClassifier()

n_estimaters=[50,100,150,200]

max_depth=[2,3,5,7]

learnin_rate=[0.05,0.1,0.15,0.20]

min_child_wgt=[1,2,3,4]







hyperparameter={

    "n_estimaters":n_estimaters,

    "max_depth":max_depth,

    "learnin_rate":learnin_rate,

    "min_child_wgt":min_child_wgt,



}



# using RandomizedSearchCV for 5 fold cross validation XGboost as estimator

random_cv2=RandomizedSearchCV(estimator=xgb2,param_distributions=hyperparameter,cv=5,n_jobs=-1)



#fitting on trainx,trainy

random_cv2.fit(train_X,train_y)
random_cv2.best_estimator_.fit(train_X,train_y)



#and predicting on trainx and validationx using best estimator

train_preds6=random_cv2.best_estimator_.predict(train_X)

valid_preds6=random_cv2.best_estimator_.predict(valid_X)
#checking the model performance using accuracy score on train and validation

print("accuracy_score on train data:",accuracy_score(train_y,train_preds6))

print("accuracy_score on validaion data:",accuracy_score(valid_y,valid_preds6))



train_score6=accuracy_score(train_y,train_preds6)

valid_score6=accuracy_score(valid_y,valid_preds6)
plot_learning_curve(random_cv2.best_estimator_,'XGboost learning curve',train_X,train_y,ylim=None, cv=5,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))
results=pd.DataFrame({

    "Model":["Logistic_Regression","Decision_tree","KNN","SVC","RandomForest","XgBOOST"],

    "train_score":[train_score1,train_score2,train_score3,train_score4,train_score5,train_score6],

    "validation_score":[valid_score1,valid_score2,valid_score3,valid_score4,valid_score5,valid_score6]

})
results
test_predictions=random_cv2.best_estimator_.predict(test)
test_predictions=pd.DataFrame(test_predictions,custid.values)
test_predictions