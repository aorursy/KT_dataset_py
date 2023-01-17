#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading the csv file
df=pd.read_csv('../input/train.csv')
df.head()
#dropping irrelivent columns from dataframe.
df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
df.head()
#for getting information about different columns
df.info()
#Rounding off Fare column to one decimal place
df.Fare=df.Fare*10
df.Fare=df.Fare.astype(int)
df.Fare=df.Fare/10
df.head()
#converting text data of sex column into numeric
pd.get_dummies(df.Sex,prefix='Sex').head()
#converting text data of embarked column into numeric
pd.get_dummies(df.Embarked).head()
#creating new dataframe with numeric features for sex and embarked
df_new=pd.concat([df,pd.get_dummies(df.Sex,prefix='Sex'),pd.get_dummies(df.Embarked,prefix='Embarked')],axis=1)
df_new.head()
#dropping the text column for sex and embarked along with one extra column from sex and embarked, 
#as they does not give any additional information i.e they are highly correlated
df_new.drop(['Sex','Embarked','Sex_female','Embarked_Q'],axis=1,inplace=True)
df_new.head()
#filling the NaN value of cabin with 0
df_new.Cabin.fillna(value=0,inplace=True)
df_new.head()
#replacing all other values with 1 usin regular expression
df_new.Cabin=df_new.Cabin.str.replace('[A-Z].*','1')
df_new.Cabin.fillna(value=0,inplace=True)
df_new.head()
#converting the value of cabin from string to int
df_new.Cabin=df_new.Cabin.astype(int)
#getting info about all column.we can see that all column is now converted into int
df_new.info()
#seperating the response and feature vector,here X is feature
X=df_new.drop(['Survived'],axis=1)
X.head()
#here y is response
y=df_new.Survived
y.head()
#importing train_test_split from sklearn.linear model
from sklearn.model_selection import train_test_split
#splitting into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#getting info about training set
X_train.info()
#filling the value of nan value in age column with mean of age.
temp=X_train.Age.mean()
temp
X_train.Age.fillna(value=29,inplace=True)
#new X_trtain
X_train.head()
X_train.info()
#filling the nan value in test data with mean of age
temp=X_test.Age.mean()
temp
X_test.Age.fillna(value=31,inplace=True)
X_test.info()
#converting age from float to int
X_train.Age=X_train.Age.astype(int)
X_test.Age=X_test.Age.astype(int)
#final x_train with no NaN and all numeric value
X_train.head()
#final x_train with no NaN and all numeric value
X_test.head()
#importing random forest classifier
from sklearn.ensemble import RandomForestClassifier
#instantiating random forest classifier
rn=RandomForestClassifier()
#fitting the random forest classifier
rn.fit(X_train,y_train)
#predicting the result
pred=rn.predict(X_test)
#importing accuracy score from metrics to evaluate the accuracy of the model
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,pred)
#accuracy given by randomforest
score
X.info()
#filling the nan value of age
X.Age.mean()
X.Age=X.Age.fillna(value=30)
X.head()
#converting age to int from float
X.Age=X.Age.astype(int)
X.info()
y.shape
#calculating the accuracy of random forest foe different value of hyperparameter n_estimator
from sklearn.model_selection import cross_val_score
n_estimator_list=list(range(5,50,5))
a=[]
for i in n_estimator_list:
    rn=RandomForestClassifier(n_estimators=i)
    scores=cross_val_score(rn,X,y,cv=10)
    a.append(scores.mean())
print(a)    
#plotting a graph to show relation between n_estimator and accuracy score
plt.plot(n_estimator_list,a)
#calculating the accuracy of random forest for different value of hyperparameter max_depth
depth_list=list(range(2,10,1))
b=[]
for i in depth_list:
    rn=RandomForestClassifier(n_estimators=10,max_depth=i)
    scores=cross_val_score(rn,X,y,cv=10)
    b.append(scores.mean())
print(b)   
#plotting a graph to show relation between n_estimator and accuracy score
plt.plot(depth_list,b)
#instantiating random forest with optimal value of n_estimator and max_depth found by two upper plots
rn=RandomForestClassifier(n_estimators=10,max_depth=6)
#fitting random forest
rn.fit(X,y)
#reading the test data to which we have to submit the results
test_df=pd.read_csv('../input/test.csv')
#first five rows of test data
test_df.head()
#shape of test data
test_df.shape
temp=test_df
#dropping irrelevent columns which might not be useful in predicting the response
test_df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
#getting info about test data
test_df.info()
#we have some inf value in fare and age columns,which can't be handled by fillna function of pandas 
#so first we will have to convert that inf into nan,that's what "use_inf_as_null" does
with pd.option_context('mode.use_inf_as_null', True):
   test_df.Fare.fillna(value=test_df.Fare.mean(),inplace=True)
test_df.info()
#doing the same thing toage column
with pd.option_context('mode.use_inf_as_null', True):
   test_df.Age.fillna(value=test_df.Age.mean(),inplace=True)
test_df.info()
#rounding fare to one decimal place
test_df.Fare=test_df.Fare*10
test_df.Fare=test_df.Fare.astype(int)
test_df.Fare=test_df.Fare/10
#converting age from float to int
test_df.Age=test_df.Age.astype(int)
test_df.head()
##creating new dataframe with numeric features for sex and embarked
test_df=pd.concat([test_df,pd.get_dummies(test_df.Sex,prefix='Sex'),pd.get_dummies(test_df.Embarked,prefix='Embarked')],axis=1)
test_df.head()
##dropping the text column for sex and embarked along with one extra column from sex and embarked, 
#as they does not give any additional information i.e they are highly correlated
test_df.drop(['Sex','Sex_female','Embarked','Embarked_Q'],axis=1,inplace=True)
test_df.head()
##replacing all other values with 1 usin regular expression
test_df.Cabin=test_df.Cabin.str.replace('[A-Z].*','1')
test_df.Cabin=test_df.Cabin.fillna(value=0)
test_df.head(10)
#cheking the no. of one and zeros
test_df.Cabin.value_counts()
#predicting result using random_forest
pred_class_rn=rn.predict(test_df)
temp=pd.read_csv('../input/test.csv')
#converting the predicted response into detaframe for submission
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':pred_class_rn}).to_csv('predictions_rn_11.csv',index=False)
#importing gradient boosting clasifier
from sklearn.ensemble import GradientBoostingClassifier
X.head()
X.info()
#creating a new feature named family by adding parents,children ans siblings
X['family']=X.SibSp+X.Parch
X.head()
#predicting results by gradientboosting using k_fold cross validation
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X,y,cv=5)
print(scores.mean())
#predicting the result excluding family feature to check if accuracy is incresing or not
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['family'],axis=1),y,cv=5)
print(scores.mean())
#trying different combinations of features this time keeping family and dropping sibsp and parch.
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['SibSp','Parch'],axis=1),y,cv=5)
print(scores.mean())
#dropping three columns and then finding the accuracy
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['SibSp','Parch','Cabin'],axis=1),y,cv=5)
print(scores.mean())
#splitting into training and testing data.
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=42)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
#using Grid searchCV to find the best hyperparameter learning rate for Gradient Boosting
lr=[0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.85,0.90,0.95,1.0]
from sklearn.model_selection import GridSearchCV
gb=GradientBoostingClassifier(n_estimators=100)
param_grid=dict(learning_rate=lr)
grid=GridSearchCV(gb,param_grid,cv=10,scoring='accuracy',return_train_score=True)
grid.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)
#finding the best score
grid.best_score_
#finding the best parameter used in finding that best score
grid.best_params_
#finding best value of two hyperparameters this time using grid search.learning_rate and max_depth
max_depth=list(range(2,8,1))
param_grid=dict(learning_rate=lr,max_depth=max_depth)
gb=GradientBoostingClassifier(n_estimators=100)
grid=GridSearchCV(gb,param_grid,cv=10,scoring='accuracy',return_train_score=True)
grid.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)
#finding the best score
grid.best_score_
#finding the best parameter used to calculate this best score
grid.best_params_
#evaluating the result on testing set using the best parameters learned.
gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=3)
gb.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)
pred_clas=gb.predict(test_x.drop(['SibSp','Parch'],axis=1))
accuracy_score(test_y,pred_clas)
#first five of the testing data to which we have to submit the results
test_df.head()
#creating family feature from sibsp and parch
test_df['family']=test_df.SibSp+test_df.Parch
# before predicting on the test dataframe we should first train our best model on whole data available.
X.head()
#training our GradientBoosing model on whole dataset
gb.fit(X.drop(['SibSp','Parch'],axis=1),y)
#test df with new added feature 'family'.
test_df.head()
#predicting the final response from gradientboosting 
final_pred=gb.predict(test_df.drop(['SibSp','Parch'],axis=1))
#converting the response into Dataframe or csv file to submit it to kaggle.
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':final_pred}).to_csv('predictions_gb_13.csv',index=False)
from sklearn.linear_model import LogisticRegression
X.head()
#checking the accuracy of logisticregression on training data
lg=LogisticRegression(C=1.0)
scores=cross_val_score(lg,X.drop(['SibSp','Parch'],axis=True),y,cv=10)
print(scores.mean())
#training lg on whole data
lg.fit(X.drop(['SibSp','Parch'],axis=1),y)
#predicting the final response using logistic regression
pred_class_lg=lg.predict(test_df.drop(['SibSp','Parch'],axis=1))
#predicted response by logisticregression
pred_class_lg
#predicted response by Gradient Boosting
pred_class_gb=final_pred
pred_class_gb
#predictyed response by random forest
pred_class_rn
#converting all the responses from three models into dataframe to apply maxVoting ensambling technique
predicted_df=pd.DataFrame({'pred_class_gb':pred_class_gb,'pred_class_rn':pred_class_rn,'pred_class_lg':pred_class_lg})
predicted_df.head(5)
#generating new predicted response by takin response which is in majority.i.e takin the mode value of the predicted response
final_pred_clas=predicted_df.mode(axis=1,numeric_only=True)
final_pred_clas.head()  
#finally converting into numpy array to submit it to kaggle
final_pred_arr=np.resize(final_pred_clas,(418,))
final_pred_arr
#converting numpy array into csv file or dataFrame.this file we can submit as our final response to kaggle.
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':final_pred_arr}).to_csv('predictions_max_voting_14.csv',index=False)
