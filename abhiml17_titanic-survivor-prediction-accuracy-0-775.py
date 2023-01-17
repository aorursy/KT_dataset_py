#Load Modules
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline

#Read the train.csv
tit_df=pd.read_csv('../input/titanic/train.csv')
#check
tit_df.head(5)
#get a feel of the data
tit_df.describe()
tit_df.info()
#make a copy and do sanitising feature engineering on the copy
tit_df_san=tit_df.copy()
tit_df_san.info()
#1.Lot of values Missing For Cabin column . Hence , dropping it
tit_df_san.drop('Cabin',axis=1,inplace=True) 
tit_df_san.head(2)
#find correlation between 'Survived' and Other Columns  to see which columns can be useful
tit_df_san.corr().loc['Survived']
tit_df['Embarked'].unique()
#convert categorical to numerical 
tit_df_san['Embarked'].replace(['S','C','Q'],[1,2,3],inplace=True)
tit_df_san['Sex'].replace(['male','female'],[1,0],inplace=True)

tit_df_san.head(3)
#Dropping Name and Ticket columns 
tit_df_san.drop(['Name','Ticket'],axis=1,inplace=True)  
tit_df_san.head()
# check details about rows whose Embarked value is Nan
tit_df_san[tit_df_san['Embarked'].isnull()]
# try to find out the mean of all passengers whose Fare value is between 70 and 90 ,Pclass =1
tit_df_san[(tit_df_san['Fare']>70.0) & (tit_df_san['Fare']<90.0) & (tit_df_san['Pclass']==1) ].groupby('Embarked').mean()
#from above most probably the passengers embarked from 'C' Numerical value 2.0
tit_df_san['Embarked'].fillna(value=2.0,axis=0,inplace=True)
tit_df_san.loc[[61,829]]
tit_df_san.info()
# create a new column 'num of relatives' = 'Sibsp'+ 'Parch'
tit_df_san['num of relatives']=tit_df_san['SibSp']+tit_df_san['Parch']
# delete 'Sibsp' and 'parch' Column as 'num of relatives' column  should be enough
tit_df_san.drop(['SibSp','Parch'],axis=1,inplace=True)  
#graphical rep of corr()
figure=plt.figure(figsize=(10,5))

sb.heatmap(tit_df_san.corr(),annot=True)
tit_df_san.head(2)
# create a multi index data structure which will have the avg age of travellers based on Sex and Pclass
age_mapper=tit_df_san[tit_df_san['Age']>0.0].groupby(['Sex','Pclass']).mean()['Age']
age_mapper
age_mapper.loc[0].loc[3]
# Update age with Nan Values with average age based on age_mapper data structure
for i in tit_df_san[tit_df_san['Age'].isnull()].index:
    tit_df_san.loc[i, 'Age'] = age_mapper.loc[tit_df_san.loc[i,'Sex']].loc[tit_df_san.loc[i,'Pclass']]


    
tit_df_san.head(2)
tit_df_san.info()
##Model Training starts
from sklearn.model_selection import train_test_split
tit_df_san.columns
# scaling the features matrix
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X=tit_df_san[['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives']]
y=tit_df_san['Survived']           
std_scaler.fit(X)
scaled_X=std_scaler.transform(X)
final_X=pd.DataFrame(scaled_X,columns=['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives'])
final_X.head(2)

X_train, X_test, y_train, y_test = train_test_split(final_X,y,test_size=0.30)

# try Support Vector Machine
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#using Logistic Regression
from sklearn.linear_model import LogisticRegression
lgrm=LogisticRegression()
lgrm.fit(X_train,y_train)
preds=lgrm.predict(X_test)
print(classification_report(y_test,preds))
#will choose SVM and train the model on whole test.csv as accuracy is 86%
model = SVC()
model.fit(final_X,y)
# Sanitising test set now
tit_df_test=pd.read_csv('../input/titanic/test.csv')

tit_df_test.head(2)
tit_df_test.describe()
tit_df_test.info()
tit_df_test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True) #dropping
tit_df_test.head(2)
tit_df_test.info()
#convert categorical to numerical 
tit_df_test['Embarked'].replace(['S','C','Q'],[1,2,3],inplace=True)
tit_df_test['Sex'].replace(['male','female'],[1,0],inplace=True)
tit_df_test.head(2)
#check the row whose Fare details is missing
tit_df_test[tit_df_test['Fare'].isnull()]
# find out avg fare of passengers  Pclass=3 and Embarked=1 
tit_df_test[(tit_df_test['Pclass']==3) & (tit_df_test['Embarked']==1) ].mean()
#fill the missing value with the mean
tit_df_test['Fare'].fillna(value=13.91,axis=0,inplace=True)
tit_df_test.loc[152]
# Update age with Nan Values with average age based on age_mapper data structure
for i in tit_df_test[tit_df_test['Age'].isnull()].index:
    tit_df_test.loc[i, 'Age'] = age_mapper.loc[tit_df_test.loc[i,'Sex']].loc[tit_df_test.loc[i,'Pclass']]
# create a new column 'num of relatives' = 'Sibsp'+ 'Parch'
tit_df_test['num of relatives']=tit_df_test['SibSp']+tit_df_test['Parch']
# delete 'Sibsp' and 'parch' Column as 'num of relatives' column  should be enough
tit_df_test.drop(['SibSp','Parch'],axis=1,inplace=True) 
tit_df_test.info()

# scaling the features matrix

std_scaler = StandardScaler()
X=tit_df_test[['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives']]
std_scaler.fit(X)
scaled_X=std_scaler.transform(X)
final_test_X=pd.DataFrame(scaled_X,columns=['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives'])
final_test_X.head(2)
predictions = model.predict(final_test_X)
len(predictions)
preds_test_whole_df=pd.DataFrame(data=predictions,columns=['Survived'])

result_df=pd.DataFrame()
result_df['PassengerId']=tit_df_test['PassengerId']
result_df['Survived']=preds_test_whole_df['Survived']
result_df.head()
result_df.to_csv('predictions.csv',index=False)
result_df['Survived'].value_counts()
