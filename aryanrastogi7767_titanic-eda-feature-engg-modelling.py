import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
train = pd.read_csv("../input/titanic/train.csv")
print(train.shape)
train.head()
train.info()
train.describe()
print("Number of people who died: ",train['Survived'].value_counts()[0])       # Imbalanced Data
print("Number of people who survived: ",train['Survived'].value_counts()[1])
sns.heatmap(train.isnull()==True)
train.drop(labels= ['Cabin','PassengerId','Name'], axis = 1,inplace = True)
train['Ticket'].nunique() #there are 681 unique values of tickets available but there are 891 passengers.
train[train['Ticket']== "349909"]
# Summing up the no. of siblings, spouse, parents ,children to get the total no. of family members onboard.
train['Family_onboard'] =  train['SibSp']+train['Parch']  
train.drop(labels=['Ticket'],axis = 1,inplace = True)   #Dropping the Ticket  column
train.head()
print('No. of null values in the Age column:',train[train['Age'].isnull()==True].shape[0]
      ,"i.e",train[train['Age'].isnull()==True].shape[0]/891 *100,"%")
grp = train.groupby(['Sex','Pclass'])
grp['Age'].median()    # Grouping the data by sex and passenger class and finding median age of the class.
train['Age'].fillna(value = 0,inplace = True)
#Filling the appropriate age where there were null values in the age column with median values of that sex and class.
start = datetime.now()
for i in range(0,891):
    if(train['Sex'][i]=='male' and train['Pclass'][i]==1):
        if(train['Age'][i]==0):
            train['Age'][i] = 40.0
            continue
    elif(train['Sex'][i]=='male' and train['Pclass'][i]==2):
        if(train['Age'][i]==0):
            train['Age'][i] = 30.0
            continue
    elif(train['Sex'][i]=='male' and train['Pclass'][i]==3):
        if(train['Age'][i]==0):
            train['Age'][i] = 25.0
            continue
    elif(train['Sex'][i]=='female' and train['Pclass'][i]==1):
        if(train['Age'][i]==0):
            train['Age'][i] = 35.0
            continue
    elif(train['Sex'][i]=='female' and train['Pclass'][i]==2):
        if(train['Age'][i]==0):
            train['Age'][i] = 28.0 
            continue
    elif(train['Sex'][i]=='female' and train['Pclass'][i]==3):
        if(train['Age'][i]==0):
            train['Age'][i] = 21.0
            continue
print("All the null values of the age column have been handled successfully!")
print("Time taken to run this cell: ",datetime.now()-start)
sns.boxplot(train['Age'])
print("No. of Children onboard:",train[train['Age']<=18]['Age'].count())
print("No. of Adults onboard:",train[train['Age']>18]['Age'].count())
train['Age'].hist(bins = 30)
data = [train]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7
    
    dataset['Age'] = dataset['Age'].astype(str)
    dataset.loc[ dataset['Age'] == '0', 'Age'] = "Children"
    dataset.loc[ dataset['Age'] == '1', 'Age'] = "Teens"
    dataset.loc[ dataset['Age'] == '2', 'Age'] = "Youngsters"
    dataset.loc[ dataset['Age'] == '3', 'Age'] = "Young Adults"
    dataset.loc[ dataset['Age'] == '4', 'Age'] = "Adults"
    dataset.loc[ dataset['Age'] == '5', 'Age'] = "Middle Age"
    dataset.loc[ dataset['Age'] == '6', 'Age'] = "Senior"
    dataset.loc[ dataset['Age'] == '7', 'Age'] = "Retired"
train['Age'].value_counts()
train.head()
train['Pclass'].value_counts()
# Maximum number of passengers belonged to the 3rd class and were probably crew members and staff
train['Pclass'].hist(by=train['Survived'])
train['Pclass'].hist(by=train['Sex'])
data = [train]

for dataset in data:
    dataset['Pclass'] = dataset['Pclass'].astype(str)
    dataset.loc[ dataset['Pclass'] == '1', 'Pclass'] = "Class1"
    dataset.loc[ dataset['Pclass'] == '2', 'Pclass'] = "Class2"
    dataset.loc[ dataset['Pclass'] == '3', 'Pclass'] = "Class3"
train.head()
train['Embarked'].hist()  # Most of the passengers boarded the titanic at Southampton
train['Embarked'].hist(by=train['Survived'])
train['Embarked'].value_counts()
train['Embarked'].fillna(train['Embarked'].mode(),inplace = True)
# train['Embarked'] = train['Embarked'].map({"S":0,'C':1,'Q':2})
# train['Embarked_C'] = train['Embarked'].map({"S":0,'C':1,'Q':0})
# train['Embarked_Q'] = train['Embarked'].map({"S":0,'C':0,'Q':1})
# train.drop(labels = ['Embarked'],axis=1,inplace = True)
train['Fare'].describe() # Maximum fare on the titanic is 512.32 Euros.
train[train['Fare']==512.329200]    # All 3 of these are 1st class passengers and hence paid the highest Fare.
grp = train.groupby(['Pclass'])
grp['Fare'].mean() 
data = [train]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset['Fare'] = dataset['Fare'].astype(str)
    dataset.loc[ dataset['Fare'] == '0', 'Fare'] = "Extremely Low"
    dataset.loc[ dataset['Fare'] == '1', 'Fare'] = "Very Low"
    dataset.loc[ dataset['Fare'] == '2', 'Fare'] = "Low"
    dataset.loc[ dataset['Fare'] == '3', 'Fare'] = "High"
    dataset.loc[ dataset['Fare'] == '4', 'Fare'] = "Very High"
    dataset.loc[ dataset['Fare'] == '5', 'Fare'] = "Extremely High"
train['Family_onboard'].unique()
train['is_alone'] = train['Family_onboard'].map({1:0,  0:1,  4:0,  2:0,  6:0,  5:0,  3:0,  7:0, 10:0})
sns.heatmap(train.corr(),annot = True)  # Checking for multicollinearity in the data.
train.drop(labels=['SibSp','Parch','Family_onboard'],axis =1,inplace = True)
train['is_alone']=train['is_alone'].map({0:"no",1:'yes'})
col_list = list(train.select_dtypes(include=['object']).columns)
for i in col_list:
    train = pd.concat([train,pd.get_dummies(train[i], prefix=i)],axis=1)
    train.drop(i, axis = 1, inplace=True)
train.head()
train.columns
X = train[['Pclass_Class1', 'Pclass_Class2', 'Pclass_Class3',\
       'Sex_female', 'Sex_male', 'Age_Adults', 'Age_Children',\
       'Age_Middle Age', 'Age_Retired', 'Age_Senior', 'Age_Teens',\
       'Age_Young Adults', 'Age_Youngsters', 'Fare_Extremely High',\
        'Fare_Extremely Low', 'Fare_High', 'Fare_Low', 'Fare_Very High',\
        'Fare_Very Low', 'Embarked_C', 'Embarked_Q', 'Embarked_S','is_alone_no', 'is_alone_yes']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
y_train.value_counts()   # we will have to give weights to make the classes balanced.
y_test.value_counts()
# This function trains the specified classifier with default parameters and prints train and test accuracy.

def basic_train_eval(classifier,xtrain,xtest,ytrain,ytest):
    model = classifier
    model.fit(xtrain,ytrain)
    train_pred = model.predict(xtrain)
    test_pred = model.predict(xtest)
    print("Training and predicting on default parameters: \n")
    print("Accuracy on train data: ",accuracy_score(ytrain,train_pred))
    print("Accuracy on test data: ",accuracy_score(ytest,test_pred))
    print("\n")
    print(classification_report(ytest,test_pred))
    print(confusion_matrix(ytest,test_pred))
basic_train_eval(LogisticRegression(class_weight='balanced'),X_train,X_test,y_train,y_test)
basic_train_eval(SVC(),X_train,X_test,y_train,y_test)  # OVERFITTING
basic_train_eval(GaussianNB(),X_train,X_test,y_train,y_test)
basic_train_eval(KNeighborsClassifier(),X_train,X_test,y_train,y_test) # OVERFITTING
basic_train_eval(RandomForestClassifier(),X_train,X_test,y_train,y_test) # OVERFITTING
basic_train_eval(xgb.XGBClassifier(),X_train,X_test,y_train,y_test) # OVERFITTING
# This function takes the classifier and parameter grid as parameters and performs Grid Search and 5 fold Cross Validation
# and gives us the best score.

def grid_tuning(classifier,param_grid):
    start = datetime.now()
    grid = GridSearchCV(estimator = classifier,param_grid=param_grid,cv=5)
    grid.fit(X_train,y_train)
    print("Best Parameters: ",grid.best_params_,"\n")
    train_pred = grid.predict(X_train)
    test_pred = grid.predict(X_test)
    print("Accuracy on train data: ",accuracy_score(y_train,train_pred))
    print("Accuracy on test data: ",accuracy_score(y_test,test_pred),"\n")
    print(classification_report(y_test,test_pred))
    print(confusion_matrix(y_test,test_pred),"\n")
    print("Time taken to run this cell: ",datetime.now()-start)
lr_param = {'penalty':['l1','l2'],'C':[0.0001,0.001,0.01,0.1,1,10],'solver':['liblinear'],'class_weight':['balanced']}
grid_tuning(LogisticRegression(),param_grid=lr_param)
svc_param={"C":[0.001,0.01,0.1,1,10],'class_weight':['balanced']}
grid_tuning(SVC(),param_grid=svc_param)   
rfc_param= {'n_estimators':[5,10,20,40,70,100,150,200],"criterion":['gini','entropy'],'class_weight':['balanced'],\
            'max_depth':[3,4,5,6,7,8]}
grid_tuning(RandomForestClassifier(),param_grid=rfc_param)   
xgb_param = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
grid_tuning(xgb.XGBClassifier(),param_grid=xgb_param)   
test= pd.read_csv('../input/titanic/test.csv')
test.head()
test.info()
test['Fare'].fillna(value=20,inplace = True)
test.info()
passId = test['PassengerId']
# Performing all the preprocessing steps on the test data to get it in the right format for prediction.
start = datetime.now()
test.drop(labels=['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace=True)
test['Age'].fillna(value = 0,inplace = True)
for i in range(0,test.shape[0]):
    if(test['Sex'][i]=='male' and test['Pclass'][i]==1):
        if(test['Age'][i]==0):
            test['Age'][i] = 40.0
            continue
    elif(test['Sex'][i]=='male' and test['Pclass'][i]==2):
        if(test['Age'][i]==0):
            test['Age'][i] = 30.0
            continue
    elif(test['Sex'][i]=='male' and test['Pclass'][i]==3):
        if(test['Age'][i]==0):
            test['Age'][i] = 25.0
            continue
    elif(test['Sex'][i]=='female' and test['Pclass'][i]==1):
        if(test['Age'][i]==0):
            test['Age'][i] = 35.0
            continue
    elif(test['Sex'][i]=='female' and test['Pclass'][i]==2):
        if(test['Age'][i]==0):
            test['Age'][i] = 28.0 
            continue
    elif(test['Sex'][i]=='female' and test['Pclass'][i]==3):
        if(test['Age'][i]==0):
            test['Age'][i] = 21.0
            continue     
print("Null values of Age column handled successfully!!\n")            
data = [test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7
    
    dataset['Age'] = dataset['Age'].astype(str)
    dataset.loc[ dataset['Age'] == '0', 'Age'] = "Children"
    dataset.loc[ dataset['Age'] == '1', 'Age'] = "Teens"
    dataset.loc[ dataset['Age'] == '2', 'Age'] = "Youngsters"
    dataset.loc[ dataset['Age'] == '3', 'Age'] = "Young Adults"
    dataset.loc[ dataset['Age'] == '4', 'Age'] = "Adults"
    dataset.loc[ dataset['Age'] == '5', 'Age'] = "Middle Age"
    dataset.loc[ dataset['Age'] == '6', 'Age'] = "Senior"
    dataset.loc[ dataset['Age'] == '7', 'Age'] = "Retired"   
print('Converted Age column to categorical successfully!!\n')
for dataset in data:
    dataset['Pclass'] = dataset['Pclass'].astype(str)
    dataset.loc[ dataset['Pclass'] == '1', 'Pclass'] = "Class1"
    dataset.loc[ dataset['Pclass'] == '2', 'Pclass'] = "Class2"
    dataset.loc[ dataset['Pclass'] == '3', 'Pclass'] = "Class3" 
print('Converted Pclass column to categorical successfully!!\n')
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset['Fare'] = dataset['Fare'].astype(str)
    dataset.loc[ dataset['Fare'] == '0', 'Fare'] = "Extremely Low"
    dataset.loc[ dataset['Fare'] == '1', 'Fare'] = "Very Low"
    dataset.loc[ dataset['Fare'] == '2', 'Fare'] = "Low"
    dataset.loc[ dataset['Fare'] == '3', 'Fare'] = "High"
    dataset.loc[ dataset['Fare'] == '4', 'Fare'] = "Very High"
    dataset.loc[ dataset['Fare'] == '5', 'Fare'] = "Extremely High"  
print('Converted Fare column to categorical successfully!!\n')    
test['Family_onboard'] =  test['SibSp']+test['Parch']    
test['is_alone'] = test['Family_onboard'].map({1:0,  0:1,  4:0,  2:0,  6:0,  5:0,  3:0,  7:0, 10:0})
test.drop(labels=['SibSp','Parch','Family_onboard'],axis =1,inplace = True)
test['is_alone']=test['is_alone'].map({0:"no",1:'yes'})
col_list = list(test.select_dtypes(include=['object']).columns)
for i in col_list:
    test = pd.concat([test,pd.get_dummies(test[i], prefix=i)],axis=1)
    test.drop(i, axis = 1, inplace=True)     
print('Preprocessing Completed Successfully!\n')
print('Time Taken in preprocessing: ',datetime.now()-start)
# Training the XGBoost model on best obtained parameters.
xgbc = xgb.XGBClassifier(colsample_bytree= 0.7, gamma= 0.1, learning_rate=0.05, max_depth= 4, min_child_weight= 5)
xgbc.fit(X,y)
xgbc_train_pred = xgbc.predict(X)
# xgbc_test_pred = xgbc.predict(X_test)
print("Training Accuracy: ",accuracy_score(y,xgbc_train_pred))
# print("Test Accuracy: ",accuracy_score(y_test,xgbc_test_pred))
result = xgbc.predict(test)
sub1 = pd.DataFrame(passId,columns = ['PassengerId'])
sub1['Survived'] = result
sub1.to_csv("Sub1.csv",index = False)
gxgb = GridSearchCV(xgb.XGBClassifier(),param_grid=xgb_param,cv = 5)
gxgb.fit(X,y)
result_new = gxgb.predict(test)
sub2 = pd.DataFrame(passId,columns = ['PassengerId'])
sub2['Survived'] = result_new
sub2.to_csv("Sub2.csv",index = False)
