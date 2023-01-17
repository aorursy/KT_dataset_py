import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
#Quick look at the training dataframe
df_train.head(5)
#keep the important features 
    #(this step is based on intuition but should be revised after serious analysis)
df_train = df_train[['PassengerId','Pclass','Survived','Sex','Age','SibSp','Parch','Fare','Embarked']]
#description of features
df_train.describe()

#There are clearly some missing values in feature 'Age' (I'll replace them with Zeros)
print("number of missing values from Age = ",df_train['Age'].isnull().sum(),"\n")
#eliminate missing values
df_train['Age'] = df_train['Age'].fillna(0)
#search for categorical non numerical features
df_train.describe(include=['O']) #O mean Object type , to check the type : column.dtype

#replace with numerical features
df_train['Sex']=df_train['Sex'].map(lambda x : 1 if x=='male' else 0 )
df_train['Embarked']=df_train['Embarked'].map(lambda x : 1 if x=='S' else 2 if x=='C' else 3)
#set id
df_train.set_index('PassengerId',inplace=True)
#Create simple model
model = GradientBoostingClassifier(random_state=0)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X_train = df_train[features]
y_train = df_train['Survived']

#fit our model models
model.fit(X_train,y_train)
#preprocessing the test dataframe
df_test['Sex']=df_test['Sex'].map(lambda x : 1 if x=='male' else 0 )
df_test['Embarked']=df_test['Embarked'].map(lambda x : 1 if x=='S' else 2 if x=='C' else 3)
df_test['Age'] = df_test['Age'].fillna(0)
df_test = df_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#search for missing values
df_test.describe()
#one missing value in Fare feature
df_test['Fare'] = df_test['Fare'].fillna(-1) #all other values are positives (min == 0)
#get predictions
X_test = df_test[features]
predictions = model.predict(X_test)
#prediction dataframe
df_predictions = pd.DataFrame(predictions,columns =['Survived'])
df_predictions['PassengerId'] = df_test.PassengerId
df_predictions = df_predictions[['PassengerId','Survived']]
df_predictions
df_predictions.to_csv ('predictions.csv', index = False, header=True)