import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
import re
#Sklearn OneHot Encoder to Encode categorical integer features
from sklearn.preprocessing import OneHotEncoder
#Sklearn train_test_split to split a set on train and test 
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split      # for old sklearn version use this to split a dataset 
# Random Forest Classifier from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#Import the training data set
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.head()
data.isnull().sum()
test.isnull().sum()
#Construct an X matrix
x_train = data[['Name', 'Pclass','Sex','Age','Parch','SibSp','Embarked', 'Fare', 'Cabin', 'Survived']].copy()
x_test = test[['Name', 'Pclass','Sex','Age','Parch','SibSp','Embarked', 'Fare', 'Cabin']].copy()
x_train.shape, x_test.shape
PassengerID = np.array(test['PassengerId'])
print(x_train.Embarked.unique())
print(x_test.Embarked.unique())
set(x_train.Embarked.unique()) == set(x_test.Embarked.unique())   # CHeck that values in the train and in the test were similar
x_train = x_train.dropna(subset=['Embarked'],axis=0)
print(x_train.Embarked.unique())
set(x_train.Embarked.unique())==set(x_test.Embarked.unique())   # CHeck that values in the train and in the test were similar
x_train.Embarked = pd.factorize(x_train.Embarked)[0]
x_test.Embarked = pd.factorize(x_test.Embarked)[0]
x_train.head()
x_train.Sex = pd.factorize(x_train.Sex)[0]
x_test.Sex = pd.factorize(x_test.Sex)[0]
x_train['Family'] = x_train['SibSp'] + x_train['Parch']
x_test['Family'] = x_test['SibSp'] + x_test['Parch']

x_train['Alone'] = x_train['Family'].map(lambda x: 1 if x==0 else 0)
x_test['Alone'] = x_test['Family'].map(lambda x: 1 if x==0 else 0)
# Find a mean Age in overall data
age = pd.concat([x_test.Age, x_train.Age], axis=0)
mean = age[1].mean()
# Identify the rows with missed Age in special column
x_train['Missed_Age'] = x_train['Age'].map(lambda x: 1 if pd.isnull(x)  else 0)
x_test['Missed_Age'] = x_test['Age'].map(lambda x: 1 if pd.isnull(x) else 0)
# Fill all age values with Age mean
x_train['Age'] = x_train['Age'].fillna(mean)
x_test['Age'] = x_test['Age'].fillna(mean)
data[data.Survived==1].Age.plot.hist(alpha=0.5,color='blue',stacked=True, bins=50)
data[data.Survived==0].Age.plot.hist(alpha=0.5,color='red', stacked=True, bins=50)
plt.legend(['Survived','Died'])
plt.show()
sns.countplot(x="Survived", data=data[data['Age'].isnull()])
sns.countplot(x="Survived", data=data[data['Age'].isnull()], hue='Pclass')
def process_age(df,cut_points,label_names):
    df["Age"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,16,100]        
label_names = [0,1,2,3]

x_train = process_age(x_train,cut_points,label_names)
x_test = process_age(x_test,cut_points,label_names)
set(x_train['Age'].unique()) == set(x_test['Age'].unique())
x_train.head()
# Fill one missed fare in the train set with mean Fare for this class
x_test.loc[x_test['Fare'].isnull()]['Pclass']  # determine a Class for this passenger
# Find the mean Fare for Class 3
fare_mean = pd.concat([x_train.loc[x_train['Pclass']==3]['Fare'], x_test.loc[x_test['Pclass']==3]['Fare']], axis=0).mean()
# Fill the data gap
x_test['Fare'] = x_test['Fare'].fillna(fare_mean)
x_test.isnull().sum()
x_train['Fare'] = (x_train['Fare']/20).astype('int64')
x_test['Fare'] = (x_test['Fare']/20).astype('int64')
set(x_train['Fare'].unique()) == set(x_test['Fare'].unique()) # Check the train and test data identity
# There a lot of missed values so lets just check do passenger have a Cabin number or not
x_train['Missed_Cabin'] = x_train['Cabin'].map(lambda x: 0 if pd.isnull(x)  else 1)
x_test['Missed_Cabin'] = x_test['Cabin'].map(lambda x: 0 if pd.isnull(x) else 1)
x_train['Cabin_num'] = x_train['Cabin'].map(lambda x: 0 if pd.isnull(x)  else len(x.split()))
x_test['Cabin_num'] = x_test['Cabin'].map(lambda x: 0 if pd.isnull(x) else len(x.split()))
x_train.head()
# Lets try to extract a Title data from name using regular expression
x_train['Title'] = x_train['Name'].map(lambda x: str(re.findall("^.*[, *](.*)[.] *", x)[0]))
x_test['Title'] = x_test['Name'].map(lambda x: str(re.findall("^.*[, ](.*)[.] *", x)[0]))
x_train['Title'].unique()
sns.countplot(x="Title", data=x_train)
x_train.Title = pd.factorize(x_train.Title)[0]
x_test.Title = pd.factorize(x_test.Title)[0]
x_train.head()
x_train['Name_Len_char'] = x_train['Name'].map(lambda x: len(x))
x_train['Name_Len_words'] = x_train['Name'].map(lambda x: len(x.split()))

x_test['Name_Len_char'] = x_test['Name'].map(lambda x: len(x))
x_test['Name_Len_words'] = x_test['Name'].map(lambda x: len(x.split()))
x_train.head()
#Create Y array
y = np.array(x_train[['Survived']])
print(y.shape)
x_train=x_train.drop(['SibSp', 'Parch', 'Name', 'Cabin', 'Survived'], axis=1)
x_test=x_test.drop(['SibSp', 'Parch', 'Name', 'Cabin'],axis=1)
x_train.head()
xn_train, xn_test, yn_train, yn_test = train_test_split(x_train, y, test_size=0.3, random_state=32)
xn_train.shape, xn_test.shape, yn_train.shape, yn_test.shape
# We can optimize the parameters using special function in sclearn, but here I will do it manually
C=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750])
scores = np.zeros(C.shape)
for i in range (len(C)):
        clf = RandomForestClassifier(n_estimators = int(C[i]), max_depth=10, random_state=0, criterion='entropy') 
        clf.fit(xn_train, yn_train) 
        scores[i] = clf.score(xn_test,yn_test)
ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
print('max Score = ',scores[ind],'\noptimal C = ',C[ind])
clf = RandomForestClassifier(n_estimators = 150, max_depth=10, random_state=0, criterion='entropy') 
clf.fit(xn_train, yn_train) 
print(clf.score(xn_train,yn_train))
print(clf.score(xn_test,yn_test))
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=x_test.columns, 
                          columns=["Importance"])
print(importance)
clf = RandomForestClassifier(n_estimators = 100, max_depth=10, random_state=0, criterion='entropy') 
clf.fit(x_train, y) 
prediction = clf.predict(x_test)
print(clf.score(xn_train,yn_train))
print(clf.score(xn_test,yn_test))
print(clf.score(x_train,y))
# Submit the result

submission_df = {"PassengerId": PassengerID,
                 "Survived": prediction}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv",index=False)