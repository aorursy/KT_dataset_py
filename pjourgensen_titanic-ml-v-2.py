#Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline 
#Read in the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full = pd.concat([train,test])
#Check head
full.head()
#Check info
full.info()
#Drop 'Cabin'
full.drop('Cabin',axis=1,inplace=True)
#Check which port most of the passengers came from
full['Embarked'].value_counts()
#Fill na values with 'S'
full['Embarked'].fillna('S',inplace = True)

#Convert 'Embarked' into dummy variables and drop 'Embarked'
full = pd.concat([full,pd.get_dummies(full['Embarked'],drop_first=True,prefix='Port')],axis=1)
full.drop('Embarked',axis=1,inplace=True)
#Fill na values with average Fare
full.loc[full['Fare'].isnull(),'Fare'] = full['Fare'].mean()
#Split names into a list
full_title = full['Name'].apply(lambda x: x.split()[1])

#Function: Takes in a title and returns the title if it's among those specified; Else returns 'No Title'
def impute_title(title):
    if title not in ['Mr.','Miss.','Mrs.','Master.']:
        return 'No title'
    else:
        return title

#Assign titles, convert into dummy variables, and drop 'Name'
full_title = full_title.apply(impute_title)
full = pd.concat([full,pd.get_dummies(full_title,drop_first=True)],axis=1)
full.drop('Name',axis=1,inplace=True)
#Store test ids for later use and drop 'PassengerId'
test_id = test['PassengerId']
full.drop('PassengerId',axis=1,inplace=True)
#Convert 'Pclass' into dummy variables and drop 'Pclass'
full = pd.concat([full,pd.get_dummies(full['Pclass'],drop_first=True,prefix='Pclass')],axis=1)
full.drop('Pclass',axis=1,inplace=True)
#Convert 'Sex' into dummy variable and drop 'Sex'
full = pd.concat([full,pd.get_dummies(full['Sex'],drop_first=True)],axis=1)
full.drop('Sex',axis=1,inplace=True)
#Drop 'Ticket'
full.drop('Ticket',axis=1,inplace=True)
#Take a look at the data frame now
full.head()
#Import Linear Regression Model
from sklearn.linear_model import LinearRegression

#Fit model on data that does have an Age entry
impute_age = LinearRegression()
impute_age.fit(full[full['Age'].isnull()==False].drop(['Survived','Age'],axis=1),
               full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age'])

#Impute ages for those that were missing
ages = impute_age.predict(full[full['Age'].isnull()].drop(['Survived','Age'],axis=1))
#Compare Age Distributions with and without imputed ages
plt.figure(figsize=(13.5,6))
plt.subplot(1,2,1)
plt.hist(full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age'],
         bins=range(0,80,5),edgecolor='white')
plt.title('Without Age Imputations')
plt.xlabel('Age')

plt.subplot(1,2,2)
plt.hist(list(full[full['Age'].isnull()==False].drop('Survived',axis=1)['Age']) + list(ages),
         bins=range(0,80,5),edgecolor='white',alpha=.5)
plt.title('With Age Imputations')
plt.xlabel('Age')
#Fill dataframe in with imputed ages
full.loc[full['Age'].isnull(),'Age'] = ages
#Produce heatmap of correlations
plt.figure(figsize=(16,8))
sns.heatmap(full.corr(),annot=True,cmap='viridis')
plt.tight_layout
#Illustrate relationship between family size and survival rate
plt.figure(figsize=(13.5,6))
sns.countplot((full['SibSp'] + full['Parch'] + 1),hue=full['Survived'],palette='viridis')
plt.xlabel('Family Size')
#Create new column for family size
full['Family'] = (full['SibSp'] + full['Parch'] + 1)

#Function: Takes in family size and returns corresponding description
def impute_alone(x):
    if x == 1:
        return 'Alone'
    elif x > 4:
        return 'Large Family'
    else:
        return 'Small Family'

#Label each passenger's family size
full['Family'] = full['Family'].apply(impute_alone)
#Again, illustrate relationship between family size and survival rate
plt.figure(figsize=(13.5,6))
sns.countplot(full['Family'],hue=full['Survived'],palette='viridis')
#Convert into dummy variable and drop 'SibSp', 'Parch', and 'Family'
full = pd.concat([full,pd.get_dummies(full['Family'],drop_first=True)],axis=1)
full.drop(['SibSp','Parch','Family'],inplace=True,axis=1)
#Take a look
full.head(10)
#Standardize Age and Fare
full['Age'] = (full['Age'] - full['Age'].mean()) / full['Age'].std()
full['Fare'] = (full['Fare'] - full['Fare'].mean()) / full['Fare'].std()
#Split the data back into train and test sets
train = full.iloc[0:len(train)]
test = full.iloc[len(train):len(full)]
#Scikit imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Split the data 'train' data into train and test sets
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
def error(clf, X, y, ntrials=100, test_size=0.2) :
    train_error = 0
    test_error = 0
    
    for i in range(ntrials):
        X_, X_t, y_, y_t = train_test_split(X, y, test_size=test_size, random_state=i)
        model = clf
        model.fit(X_,y_)
        train_pred = model.predict(X_)
        test_pred = model.predict(X_t)
        train_error += 1 - accuracy_score(y_, train_pred, normalize=True)
        test_error += 1 - accuracy_score(y_t, test_pred, normalize=True)
    
    train_error /= ntrials
    test_error /= ntrials
    
    return train_error, test_error
depth = list(range(1,21,1))
depth_train_error = []
depth_test_error = []
    
for i in depth:
    results = error(DecisionTreeClassifier(criterion='entropy',max_depth=i),X,y,ntrials=100,test_size=.2)
    depth_train_error.append(results[0])
    depth_test_error.append(results[1])

plt.plot(depth,depth_train_error,label='Train Error')
plt.plot(depth,depth_test_error,label='Test Error')
plt.title('Error vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Error')
plt.legend()
#plt.savefig('Depth.png')
plt.show()
#Fit and predict with Decision Tree
dtree = DecisionTreeClassifier(criterion='entropy',max_depth=3)
dtree.fit(X_train,y_train)
pred_dtree = dtree.predict(X_test)

scores = cross_val_score(dtree, X_train, y_train, cv=10)
print(scores.mean())
print('\n')
print(confusion_matrix(y_test,pred_dtree))
print('\n')
print(classification_report(y_test,pred_dtree))
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(X_train.columns)

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 
pred_final = dtree.predict(test.drop('Survived',axis=1))
pred_final = [int(x) for x in pred_final]
#Create Submission
submission = pd.DataFrame(
    {'PassengerId' : test_id,
     'Survived' : pred_final}
)
#Make sure everything looks good
submission.head()
#Store it
submission.to_csv('Submission',index=False)
