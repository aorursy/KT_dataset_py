# Import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

# % matplotlib inline



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



# Read train and test data

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
print("Train: \n", train_df.isnull().sum())

print("\nTest: \n", test_df.isnull().sum())
# Fill missing values because test has only one missing value

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

# The entire data

data_df = train_df.append(test_df)

passenger_id=test_df.PassengerId



# Drop PassengerID because will not be usefull

train_df.drop(["PassengerId"], axis=1, inplace=True)

test_df.drop(["PassengerId"], axis=1, inplace=True)

test_df.shape
sns.boxplot(x='Survived',y='Fare',data=train_df)
train_df=train_df[train_df['Fare']<400]



train_df['Sex'] = pd.Categorical(train_df.Sex).codes

test_df['Sex'] = pd.Categorical(test_df.Sex).codes
# The mean() strategy.

for name_string in data_df['Name']:

    data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)



# Replacing rare titles with more common ones.

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)
data_df.groupby('Title')['Age'].median()
data_df['Title'].value_counts()

train_df['Title']=data_df['Title'][:891]

test_df['Title']=data_df['Title'][891:]



titles=['Mr','Miss','Mrs','Master','Rev','Dr']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].mean()[titles.index(title)]

    print(age_to_impute)

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

data_df.isnull().sum()



train_df['Age']=data_df['Age'][:891]

test_df['Age']=data_df['Age'][891:]

test_df.isnull().sum()
train_df.head()
## A good feature to create: `family_size`

train_df['family_size'] = train_df.SibSp + train_df.Parch+1

test_df['family_size'] = test_df.SibSp + test_df.Parch+1
def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a



train_df['family_group'] = train_df['family_size'].map(family_group)

test_df['family_group'] = test_df['family_size'].map(family_group)
train_df['is_alone'] = [1 if i<2 else 0 for i in train_df.family_size]

test_df['is_alone'] = [1 if i<2 else 0 for i in test_df.family_size]
train_df['child'] = [1 if i<16 else 0 for i in train_df.Age]

test_df['child'] = [1 if i<16 else 0 for i in test_df.Age]

train_df.child.value_counts()
train_df['calculated_fare'] = train_df.Fare/train_df.family_size

test_df['calculated_fare'] = test_df.Fare/test_df.family_size
train_df.calculated_fare.mean()
def fare_group(fare):

    a= ''

    if fare <= 4:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a
train_df['fare_group'] = train_df['calculated_fare'].map(fare_group)

test_df['fare_group'] = test_df['calculated_fare'].map(fare_group)
train_df = pd.get_dummies(train_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

test_df = pd.get_dummies(test_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

train_df.drop(['Cabin', 'family_size','Ticket','Name', 'Fare'], axis=1, inplace=True)

test_df.drop(['Ticket','Name','family_size',"Fare",'Cabin'], axis=1, inplace=True)
pd.options.display.max_columns = 99
def age_group_fun(age):

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a
train_df['age_group'] = train_df['Age'].map(age_group_fun)

test_df['age_group'] = test_df['Age'].map(age_group_fun)
train_df = pd.get_dummies(train_df,columns=['age_group'], drop_first=True)

test_df = pd.get_dummies(test_df,columns=['age_group'], drop_first=True)

#Lets try all after dropping few of the column.

train_df.drop(['Age','calculated_fare'],axis=1,inplace=True)

test_df.drop(['Age','calculated_fare'],axis=1,inplace=True)
train_df.head()



train_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)

test_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)
X = train_df.drop('Survived', 1)

y = train_df['Survived']
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
# Classifier comparision

classifiers = [

    KNeighborsClassifier(3),

    svm.SVC(probability=True),

    DecisionTreeClassifier(),

    XGBClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

    





log_cols = ["Classifier", "Accuracy"]

log= pd.DataFrame(columns=log_cols)
SSplit=StratifiedShuffleSplit(test_size=0.3,random_state=7)

acc_dict = {}



for train_index,test_index in SSplit.split(X,y):

    X_train,X_test=X.iloc[train_index],X.iloc[test_index]

    y_train,y_test=y.iloc[train_index],y.iloc[test_index]

    

    for clf in classifiers:

        name = clf.__class__.__name__

          

        clf.fit(X_train,y_train)

        predict=clf.predict(X_test)

        acc=accuracy_score(y_test,predict)

        if name in acc_dict:

            acc_dict[name]+=acc

        else:

            acc_dict[name]=acc
log['Classifier']=acc_dict.keys()

log['Accuracy']=acc_dict.values()

#log.set_index([[0,1,2,3,4,5,6,7,8,9]])

%matplotlib inline

sns.set_color_codes("muted")

ax=plt.subplots(figsize=(10,8))

ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')

ax.set_xlabel('Accuracy',fontsize=20)

plt.ylabel('Classifier',fontsize=20)

plt.grid(color='r', linestyle='-', linewidth=0.5)

plt.title('Classifier Accuracy',fontsize=20)
## Necessary modules for creating models. 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

testframe = std_scaler.fit_transform(test_df)

testframe.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)



xgb=XGBClassifier(max_depth=2, n_estimators=700, learning_rate=0.009,nthread=-1,subsample=1,colsample_bytree=0.8)

xgb.fit(X_train,y_train)

predict=xgb.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))



# lda=LinearDiscriminantAnalysis()

# lda.fit(X_train,y_train)

# predict=lda.predict(X_test)

# print(accuracy_score(y_test,predict))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)



logreg = LogisticRegression(solver='liblinear', penalty='l1')

logreg.fit(X_train,y_train)

predict=logreg.predict(X_test)

# print(accuracy_score(y_test,predict))

# print(confusion_matrix(y_test,predict))



param = {'penalty': ['l1','l2'], 'C': [0.0001, 0.001, 0.01, 0.1,.15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2., 10.0, 100.0, 1000.0] }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=10,shuffle=True), n_jobs=1,scoring='accuracy')



grid.fit(X_train,y_train)



print (grid.best_params_)

print (grid.best_score_)

print(grid.best_estimator_)



grid.best_estimator_.fit(X_train,y_train)

predict=grid.best_estimator_.predict(X_test)

print(accuracy_score(y_test,predict))
# from sklearn.ensemble import VotingClassifier



# voting_classifier = VotingClassifier(estimators=[('logreg',logreg),

#                                                  ('XGB Classifier', xgb)])

# voting_classifier.fit(X_train,y_train)

# y_pred = voting_classifier.predict(X_test)

# voting_accuracy = accuracy_score(y_pred, y_test)

# print(voting_accuracy)
# y_predict=xgb.predict(testframe)

y_predict=grid.best_estimator_.predict(testframe)
y_predict.shape
y_predict
# from google.colab import files



temp = pd.DataFrame(pd.DataFrame({

        "PassengerId": passenger_id,

        "Survived": y_predict

    }))





temp.to_csv("submission.csv", index = False)

# files.download('submission.csv') 