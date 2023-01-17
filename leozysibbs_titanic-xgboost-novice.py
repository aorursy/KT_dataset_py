

# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

#Data Understanding (Dataset what its all about), Missing Values, Imbalance, Correlation of features to Target, 

#Preprocessing

#Modelling

#Evaluation

#loading data

test_df = pd.read_csv('../input/titanic/test.csv')

train_df = pd.read_csv('../input/titanic/train.csv')
train_df.head()
train_df.info()
train_df.isnull().sum()
train_df.describe()
test_df.info()

test_df.isnull().sum()
plt.figure(figsize=(8, 8))

sns.countplot("Survived", data=train_df)

plt.title('Target Balance')

plt.show()
train_df.shape
print('Evaluating the training dataset target :')

train_df["Survived"].value_counts()

sum=((train_df["Survived"]==1)/len(train_df)).sum()*100

sum

#sum.sum()*100
#Age

feature_median = round(pd.to_numeric(train_df["Age"]).median())

fmedian=round(pd.to_numeric(test_df["Age"]).median())

train_df["Age"]=train_df["Age"].fillna(feature_median)

test_df["Age"]=test_df["Age"].fillna(fmedian)

#train_df.isnull().sum()

test_df.isnull().sum()
#Embarked

train_df["Embarked"]=train_df["Embarked"].fillna("S")

train_df.isnull().sum()
sns.barplot(x='Sex', y='Survived', data=train_df)
sns.barplot(x='Pclass', y='Survived', data=train_df)
sns.barplot(x='Embarked', y='Survived', data=train_df)
g = sns.FacetGrid(train_df, col='Survived')

g = g.map(sns.distplot, "Age")
sns.barplot(x='Parch', y='Survived', data=train_df)
sns.barplot(x='SibSp', y='Survived', data=train_df)
sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train_df)
train_df["Embarked"].value_counts()
sns.boxplot(y=train_df['Fare'])
g = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train_df);
sns.boxplot(y=train_df['Age'])
g = sns.FacetGrid(train_df, col='Survived')

g = g.map(sns.distplot, "Fare")
#test_df=test_df=test_df.drop(["Name", "PassengerId", "Cabin"], axis=1)

 # Create a function that converts all values of df['score'] into numbers

from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):

    columnsToEncode = list(df.select_dtypes(include=['category','object']))

    le = LabelEncoder()

    for feature in columnsToEncode:

        try:

            df[feature] = le.fit_transform(df[feature])

        except:

            print('Error encoding '+feature)

    return df

df = dummyEncode(train_df)

dy=dummyEncode(test_df)
train_df.info()

#train_df=train_df.drop(["Cabin"], axis=1)
test_df.info()

sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train_df)
train_df["Embarked"].value_counts()
import xgboost

print(xgboost.__version__)
train_df= train_df.drop(["Fare", "Ticket", "Name", "Cabin", 'PassengerId'], axis=1)

test_df= test_df.drop(["Fare", "Ticket", "Name", "Cabin"], axis=1)
train_df.info()
test_df.info()
# Algorithm

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score
X=train_df.drop(["Survived"], axis=1)

y=train_df["Survived"]

X_test=test_df.drop(['PassengerId'], axis=1)
scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

scaler.fit(X_test)

X_test=scaler.transform(X_test)
#train test split Spliting our dataset training :80 Testing :20



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# define the model

model = XGBClassifier()

# fit the model

model.fit(X, y)

y_pred = model.predict(x_test)

print(accuracy_score(y_pred, y_test)*100)

score_xgb = cross_val_score(model, X, y, cv=5).mean()

print(score_xgb)
#Classification report

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

plt.show()




# define function to calculate and print model metrics.

def AlgMetrics(y_test,y_pred):

    cp = confusion_matrix(y_test,y_pred)

    sensitivity = cp[1,1]/(cp[1,0]+cp[1,1])

    specificity =  cp[0,0]/(cp[0,1]+cp[0,0])

    precision = cp[1,1]/(cp[0,1]+cp[1,1])

    print('Confusion Matrix: \n',cp)

    print("Sensitivity: ", sensitivity)

    print("Specificity: ",specificity)

    print("AUC Score: ", roc_auc_score(y_test,y_pred)) 

    print("Precision: ",precision)

    print("f1 Score: ",f1_score(y_test,y_pred))

    

AlgMetrics(y_test,y_pred)
#set ids as PassengerId and predict survival 

Pid = test_df['PassengerId']

predictions = model.predict(X_test)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : Pid, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")

output