import pandas as pd

import numpy as np



df_train = pd.read_csv(r'../input/titanic/train.csv')

df_train.head()

df_test = pd.read_csv(r'../input/titanic/test.csv')

df_test.head()
df_train.shape
from collections import Counter

import numpy as np



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   
Outliers_to_drop = detect_outliers(df_train,2,["Age","SibSp","Parch","Fare"])
df_train.loc[Outliers_to_drop] # Show the outliers rows
df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df_train.shape
len_train=len(df_train)

len_train
df_train=pd.concat([df_train,df_test],sort=False)
df_train.shape
df_train.isnull().sum()
df_train[["Age"]].describe()
#df_train.dropna(subset=["Age"],inplace=True)

df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())
df_train.isnull().sum()
df_train["Fare"] = df_train["Fare"].fillna(df_train["Fare"].mean())
df_train.isnull().sum()
df_train['Cabin'] = df_train['Cabin'].astype(str).str[0]
df_train['Cabin'].value_counts()
df_train.isnull().sum()
df_train["Embarked"].mode()
df_train["Embarked"] = df_train["Embarked"].fillna('S')
df_train.isnull().sum()
df_train.info()
import seaborn as sns

import matplotlib.pyplot as plt



sns.heatmap(df_train[["Survived","Age","SibSp","Parch","Fare"]].corr(),annot=True,cmap="BrBG")
sns.catplot(x="Survived", y="Age", kind="box", data=df_train)
sns.catplot(y="Survived", x="SibSp",kind="box", data=df_train)
sns.catplot(y="Survived", x="Parch", kind="box", data=df_train);
sns.catplot(x="Survived", y="Fare", kind="bar", data=df_train);
df_train.info()
sns.catplot(x="Sex", y="Survived", kind="bar", hue="Pclass",data=df_train)
sns.catplot(x="Cabin", y="Survived", kind="bar",data=df_train)
sns.catplot(x="Embarked", y="Survived", kind="bar",data=df_train)
df_train.head()
df_train = pd.concat([df_train, pd.get_dummies(df_train['Sex'])], axis=1)

df_train=df_train.drop(['Sex'],axis=1)

df_train.head()
df_train["F_size"]=df_train["SibSp"]+df_train["Parch"]+1

df_train=df_train.drop(['SibSp','Parch'],axis=1)

df_train.head()
sns.catplot(x="F_size", y="Survived", kind="bar",data=df_train)
df_train['Single'] = df_train['F_size'].map(lambda s: 1 if s == 1 else 0)

df_train['Small_F'] = df_train['F_size'].map(lambda s: 1 if s == 2 else 0)

df_train['Medium_F'] = df_train['F_size'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_train['Large_F'] = df_train['F_size'].map(lambda s: 1 if s >= 5 else 0)

df_train.head()
df_train = pd.concat([df_train, pd.get_dummies(df_train['Pclass'], prefix="Pclass")], axis=1)

df_train = pd.concat([df_train, pd.get_dummies(df_train['Cabin'], prefix="Cabin")], axis=1)

df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix="Embarked")], axis=1)

df_train=df_train.drop(['Pclass','Cabin','Embarked','F_size'],axis=1)

df_train.head()
features = df_train.drop(['PassengerId','Name','Ticket'],axis=1)

features.head()
X_train = features[features['Survived'].notnull()]

X_train.shape
X_test = features[features['Survived'].isnull()]

X_test.shape
X_train.isnull().sum()
Y_train = X_train['Survived']

X_train = X_train.drop(['Survived'],axis=1)
X_test=X_test.drop(['Survived'],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
kfold = StratifiedKFold(n_splits=10)
random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","KNeighboors","LogisticRegression"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res
LR = LogisticRegression()

LR.fit(X_train,Y_train)
Predict=LR.predict(X_test)
Y_test = pd.read_csv(r'../input/titanic/gender_submission.csv')

Y_test.head()
#Y_test=pd.merge(df_test, Y_test, on='PassengerId')

#Y_test=Y_test[['Age','Survived']]

#Y_test=Y_test.dropna()

PassengerId=Y_test['PassengerId']

Y_test=Y_test['Survived']

Y_test.head()
from sklearn.metrics import classification_report



print(classification_report(Y_test, Predict))
from sklearn.metrics import accuracy_score



accuracy_score(Y_test, Predict)
Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': Predict })
Submission.to_csv("submission.csv", index=False)