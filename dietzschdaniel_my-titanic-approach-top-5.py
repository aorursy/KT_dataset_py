# Data Processing

import numpy as np 

import pandas as pd 



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='whitegrid')



# Modeling

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import roc_auc_score



from sklearn.model_selection import RandomizedSearchCV
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train
df_test
b = sns.countplot(x='Survived', data=df_train)

b.set_title("Survived Distribution");
b = sns.countplot(x='Pclass', data=df_train)

b.set_title("Pclass Distribution");
pd.crosstab(df_train['Survived'], df_train['Pclass']).plot(kind="bar", figsize=(10,6))



plt.title("Survived distribution for Pclass")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Count")

plt.legend(["Pclass 1", "Pclass 2", "Pclass 3"])

plt.xticks(rotation=0);
b = sns.countplot(x='Sex', data=df_train)

b.set_title("Sex Distribution");
pd.crosstab(df_train['Survived'], df_train['Sex']).plot(kind="bar", figsize=(10,6))



plt.title("Survived distribution for Sex")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Count")

plt.legend(["male", "female"])

plt.xticks(rotation=0);
b = sns.distplot(df_train['Age'])

b.set_title("Age Distribution");
b = sns.boxplot(y = 'Age', data = df_train)

b.set_title("Age Distribution");
b = sns.boxplot(y='Age', x='Survived', data=df_train);

b.set_title("Age Distribution for Survived");
b = sns.countplot(x='SibSp', data=df_train)

b.set_title("SibSp Distribution");
pd.crosstab(df_train['Survived'], df_train['SibSp']).value_counts()
df_train['Parch'].value_counts()
b = sns.countplot(x='Parch', data=df_train)

b.set_title("Parch Distribution");
pd.crosstab(df_train['Survived'], df_train['Parch']).value_counts()
b = sns.distplot(df_train['Fare'])

b.set_title("Fare Distribution");
b = sns.boxplot(y = 'Fare', data = df_train)

b.set_title("Fare Distribution");
b = sns.boxplot(y='Fare', x='Survived', data=df_train);

b.set_title("Fare Distribution for Survived");
df_train['Embarked'].value_counts()
b = sns.countplot(x='Embarked', data=df_train)

b.set_title("Parch Distribution");
pd.crosstab(df_train['Survived'], df_train['Embarked']).plot(kind="bar", figsize=(10,6))



plt.title("Survived distribution for Embarked")

plt.xlabel("0 = Not Survived, 1 = Survived")

plt.ylabel("Count")

plt.legend(["C", "Q", "S"])

plt.xticks(rotation=0);
df_train.isna().sum()
df_test.isna().sum()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_train['Cabin'] = df_train['Cabin'].fillna("Missing")

df_test['Cabin'] = df_test['Cabin'].fillna("Missing")
df_train = df_train.dropna()
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
df_train.isna().sum()
df_test.isna().sum()
df_train.shape
df_test.shape
df_train.head()
df_test.head()
df_train = df_train.drop(columns=['Name'], axis=1)

df_test = df_test.drop(columns=['Name'], axis=1)
sex_mapping = {

    'male': 0,

    'female': 1

}



df_train.loc[:, "Sex"] = df_train['Sex'].map(sex_mapping)

df_test.loc[:, "Sex"] = df_test['Sex'].map(sex_mapping)
df_train = df_train.drop(columns=['Ticket'], axis=1)

df_test = df_test.drop(columns=['Ticket'], axis=1)
df_train = df_train.drop(columns=['Cabin'], axis=1)

df_test = df_test.drop(columns=['Cabin'], axis=1)
df_train.head()
df_test.head()
df_test['Embarked'].value_counts()
df_train = pd.get_dummies(df_train, prefix_sep="__",

                              columns=['Embarked'])

df_test = pd.get_dummies(df_test, prefix_sep="__",

                              columns=['Embarked'])
df_train.head()
df_test.head()
# Everything except target variable

X = df_train.drop("Survived", axis=1)



# Target variable

y = df_train['Survived'].values
# Random seed for reproducibility

np.random.seed(42)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 
# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(),

          "Logistic Regression": LogisticRegression(max_iter=10000), 

          "Random Forest": RandomForestClassifier(),

          "SVC" : SVC(probability=True),

          "DecisionTreeClassifier" : DecisionTreeClassifier(),

          "AdaBoostClassifier" : AdaBoostClassifier(),

          "GradientBoostingClassifier" : GradientBoostingClassifier(),

          "GaussianNB" : GaussianNB(),

          "LinearDiscriminantAnalysis" : LinearDiscriminantAnalysis(),

          "QuadraticDiscriminantAnalysis" : QuadraticDiscriminantAnalysis()}



# Create function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dict of different Scikit-Learn machine learning models

    X_train : training data

    X_test : testing data

    y_train : labels assosciated with training data

    y_test : labels assosciated with test data

    """

    # Random seed for reproducible results

    np.random.seed(42)

    # Make a list to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Predicting target values

        y_pred = model.predict(X_test)

        # Evaluate the model and append its score to model_scores

        #model_scores[name] = model.score(X_test, y_test)

        model_scores[name] = roc_auc_score(y_test, y_pred)

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)

model_scores
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)
y_pred = gbc.predict(df_test)
y_pred
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sub.head()
sub['Survived'] = y_pred

sub.to_csv("results_titanic.csv", index=False)

sub.head()