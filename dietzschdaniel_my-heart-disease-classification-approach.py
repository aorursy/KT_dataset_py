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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df
df['target'].value_counts()
b = sns.countplot(x='target', data=df)

b.set_title("Target Distribution");
df['age'].describe()
b = sns.distplot(df['age'])

b.set_title("Age Distribution");
b = sns.boxplot(y = 'age', data = df)

b.set_title("Age Distribution");
b = sns.boxplot(y='age', x='target', data=df)

b.set_title("Age Distribution for Target")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease");
df['sex'].value_counts()
b = sns.countplot(x='sex', data=df)

b.set_title("Target Distribution");
pd.crosstab(df['target'], df['sex']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for Sex")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["female", "male"])

plt.xticks(rotation=0);
df['cp'].value_counts()
b = sns.countplot(x='cp', data=df)

b.set_title("cp Distribution");
pd.crosstab(df['target'], df['cp']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for cp")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1", "2", "3"])

plt.xticks(rotation=0);
df['trestbps'].describe()
b = sns.distplot(df['trestbps'])

b.set_title("trestbps Distribution");
b = sns.boxplot(y = 'trestbps', data = df)

b.set_title("trestbps Distribution");
b = sns.boxplot(y='trestbps', x='target', data=df)

b.set_title("trestbps Distribution for Target")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease");
df['chol'].describe()
b = sns.distplot(df['chol'])

b.set_title("trestbps Distribution");
b = sns.boxplot(y = 'chol', data = df)

b.set_title("trestbps Distribution");
b = sns.boxplot(y='chol', x='target', data=df)

b.set_title("chol Distribution for Target")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease");
df['fbs'].value_counts()
b = sns.countplot(x='fbs', data=df)

b.set_title("fbs Distribution");
pd.crosstab(df['target'], df['fbs']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for fbs")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1"])

plt.xticks(rotation=0);
df['restecg'].value_counts()
b = sns.countplot(x='restecg', data=df)

b.set_title("restecg Distribution");
pd.crosstab(df['target'], df['restecg']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for restecg")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1", "2"])

plt.xticks(rotation=0);
df['thalach'].describe()
b = sns.distplot(df['thalach'])

b.set_title("trestbps Distribution");
b = sns.boxplot(y = 'thalach', data = df)

b.set_title("thalach Distribution");
b = sns.boxplot(y='thalach', x='target', data=df)

b.set_title("thalach Distribution for Target")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease");
df['exang'].value_counts()
b = sns.countplot(x='exang', data=df)

b.set_title("exang Distribution");
pd.crosstab(df['target'], df['exang']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for exang")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1"])

plt.xticks(rotation=0);
df['oldpeak'].describe()
b = sns.distplot(df['oldpeak'])

b.set_title("oldpeak Distribution");
b = sns.boxplot(y = 'oldpeak', data = df)

b.set_title("oldpeak Distribution");
b = sns.boxplot(y='oldpeak', x='target', data=df)

b.set_title("oldpeak Distribution for Target")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease");
df['slope'].value_counts()
b = sns.countplot(x='slope', data=df)

b.set_title("slope Distribution");
pd.crosstab(df['target'], df['slope']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for slope")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1", "2"])

plt.xticks(rotation=0);
df['ca'].value_counts()
b = sns.countplot(x='ca', data=df)

b.set_title("ca Distribution");
pd.crosstab(df['target'], df['ca']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for ca")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1", "2", "3", "4"])

plt.xticks(rotation=0);
df['thal'].value_counts()
b = sns.countplot(x='thal', data=df)

b.set_title("thal Distribution");
pd.crosstab(df['target'], df['thal']).plot(kind="bar", figsize=(10,6))



plt.title("Target distribution for thal")

plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")

plt.ylabel("Count")

plt.legend(["0", "1", "2", "3"])

plt.xticks(rotation=0);
df.isna().sum()
# Everything except target variable

X = df.drop("target", axis=1)



# Target variable

y = df['target'].values
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
