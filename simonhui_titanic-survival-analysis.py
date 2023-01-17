import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.feature_selection import SelectKBest, f_classif



pd_data = pd.read_csv('/kaggle/input/titanic/train.csv')
#g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")



# Explore Sex vs Survived

g = sns.barplot(x="Sex",y="Survived",data=pd_data)

g = g.set_ylabel("Survival Probability")
# Explore Parch vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=pd_data,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)
# Explore Age vs Survived

#g = sns.FacetGrid(pd_data, col='Survived')

#g = g.map(sns.distplot, "Age")

#g = g.set_ylabels("Survival probability")



sns.distplot(pd_data['Age'].dropna(), [0,20,40,100])
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=pd_data, size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
# Explore Fare distribution 

#g = sns.distplot(pd_data["Fare"], color="m", label="Skewness : %.2f"%(pd_data["Fare"].skew()))

#g = g.legend(loc="best")

sns.distplot(pd_data['Fare'].dropna().map(lambda i: np.log(i) if i > 0 else 0))
X = pd_data.loc[:, ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')]



labelencoder = LabelEncoder()

X['Sex'] = labelencoder.fit_transform(X['Sex'])



X['Embarked'] = np.where(X['Embarked'].isna(), 'S', X['Embarked'])

#pd.get_dummies(X)

one_hot = pd.get_dummies(X['Embarked'])

X = X.drop('Embarked', axis=1)

X = X.join(one_hot)



## Fill missing value of Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(X["Age"][X["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = X["Age"].median()

    age_pred = X["Age"][((X['SibSp'] == X.iloc[i]["SibSp"]) & (X['Parch'] == X.iloc[i]["Parch"]) & (X['Pclass'] == X.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        X['Age'].iloc[i] = age_pred

    else :

        X['Age'].iloc[i] = age_med



X['CategoricalAge'] = pd.cut(X['Age'], [0,20,40,100])

one_hot = pd.get_dummies(X['CategoricalAge'])

X = X.drop('Age', axis=1)

X = X.drop('CategoricalAge', axis=1)

X = X.join(one_hot)



#Fill Fare missing values with the median value

X["Fare"] = X["Fare"].fillna(X["Fare"].median())

# Apply log to Fare to reduce skewness distribution

X["Fare"] = X["Fare"].map(lambda i: np.log(i) if i > 0 else 0)



#scaler = StandardScaler()

#scaler.fit(X)

#StandardScaler(copy=True, with_mean=True, with_std=True)

#X = scaler.transform(X)



y = pd_data.loc[:, 'Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train_target = pd_data['Survived'].values

possible_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S']



# Check feature importances

selector = SelectKBest(f_classif, len(possible_features))

selector.fit(X, train_target)

scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]



print('Feature importances:')

for i in range(len(scores)):

    print('%.2f %s' % (scores[indices[i]], possible_features[indices[i]]))
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(solver='lbfgs', multi_class='auto')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# Neural Network

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(9,9,9),max_iter=500)

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
pd_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

X_t = pd_test_data.loc[:, ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')]



## Fill missing value of Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(X_t["Age"][X_t["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = X_t["Age"].median()

    age_pred = X_t["Age"][((X_t['SibSp'] == X_t.iloc[i]["SibSp"]) & (X_t['Parch'] == X_t.iloc[i]["Parch"]) & (X_t['Pclass'] == X_t.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        X_t['Age'].iloc[i] = age_pred

    else :

        X_t['Age'].iloc[i] = age_med



X_t['CategoricalAge'] = pd.cut(X_t['Age'], [0,20,40,100])

one_hot = pd.get_dummies(X_t['CategoricalAge'])

X_t = X_t.drop('Age', axis=1)

X_t = X_t.drop('CategoricalAge', axis=1)

X_t = X_t.join(one_hot)



#Fill Fare missing values with the median value

X_t["Fare"] = X_t["Fare"].fillna(X_t["Fare"].median())

# Apply log to Fare to reduce skewness distribution

X_t["Fare"] = X_t["Fare"].map(lambda i: np.log(i) if i > 0 else 0)



labelencoder = LabelEncoder()

X_t['Sex'] = labelencoder.fit_transform(X_t['Sex'])



X_t['Embarked'] = np.where(X_t['Embarked'].isna(), 'S', X_t['Embarked'])

one_hot = pd.get_dummies(X_t['Embarked'])

X_t = X_t.drop('Embarked', axis=1)

X_t = X_t.join(one_hot)



pred = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])



y_logreg_pred = logreg.predict(X_t)

pred['Survived'] = y_logreg_pred.astype(int)

pred.to_csv("../working/submission_logreg_5.csv", index = False)



y_mlp_pred = mlp.predict(X_t)

pred['Survived'] = y_mlp_pred.astype(int)

pred.to_csv("../working/submission_mlp_5.csv", index = False)
for dirname, _, filenames in os.walk('../working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.chdir(r'../working')

from IPython.display import FileLink

#FileLink(r'submission_logreg_5.csv')

FileLink(r'submission_mlp_5.csv')