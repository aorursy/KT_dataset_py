import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="whitegrid", palette="pastel")

import re

pd.options.display.float_format = '{:,.2f}'.format



data = pd.read_csv('../input/train.csv')
sns.violinplot(x='Survived', y='Age', hue='Sex', data=data, split=True)

plt.gca().set_title("Age Distribution");
gender = data.pivot_table(values='PassengerId', 

                          index='Sex', 

                          columns=['Survived'], 

                          aggfunc=len)



gender.plot(kind='barh', stacked=True)

plt.gca().set_title("Gender Distribution");
def parse_title(s): 

    title = re.search(",\s(\w+).\s", s).group(1)

    repl = {'th': 'Mme', 'Jonkheer': 'Mr'}

    title = title if title not in repl.keys() else repl[title]

    return title



data['Title'] = data.Name.map(parse_title)

#data.loc[data.Title.isin(['th', 'Jonkheer']), 'Title'] = ('Mme', 'Mr')



title = (data.pivot_table(values="PassengerId", index="Title", columns=["Survived"], 

                         aggfunc=len).sort_values(0, ascending=False))



title.plot(kind='barh', stacked=True)

plt.gca().invert_yaxis()

plt.gca().set_title("Survival according to passenger's title");
data['Deck'] = data['Cabin'].map(lambda s: s[0], na_action='ignore')

g = sns.PairGrid(data, y_vars="Survived",

                 x_vars=["Deck", "Pclass", "Embarked"],

                 size=5, aspect=1.3)

plt.gcf().suptitle('Class and Location of Cabin impact on Survival')

g.map(sns.pointplot);
embarked = data.pivot_table(values="PassengerId", index="Embarked", columns=["Pclass"], aggfunc=len)

embarked.plot(kind="bar", stacked=True);
sns.violinplot(x="Survived", y="Fare", data=data)

plt.gca().set_title("Fare Distribution");
data['Family'] = data['SibSp'] + data['Parch']



fig = plt.figure(figsize=(12,18))

for i, feature in enumerate(['SibSp', 'Parch', 'Family']):

    ax1 = fig.add_subplot(3,2,2*i+1)

    ax2 = fig.add_subplot(3,2,2*i+2)

    sns.pointplot(x=feature, y='Survived', data=data, ax=ax1)

    sns.swarmplot(x=feature, y='Age', hue='Survived', data=data, ax=ax2)



plt.gcf().suptitle('Family on board of the Titanic');
selected_features = ['PassengerId', 'Survived', 'Pclass', 'Age', 'Sex', 'Fare', 'Title', 'Family'] 
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, make_scorer, classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv('../input/train.csv').set_index('PassengerId')

X = data.iloc[:, 1:]

y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0])

   



def feature_engineering(X_train, X_test):

    

    input_values = X_train[['Pclass', 'Fare']].groupby('Pclass', as_index=False).mean()

    

    output = []

    

    for i, df in enumerate([X_train, X_test]):

        df = df.copy()

        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        df['Family'] = df['SibSp'] + df['Parch']

               

        df.drop([col for col in df.columns if col not in selected_features], axis=1, inplace=True)

        df.update(df[['Age']].fillna(X_train["Age"].mean()))

        df.update(df[['Fare']].fillna(X_train["Fare"].mean()))

        output.append(df)

        

    return output

    

X_train, X_test = feature_engineering(X_train, X_test)
def train_model(X_train, y_train):



    clf = RandomForestClassifier(random_state=42)



    params = {'max_features': range(2,6), 

              'min_samples_leaf': (2,3,5,7,9,15), 

              'n_estimators': range(5,21,5)}



    gs = GridSearchCV(clf, params, scoring='roc_auc', n_jobs=-1)

    gs.fit(X_train, y_train)

    

    return gs.best_estimator_



clf = train_model(X_train, y_train)

clf
y_pred = clf.predict(X_test)

print("AUC: {:.3}".format(roc_auc_score(y_test, y_pred)), end='\t')

print("Accuracy: {:.3}".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))
train = pd.read_csv('../input/train.csv').set_index('PassengerId')

X_test = pd.read_csv('../input/test.csv').set_index('PassengerId')

X_train = train.iloc[:, 1:]; y_train = train.iloc[:, 0]



X_train, X_test = feature_engineering(X_train, X_test)

clf = train_model(X_train, y_train)

predictions = clf.predict(X_test)

submission = pd.DataFrame(predictions, index=X_test.index)

submission.to_csv("submission.csv");