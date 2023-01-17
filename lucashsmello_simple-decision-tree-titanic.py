import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import pandas_profiling # library for automatic EDA



def pre_processData(D, categorical_cols, means, modes):

    nulled_cols = ['Age', 'Embarked']

    D.drop(['Name','PassengerId','Ticket', 'Cabin'],axis=1,  inplace=True)

    for col,mean in means.items():

        D[col].fillna(mean, inplace=True)

    for col,mode in modes.items():

        D[col].fillna(mode, inplace=True)

    

    for col in categorical_cols:

        if(len(D[col].unique())==2):

            D[col]=pd.factorize(D[col])[0]

        else:

            D=D.join(pd.get_dummies(D[col], prefix=col))

            D.drop(col,axis=1, inplace=True)

    

    return D



data_dir = '/kaggle/input/titanic'

D = pd.read_csv(data_dir+'/train.csv')

categorical_cols = ['Sex', 'Embarked']

numerical_cols = ['Age','Fare']

modes = {col:D[col].mode()[0] for col in categorical_cols}

means = {col:D[col].mean() for col in numerical_cols}

D = pre_processData(D,categorical_cols, means, modes)

D.info()

# report = pandas_profiling.ProfileReport(D)

# display(report)
sns.pairplot(D[['Pclass','Sex','SibSp', 'Parch','Fare','Survived']], hue='Survived', diag_kind='hist');
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_validate



metrics = ('accuracy','f1_macro')



sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

dtree = DecisionTreeClassifier(min_impurity_decrease=0.005)



X,Y = D.drop(['Survived'], axis=1), D['Survived']

scores = cross_validate(dtree,X,Y, scoring=metrics, cv=sampler, return_estimator=True)

for m in metrics:

    print("%s %.2f%%" % (m,100*scores['test_'+m]))

dtree = scores['estimator'][0]
from sklearn.tree import plot_tree, export_graphviz

import matplotlib.pyplot as plt





plt.rcParams['figure.dpi'] = 128

plt.rcParams['figure.figsize']=(9,6)



plot_tree(dtree, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True, proportion=True);
dtree = DecisionTreeClassifier(min_impurity_decrease=0.005).fit(X,Y)

plot_tree(dtree, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True, proportion=True);
Dtest_orig = pd.read_csv(data_dir+'/test.csv')

Dtest = pre_processData(Dtest_orig.copy(), categorical_cols, means, modes)

Dtest.info()
preds = dtree.predict(Dtest)

Dtest_orig['Survived'] = preds

Dtest_orig.to_csv('submission.csv', index=False, columns=['PassengerId','Survived'])