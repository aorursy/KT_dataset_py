# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col=0)

df.head()
print('Dataset shape: ', df.shape)

print('Survived: {}, Not Survived: {}'.format(df[df['Survived']==1]['Survived'].count(), df[df['Survived']==0]['Survived'].count()))
print(df.dtypes)

df.isnull().sum()
print('Unique values for SibSp column: ', df['SibSp'].unique())

print('Unique values for Parch column: ', df['Parch'].unique())

print('Unique values for Embarked column: ', df['Embarked'].unique())
print(df['Age'].describe())

df['Age'].hist()
df[df['Embarked'].isnull()==True]
df['Pclass'].corr(df['Fare'])
df['Embarked'] = df['Embarked'].fillna('N') # Embarked column = substitute with a new 'N' category
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)

df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1).drop(['Embarked', 'Embarked_N'],axis=1)

df.head()
# creating a new feature from Parch and SibSp

# if there are some values in those features other than 0, then passenger is not alone, otherwise 0

def isAlone(df):

    if df['SibSp']==0 and df['Parch']==0:

        return 1

    else: return 0



# constructing the new column

df['isAlone'] = df.apply(isAlone, axis=1)

df.drop(['SibSp', 'Parch', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True) # dropping all the other columns

df.head()
print(df.corrwith(df['Survived']))
#in this cell, you have to fill-in the NaN Age column values...

#df[df['Age'].notnull()].head()



meanAge = df['Age'].mean()

df.fillna(meanAge, inplace=True)

df.isnull().sum()
titanic_y = df['Survived']

titanic_X = df.drop(['Survived'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score



# code here

# splitting the training and testing set: 80-20 separation

# with stratify property to resolve class imbalances of the target inside the splits

X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.2, random_state=42, stratify=titanic_y)



# initialising classifiers/models

# SVC = Support Vector Machine Classifier

# LR = Logistic Regression

# DT = Decision Tree Classifier

# RF = Random Forest Classifier

clf_svc = SVC(random_state=0, max_iter=1000)

clf_lr = LogisticRegression(solver='lbfgs', random_state=0, max_iter=1000)

clf_dt = DecisionTreeClassifier(random_state=0)

clf_rf = RandomForestClassifier(random_state=0)



# building the model pipelines for each model/classifier  including preprocessing, e.g., standardisation when needed

pipeline_svc = Pipeline([('std', StandardScaler()), ('clf_svc', clf_svc)])

pipeline_lr = Pipeline([('std', StandardScaler()), ('clf_lr', clf_lr)])

pipeline_dt = Pipeline([('clf_dt', clf_dt)])  # no rescaling or standardization is required for DT

pipeline_rf = Pipeline([('clf_rf', clf_rf)])  # no rescaling or standardization is required for RF



# setting up the parameter grids for each classifier/model

parameter_grid_svc = [{'clf_svc__kernel': ['rbf'], 'clf_svc__C': np.logspace(-4, 4, 9), 'clf_svc__gamma':np.logspace(-4, 0, 4)},

                  {'clf_svc__kernel':['linear'], 'clf_svc__C': np.logspace(-4, 4, 9)}]

parameter_grid_lr = [{'clf_lr__penalty': ['l1', 'l2']},

                  {'clf_lr__C': np.logspace(0, 4, 10)}]

parameter_grid_dt = [{'clf_dt__criterion': ['gini', 'entropy']},

                  {'clf_dt__max_depth': [4,6,8,12]}]

parameter_grid_rf = [{'clf_rf__n_estimators': [10, 50, 100, 250, 500, 1000]},

                  {'clf_rf__max_features': ['sqrt', 'log2']},

                  {'clf_rf__min_samples_leaf': [1, 3, 5]}]
import warnings

warnings.filterwarnings('ignore')



# code here

# setting up multiple GridSearchCV instances: one for each model/classifier

# used 5x2 nested cross validation

gridcvs = {}

innerCV = StratifiedKFold(n_splits=2, shuffle=True, random_state=2) # stratified inner (2) fold



for pgrid, est, alg in zip((parameter_grid_svc, parameter_grid_lr, parameter_grid_dt, parameter_grid_rf),

                            (pipeline_svc, pipeline_lr, pipeline_dt, pipeline_rf),

                            ('SVM', 'LogisticRegression', 'DecisionTree', 'RandomForest')):

#for pgrid, est, alg in zip((parameter_grid_lr, parameter_grid_dt, parameter_grid_rf),

#                            (pipeline_lr, pipeline_dt, pipeline_rf),

#                            ('LogisticRegression', 'DecisionTree', 'RandomForest')):

  gcv = GridSearchCV(estimator=est, param_grid=pgrid, scoring='accuracy', n_jobs=1, cv=innerCV, verbose=0, refit=True)

  gridcvs[alg] = gcv



# outer (5) cross validation folds: note the stratified nature again

outerCV = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

outerScores = {}

for alg, est in sorted(gridcvs.items()):

  nestedScore = cross_val_score(est, X=X_train, y=y_train, cv=outerCV, n_jobs=1)

  outerScores[alg] = nestedScore

  print(f'{alg}: outer accuracy {100*nestedScore.mean():.2f} +/- {100*nestedScore.std():.2f}')
from sklearn.model_selection import learning_curve



# plotting the learning curves to demonstrate bias and variance during the model selection/evaluation process

def plot_learning_curve(ax, i, estimator, clf, X, y, ylim=None, cv=None, train_sizes=None):

    ax[i].set_title(f'Learning Curves ({clf})')

    ax[i].set_ylim(*ylim)

    ax[i].set_xlabel("Training samples")

    ax[i].set_ylabel("Score")

    train_sizes, train_scores, validation_scores = learning_curve(

        estimator, X, y, cv=cv, train_sizes=train_sizes)

    # computes the training and validation scores

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    validation_scores_mean = np.mean(validation_scores, axis=1)

    validation_scores_std = np.std(validation_scores, axis=1)



    # filling the curve with +/- standard deviation area

    ax[i].fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1, color="black")

    # filling the curve with +/- standard deviation area

    ax[i].fill_between(train_sizes, validation_scores_mean - validation_scores_std,

                     validation_scores_mean + validation_scores_std, alpha=0.1, color="blue")

    ax[i].plot(train_sizes, train_scores_mean, 'o-', color="black",

             label="Training score")

    ax[i].plot(train_sizes, validation_scores_mean, 'o-', color="blue",

             label="Cross-validation score")



    ax[i].legend(loc="lower right")

    ax[i].grid(True)

    return



# subplots for the learning curves of the four algorithms tried

fig, ax = plt.subplots(1, 4, figsize=(20, 5))

train_sizes = np.linspace(.1, 1.0, 5)

ylim = (0.70, 1.03)

cv = 5



# calling the learning curve drawing function for four different algorithms

plot_learning_curve(ax, 0, pipeline_svc, "SVC", X_train, y_train, 

                    ylim=ylim, cv=cv, train_sizes=train_sizes)

plot_learning_curve(ax, 1, pipeline_lr, "Logistic Regression", X_train, y_train, 

                    ylim=ylim, cv=cv, train_sizes=train_sizes)

plot_learning_curve(ax, 2, pipeline_dt, "Decision Tree", X_train, y_train, 

                    ylim=ylim, cv=cv, train_sizes=train_sizes)

plot_learning_curve(ax, 3, pipeline_rf, "Random Forest", X_train, y_train, 

                    ylim=ylim, cv=cv, train_sizes=train_sizes)

plt.show()
# fit "best" algorithm on the full training set

bestModel = 'SVM'

model = gridcvs[bestModel]  # getting the model parameters or loading the best model

model.fit(X_train, y_train) # fitting it to the whole training set (including the validation set)

train_acc = accuracy_score(y_true=y_train, y_pred=model.predict(X_train)) # running for the training set

test_acc = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))  # running for unseen testing set: generalised performance



# evaluate performance and compare to cross-validation results

print(f'Accuracy (mean cross-vaidated score of the best_estimator): {100*model.best_score_:.2f}')

print(f'Best Parameters: {gridcvs[bestModel].best_params_}')



# showing both training and testing accuracy

print(f'Training Accuracy: {100*train_acc:.2f}')

print(f'Test Accuracy: {100*test_acc:.2f}')
df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)

df.head()
df['Age'] = df['Age'].fillna(meanAge) # Age column = substitute with mean value

df['Embarked'] = df['Embarked'].fillna('N') # Embarked column = substitute with a new 'N' category

df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)

df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1).drop(['Embarked'],axis=1)



# constructing the new column

df['isAlone'] = df.apply(isAlone, axis=1)

df.drop(['SibSp', 'Parch', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True) # dropping all the other columns

df.head()
df[df['Fare'].isnull()==True]
# filling the fare with pclass mean value and from where the passenger has embarked...

df['Fare'].fillna(df[(df['Embarked_S']==1) & (df['Pclass']==3)]['Fare'].mean(), inplace=True)
df.isnull().sum()
y_pred = model.predict(df)

y_pred
df['Survived'] = y_pred # appending to the new 'Survived' column: the results

df = df.filter(['Survived']) # only the Passengerid and Survived columns...

df.to_csv('output.csv')