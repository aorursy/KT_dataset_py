import numpy as np 

import pandas as pd 

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os

# print(os.listdir("../input"))



df = pd.read_csv('../input/train.csv')



for row_label, row in df.iterrows():

    df.loc[row_label, 'Sex'] = 1 if row['Sex'] == 'male' else 0

for row_label, row in df.iterrows():

    try:

        df.loc[row_label, 'has_cabin'] = 0 if math.isnan(row['Cabin']) else 1

    except TypeError:

        df.loc[row_label, 'has_cabin'] = 1



df = df.drop(['Cabin','PassengerId', 'Ticket', 'Name'], axis=1)

df = pd.get_dummies(df)
df['Age'].hist(alpha=0.5, color='green', edgecolor='black', linewidth=2, bins=20)
# Create a 2 by 3 matrix of zeroes

median_ages = np.zeros((2,3))



# For each cell in the 2 by 3 matrix

for i in range(0,2):

    for j in range(0,3):

        # Set the value of the cell to be the median of all `Age` values

        # matching the criterion 'Corresponding sex and Pclass', leaving out all NaN values

        median_ages[i,j] = df[ (df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()



# Create new column AgeFill to put values into. 

# This retains the state of the original data.

df['AgeFill'] = df['Age']



# creates boolean age_filled column to mark passengers who didnt fill in age

for row_label, row in df.iterrows():

    df.loc[row_label, 'age_filled'] = 1 if row['Age'] != row['AgeFill'] else 0



df[ df['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']].head(10)



# Put our estimates into NaN rows of new column AgeFill.

# df.loc is a purely label-location based indexer for selection by label.



for i in range(0, 2):

    for j in range(0, 3):



        # Locate all cells in dataframe where `Sex` == i, `Pclass` == j+1

        # and `Age` == null. 

        # Replace them with the corresponding estimate from the matrix.

        df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]



df[['Age','AgeFill']].hist(alpha=0.5, color='green', edgecolor='black', linewidth=2, figsize=(20,5), bins=20)
df['Age'] = df['AgeFill']

df = df.drop('AgeFill', axis=1)
df.hist(figsize=(15,8), layout=(3, 5), alpha=0.4, edgecolor='black', linewidth=2)
import seaborn as sns



corr = df.corr()



sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
from scipy.stats import pearsonr



correlation_dict = {

  'property': [],

  'correlation': [],

  'pscore': []

}



for column in df.drop('Survived', axis=1).columns:

    

    # Two arguments here are the label (y) and feature (x) columns

    value = pearsonr(df['Survived'], df[column])

    print(column, value[0], '%.08f' % value[1])

    

    # This table is only including values that correlate with p < 0.05

    

    if value[1] < 0.05:

        correlation_dict['property'].append(column)

        correlation_dict['correlation'].append(value[0])

        correlation_dict['pscore'].append(value[1])



pd.DataFrame.from_dict(correlation_dict).sort_values('correlation')
features = ['Sex','Pclass','Embarked_S','age_filled','Parch','Embarked_C','Fare','has_cabin']
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, train_test_split



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



X = df.drop('Survived', axis=1).values

y = df['Survived']

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, 

                                                    random_state=SEED)



from sklearn.neighbors import KNeighborsClassifier



steps = [('scaler', StandardScaler()),

         ('knn', KNeighborsClassifier())]



pipeline = Pipeline(steps)



parameters = {'knn__n_neighbors': np.arange(1, 20),

              'knn__weights': ['distance', 'uniform']}



# instantiate 10-fold cv grid search object

grid = GridSearchCV(pipeline,

                    param_grid=parameters,

                    cv=10,

                    n_jobs=-1)



grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)



# Extract best hyperparameters from grid_dt

knn_best_params = grid.best_params_

print(knn_best_params)





from sklearn.metrics import confusion_matrix, classification_report

# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



from sklearn.linear_model import LogisticRegression



SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, 

                                                    random_state=SEED)



steps = [('scaler', StandardScaler()),

        ('logreg', LogisticRegression())]

pipeline = Pipeline(steps)



params = {'logreg__penalty': ['l1', 'l2'],

         'logreg__solver': ['liblinear']}



grid = GridSearchCV(pipeline, 

                   param_grid=params,

                   cv=10)



grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)



# Extract best hyperparameters from grid_dt

logreg_best_params = grid.best_params_

print(logreg_best_params)





from sklearn.metrics import confusion_matrix, classification_report

# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



# create roc curve

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

# predict_proba returns 2 column df, probabilities of each datapoint being 0/1

y_pred_prob = grid.predict_proba(X_test)[:,1]



### roc_curve takes 2 args, the test label results, probability array above

# roc_curve gives 3 outputs: 

# fpr = false positive rate, tpr = true positive rate, and array of thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('True Positive Rate')

plt.ylabel('False Positive Rate')

plt.title('ROC Curve')

plt.show()
from sklearn.ensemble import RandomForestClassifier



SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, 

                                                    stratify=y,

                                                    random_state=SEED)



steps = [('scaler', StandardScaler()),

        ('rf', RandomForestClassifier())]

pipeline = Pipeline(steps)



params = {'rf__n_estimators': range(100, 500, 25),

         'rf__random_state': [SEED]}



grid = GridSearchCV(pipeline, 

                   param_grid=params,

                   cv=10)



grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)



# Extract best hyperparameters from grid_dt

rf_best_params = grid.best_params_

print(rf_best_params)



from sklearn.metrics import confusion_matrix, classification_report

# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score



print('KNN Best Params: {} \nLogisticRegression Best Params: {} \nRandom Forest Best Params: {} \n'.format(knn_best_params,

                                                                                                           logreg_best_params,

                                                                                                           rf_best_params))



knn = KNeighborsClassifier(n_neighbors=knn_best_params['knn__n_neighbors'],

                          weights=knn_best_params['knn__weights'])

logreg = LogisticRegression(penalty=logreg_best_params['logreg__penalty'],

                           solver=logreg_best_params['logreg__solver'])

rf = RandomForestClassifier(n_estimators=rf_best_params['rf__n_estimators'],

                           random_state=SEED)



vc = VotingClassifier(estimators=[('knn', knn), ('logreg', logreg), ('rf', rf)],

                     voting='hard')



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, 

                                                    random_state=SEED)



vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)



# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
df = pd.read_csv('../input/test.csv')



for row_label, row in df.iterrows():

    df.loc[row_label, 'Sex'] = 1 if row['Sex'] == 'male' else 0

for row_label, row in df.iterrows():

    try:

        df.loc[row_label, 'has_cabin'] = 0 if math.isnan(row['Cabin']) else 1

    except TypeError:

        df.loc[row_label, 'has_cabin'] = 1



df = df.drop(['Cabin', 'Ticket', 'Name'], axis=1)

df = pd.get_dummies(df)



# Create a 2 by 3 matrix of zeroes

median_ages = np.zeros((2,3))



# For each cell in the 2 by 3 matrix

for i in range(0,2):

    for j in range(0,3):

        # Set the value of the cell to be the median of all `Age` values

        # matching the criterion 'Corresponding sex and Pclass', leaving out all NaN values

        median_ages[i,j] = df[ (df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()



# Create new column AgeFill to put values into. 

# This retains the state of the original data.

df['AgeFill'] = df['Age']



# creates boolean age_filled column to mark passengers who didnt fill in age

for row_label, row in df.iterrows():

    df.loc[row_label, 'age_filled'] = 1 if row['Age'] != row['AgeFill'] else 0



df[ df['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']].head(10)



# Put our estimates into NaN rows of new column AgeFill.

# df.loc is a purely label-location based indexer for selection by label.



for i in range(0, 2):

    for j in range(0, 3):



        # Locate all cells in dataframe where `Sex` == i, `Pclass` == j+1

        # and `Age` == null. 

        # Replace them with the corresponding estimate from the matrix.

        df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]



df['Age'] = df['AgeFill']

df = df.drop('AgeFill', axis=1)

df = df.fillna(0)



ids = df['PassengerId']

df = df.drop('PassengerId', axis=1)



X_test_final = df.values

SEED = 1



knn = KNeighborsClassifier(n_neighbors=knn_best_params['knn__n_neighbors'],

                          weights=knn_best_params['knn__weights'])

logreg = LogisticRegression(penalty=logreg_best_params['logreg__penalty'],

                           solver=logreg_best_params['logreg__solver'])

rf = RandomForestClassifier(n_estimators=rf_best_params['rf__n_estimators'],

                           random_state=SEED)



vc = VotingClassifier(estimators=[('knn', knn), ('logreg', logreg), ('rf', rf)],

                     voting='hard')



steps = [('scaler', StandardScaler()),

        ('vc', vc)]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)



y_pred = pipeline.predict(X_test_final)



predictions = pd.DataFrame({'PassengerId': ids, 'Survived':y_pred})

predictions.to_csv('predictions.csv', index=False)