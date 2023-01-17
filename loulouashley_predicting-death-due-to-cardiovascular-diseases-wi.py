import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.svm import SVC, LinearSVC

from sklearn import svm
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()

df['anaemia'] = df['anaemia'].astype(float)

df['creatinine_phosphokinase'] = df['creatinine_phosphokinase'].astype(float)

df['diabetes'] = df['diabetes'].astype(float)

df['ejection_fraction'] = df['ejection_fraction'].astype(float)

df['high_blood_pressure'] = df['high_blood_pressure'].astype(float)

df['platelets'] = df['platelets'].astype(float)

df['serum_creatinine'] = df['serum_creatinine'].astype(float)

df['serum_sodium'] = df['serum_sodium'].astype(float)

df['sex'] = df['sex'].astype(float)

df['smoking'] = df['smoking'].astype(float)

df['time'] = df['time'].astype(float)
df.info()

df.isnull().sum()
df.describe()
#preprocess data

bins =(-1,0.5,2)

groups_names=['survived','dead']

df['DEATH_EVENT'] = pd.cut(df['DEATH_EVENT'], bins=bins, labels = groups_names)

df['DEATH_EVENT'].unique()
label_quality=LabelEncoder()

df['DEATH_EVENT'] = label_quality.fit_transform(df['DEATH_EVENT'])

df.head(10)
df['DEATH_EVENT'].value_counts()
sns.countplot(df['DEATH_EVENT'])
#Now we can separate the dataset as response variable and feature variable

X = df.drop('DEATH_EVENT', axis = 1)

y = df['DEATH_EVENT']


corr = df.corr() #Correlation matrix for CB player

corr
fig = plt.figure(figsize=(8,8))

plt.matshow(corr, cmap='RdBu', fignum=fig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns);
ax = df[['platelets', 'DEATH_EVENT']].boxplot(by='DEATH_EVENT', figsize=(10,6))

ax.set_ylabel('platelets')
ax = df[['ejection_fraction', 'DEATH_EVENT']].boxplot(by='DEATH_EVENT', figsize=(10,6))

ax.set_ylabel('ejection_fraction')
df['age'].plot(kind='density', figsize=(14,6))
all_inputs = df[['age','anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',

'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',

'sex', 'smoking', 'time']].values

# extracting quality labels

all_labels = df['DEATH_EVENT'].values

# a test to see what the inputs look like

all_inputs[:2]
#Train and split the data

X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_labels, test_size= 0.2, random_state=42)
#Test of firsts values

X_train[:1]
#trying decision tree classfier 



from sklearn.tree import DecisionTreeClassifier



# Create the classifier

decision_tree_classifier = DecisionTreeClassifier()





# Train the classifier on the training set

decision_tree_classifier.fit(X_train, y_train)





# Validate the classifier on the testing set using classification accuracy

decision_tree_classifier.score(X_test, y_test)
rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
pred_rfc[:20]
X_test[:2]
# Let's see  the RFC efficients 

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test,pred_rfc))


# Validate the classifier on the testing set using classification accuracy

rfc.score(X_test, y_test)
Clf = svm.SVC()

Clf.fit(X_train,y_train)

pref_clf = Clf.predict(X_test)
# Let's see  the SVM efficients 

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test,pred_rfc))


# Validate the classifier on the testing set using classification accuracy

Clf.score(X_test, y_test)
#selecting the models and the model names in an array

models=[LogisticRegression(),

        LinearSVC(),

        SVC(kernel='rbf'),

        KNeighborsClassifier(),

        RandomForestClassifier(),

        DecisionTreeClassifier(),

        GradientBoostingClassifier(),

        GaussianNB()]

model_names=['Logistic Regression',

             'Linear SVM',

             'rbf SVM',

             'K-Nearest Neighbors',

             'Random Forest Classifier',

             'Decision Tree',

             'Gradient Boosting Classifier',

             'Gaussian NB']





# creating an accuracy array and a matrix to join the accuracy of the models

# and the name of the models so we can read the results easier

acc=[]

m={}





# next we're going to iterate through the models, and get the accuracy for each

for model in range(len(models)):

     clf=models[model]

     clf.fit(X_train,y_train)

     pred=clf.predict(X_test)

     acc.append(accuracy_score(pred,y_test))





m={'Algorithm':model_names,'Accuracy':acc}





# just putting the matrix into a data frame and listing out the results

acc_frame=pd.DataFrame(m)

acc_frame
random_forest_classifier = RandomForestClassifier()





# setting up the parameters for our grid search

# You can check out what each of these parameters mean on the Scikit webiste!

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

parameter_grid = {'n_estimators': [10, 25, 50, 100, 200],

'max_features': ['auto', 'sqrt', 'log2'],

'criterion': ['gini', 'entropy'],

'max_features': [1, 2, 3, 4]}





# Stratified K-Folds cross-validator allows us mix up the given test/train data per run

# with k-folds each test set should not overlap across all shuffles. This allows us to 

# ultimately have "more" test data for our model

cross_validation = StratifiedKFold(n_splits=10)





# running the grid search function with our random_forest_classifer, our parameter grid

# defineda bove, and our cross validation method

grid_search = GridSearchCV(random_forest_classifier,

param_grid=parameter_grid,

cv=cross_validation)





# using the defined grid search above, we're going to test it out on our

# data set

grid_search.fit(all_inputs, all_labels)





# printing the best scores, parameters, and estimator for our Random Forest classifer

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))





grid_search.best_estimator_
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',



          max_depth=None, max_features=2, max_leaf_nodes=None,



          min_impurity_decrease=0.0, min_impurity_split=None,



          min_samples_leaf=1, min_samples_split=2,



          min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,



          oob_score=False, random_state=None, verbose=0,



          warm_start=False)
random_forest_classifier = grid_search.best_estimator_





rf_df = pd.DataFrame({'accuracy': cross_val_score(random_forest_classifier, all_inputs, all_labels, cv=10),

                      'classifier': ['Random Forest'] * 10})

rf_df.mean()
sns.boxplot(x='classifier', y='accuracy', data=rf_df)

sns.stripplot(x='classifier', y='accuracy', data=rf_df, jitter=True, color='black')