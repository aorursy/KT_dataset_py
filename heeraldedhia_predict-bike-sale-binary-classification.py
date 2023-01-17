import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization purposes

import seaborn as sns # for statistical data visualization

%matplotlib inline
df = pd.read_csv('../input/bike-buyers/bike_buyers_clean.csv', sep=',')
df.head()
df.shape
df.columns
df.info()
corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables'.format(len(numerical)))

print('The numerical variables are :', numerical)
df[numerical].head()
# check missing values in numerical variables

df[numerical].isnull().sum()
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))

print('The categorical variables are :', categorical)
df[categorical].head()
df[categorical].isnull().sum()
# view frequency counts of values in categorical variables

for var in categorical: 

    print(df[var].value_counts())

    print(df[var].value_counts()/np.float(len(df)))

    print()
# check for cardinality in categorical variables

for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 



df['Marital Status'] = label_encoder.fit_transform(df['Marital Status'])

df['Gender'] = label_encoder.fit_transform(df['Gender'])

df['Education'] = label_encoder.fit_transform(df['Education'])

df['Occupation'] = label_encoder.fit_transform(df['Occupation'])

df['Home Owner'] = label_encoder.fit_transform(df['Home Owner'])

df['Commute Distance'] = label_encoder.fit_transform(df['Commute Distance'])

df['Region'] = label_encoder.fit_transform(df['Region'])

df['Purchased Bike'] = label_encoder.fit_transform(df['Purchased Bike'])

df.head()
df['Age'].describe()
df['Age'] = pd.cut(x = df['Age'], bins = [0,30,40,50,60,100,150], labels = [0, 1, 2, 3, 4, 5])

df['Age'] = df['Age'].astype('int64') 

df['Age'].isnull().sum()
df['Income'].describe()
df['Income'] = pd.cut(x = df['Income'], bins = [0, 30000, 50000, 75000, 100000, 150000, 200000], labels = [1, 2, 3, 4, 5, 6])

df['Income'] = df['Income'].astype('int64') 

df['Income'].isnull().sum()
df.dtypes
X = df.drop(['Purchased Bike'], axis=1)

y = df['Purchased Bike']
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)

X_train.shape, X_test.shape
X_train.head()
X_train.shape
X_test.head()
X_test.shape
# train a Gaussian Naive Bayes classifier on the training set

from sklearn.naive_bayes import GaussianNB



# instantiate the model

gnb = GaussianNB()



# fit the model

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)



y_pred[:10]

len(y_pred)
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], 

                                 index=['Predict Positive', 'Predict Negative'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

plt.show()
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=999)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



# Predict the response for test dataset

y_pred2 = clf.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred2)))
y_pred_train2 = clf.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train2)))
from sklearn import tree



fn=['ID', 'Marital Status', 'Gender', 'Income', 'Children', 'Education',

       'Occupation', 'Home Owner', 'Cars', 'Commute Distance', 'Region', 'Age']

cn=['Bought', 'Not Bought']



fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(clf, feature_names = fn, 

               class_names=cn, filled=True)
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

grid_values = {'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2'], 

               'max_depth':[4,5,6,7,8,9,10], 'min_samples_split': [2,3,4]}

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)

grid_clf_acc.fit(X_train, y_train)
y_pred_acc = grid_clf_acc.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))



#Confusion matrix

cm = confusion_matrix(y_test,y_pred_acc)

print(cm)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], 

                                 index=['Predict Positive', 'Predict Negative'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

plt.show()
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred2))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train) 

result = model.score(X_test, y_test)



print('Model accuracy score: {0:0.4f}'. format(result))
from sklearn.metrics import make_scorer, accuracy_score



rfc = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Run the grid search

grid_obj = GridSearchCV(rfc, parameters)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

rfc = grid_obj.best_estimator_



# Fit the best algorithm to the data

rfc.fit(X_train, y_train)
y_pred4 = grid_clf_acc.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred4)))
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))