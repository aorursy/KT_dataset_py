import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import metrics

from keras.layers import Dropout



from keras.layers import Dense

from keras.models import Sequential

plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.notebook_repr_html', True)
import os

print(os.listdir('../input/titanic/'))
# reading dataset

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

# concatenating both datasets

df = pd.concat([train, test])

df.head()
test.head()
# checking for shape

print(train.shape, test.shape, df.shape)
df.describe()
# checking dtypes

df.dtypes
df.Cabin.unique()
df.Embarked.unique()
df.isnull().sum()
sns.countplot(x='Sex',hue ='Survived', data=train);
sns.countplot(x='Pclass',hue ='Survived', data=train);
sns.countplot(x='Embarked',hue ='Survived', data=train);
# Replacing female as 0 and Male as 1

df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
# Filling all NaN with S

df['Embarked'] = df['Embarked'].fillna('S')

# Replacing S with 0 and C with 1 and Q with 2

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Creating new feature by adding SibSp with Parch

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Creating new feature IsAlone if Family Size is 1

df['IsAlone'] = 0

df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
sns.countplot(x='IsAlone',hue ='Survived', data=df);
# Creating new feature Has_Cabin column

df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
sns.countplot(x='Has_Cabin',hue ='Survived', data=df);
# Creating new feature by defining range of fare prices

df.Fare = df.Fare.fillna(df.Fare.mean())

df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

df.loc[ df['Fare'] > 31, 'Fare'] = 3

df['Fare'] = df['Fare'].astype(int)
# Filling NaN value for Age column with median

df['Age'] = df['Age'].fillna(df['Age'].median())
# Creating new feature from name Title

df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df.Title.unique()
# Count of Unique value for column Title

df.Title.value_counts()
# Creating new feature from Title column 

df['Title'] = df['Title'].replace(['Lady','the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')
sns.countplot(x='Title',hue ='Survived', data=df);
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']

df = df.drop(drop_elements, axis = 1)
df.head(2)
#### Again Check Missing Value

df.isnull().sum()/(df.isnull().count()*1).sort_values(ascending = False)
# Export dataset

df.to_csv("cleaned_data.csv")
np=df.copy()

np['Title'] = np['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master':4, 'Rare':5})
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(np.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
#### Lets map Title column Mr, Miss etc with one hot encoding

df = pd.get_dummies(df,prefix=['Title'], drop_first=True)
X = train.drop(['Survived'], axis=1)

Y = train.Survived

# Create Train & Test Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
# Resetting index

df = df.reset_index(drop=True)
# Seperating Train and Test Set

train_set = df[~df.Survived.isnull()]

test_set = df[df.Survived.isnull()]
# Shape

print(train_set.shape,test_set.shape)
# Defining X and Y

X = train_set.drop(['Survived'], axis=1)

y = train_set.Survived

# Droping target columns

test_set = test_set.drop(['Survived'], axis=1)
# train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Running logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

result = model.fit(X_train, y_train)

prediction_test = model.predict(X_test)

# Print the prediction accuracy

lr = round(metrics.accuracy_score(y_test, prediction_test)*100,2)
model.svm = SVC(kernel='linear') 

model.svm.fit(X_train,y_train)

preds = model.svm.predict(X_test)

svc = round(metrics.accuracy_score(y_test, preds)*100,2)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

xgb = round(metrics.accuracy_score(y_test, preds)*100,2)
model_rf = RandomForestClassifier(min_samples_leaf = 3, 

                                       n_estimators=200, 

                                       max_features=0.5, 

                                       n_jobs=-1)

model_rf.fit(X_train, y_train)



# Make predictions

prediction_test = model_rf.predict(X_test)

rf = round(metrics.accuracy_score(y_test, prediction_test)*100,2)
importances = pd.DataFrame({'feature':X_train.columns,'importance':model_rf.feature_importances_})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances
results = pd.DataFrame({

    'Model': ['Support Vector Machines','Logistic Regression', 

              'Random Forest','XGB Classifier'],

    'Score': [svc,lr,rf,xgb]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head()
import itertools    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, prediction_test)



np.set_printoptions(precision=2)

class_names = ['Not Survived','Survived']

# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()



from sklearn.metrics import classification_report

eval_metrics = classification_report(y_test, preds, target_names=class_names)

print(eval_metrics)
#Prediction on test set

test_pred = model_rf.predict(test_set)
test_pred
# creating series 

df = pd.Series(test_pred) 

# Changing Dtype

df = df.astype('Int64')

# Provide 'Predicted' as the column name 

ndf = pd.DataFrame()

ndf['Survived'] = df
# Reset Index

test = test_set['PassengerId'].reset_index(drop=True)
# Submission File

subm_file = pd.concat([test,ndf], axis=1)
# Export File

subm_file.to_csv("submission_data.csv",index=False)
#--not to use--#

# prepare configuration for cross validation test harness

seed = 7

# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', DecisionTreeClassifier()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()