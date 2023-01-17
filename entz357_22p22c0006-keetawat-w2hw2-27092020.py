# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
train.tail()
train.info()
print('-'*40)
test.info()
train.isnull().sum()
train.describe()
train.describe(include=['O'])
train.nunique()
plt.figure(figsize=(6, 5))
ax = sns.countplot(x = 'Survived', data=train)
plt.title('Count of Survival', fontsize=15)
plt.xlabel('Survived vs Died')
plt.ylabel('Number of passengers')
plt.xticks([0, 1], ['Died', 'Survived'])
train['Survived'].value_counts(normalize=True)
# Age distribution
plt.figure(figsize=(14,4))
sns.distplot(train[(train['Age']>0)]['Age'], bins=50)

plt.title('Distribution of passengers age', fontsize=15)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
# Age by surviving status

plt.figure(figsize=(14,4))
sns.boxplot(data=train, y='Survived', x='Age', orient='h')
sns.stripplot(data=train, y='Survived', x='Age', orient='h')
plt.yticks([0,1], ['Died', 'Survived'])
plt.title('Age distribution grouped by surviving status')
plt.ylabel('Passerger status')
plt.tight_layout()
pd.DataFrame(train.groupby('Survived')['Age'].describe())
def add_family_group_size(df):
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_size_group'] = df['Family_size'].map(lambda x:
                                                    'f_single' if x==1
                                                   else ('f_usual' if 5 > x >= 2
                                                        else ('f_big' if 8 > x >= 5
                                                             else 'f_large')))
    df.drop(['SibSp', 'Parch', 'Family_size'], axis=1, inplace=True)
    return df

train = add_family_group_size(train)
print('Family size group and number of passengers:')
print(train['Family_size_group'].value_counts())
fig = plt.figure(figsize = (12,4))

ax1 = fig.add_subplot(121)
ax = sns.countplot(train['Family_size_group'], ax = ax1)
    
plt.title('Passengers distribution by family size')
plt.ylabel('Number of passengers')

ax2 = fig.add_subplot(122)
d = train.groupby('Family_size_group')['Survived'].value_counts(normalize = True).unstack()
d.plot(kind='bar', stacked='True', ax = ax2)
plt.title('Proportion of survived/died passengers by family size (train data)')
plt.legend(( 'Died', 'Survived'), loc=(1.04,0))
plt.xticks(rotation = False)

plt.tight_layout()
fig = plt.figure(figsize = (12,4))

ax1 = fig.add_subplot(121)
ax = sns.countplot(train['Pclass'], ax = ax1)
plt.title('Passengers distribution by family size')
plt.ylabel('Number of passengers')
plt.tight_layout()

ax2 = fig.add_subplot(122)
d = train.groupby('Pclass')['Survived'].value_counts(normalize = True).unstack()
d.plot(kind='bar', stacked='True', ax = ax2)
plt.title('Proportion of survived/died passengers by class (train data)')
plt.legend(( 'Died', 'Survived'), loc=(1.04,0))
_ = plt.xticks(rotation=False)

plt.tight_layout()
fig = plt.figure(figsize = (12,4))

ax1 = fig.add_subplot(121)
ax = sns.countplot(train['Sex'], ax = ax1)
    
plt.title('Passengers distribution by Sex')
plt.ylabel('Number of passengers')

ax2 = fig.add_subplot(122)
d = train.groupby('Sex')['Survived'].value_counts(normalize = True).unstack()
d.plot(kind='bar', stacked='True', ax = ax2)
plt.title('Proportion of survived/died passengers by Sex')
plt.legend(( 'Died', 'Survived'), loc=(1.04,0))
plt.xticks(rotation = False)

plt.tight_layout()
plt.figure(figsize = (15,4))

plt.subplot (1,2,1)
sns.countplot( x = 'Pclass', data = train, hue = 'Sex')
plt.title('Number of male/female passengers by class')
plt.ylabel('Number of passengers')
plt.legend(loc=(1.04,0))

plt.subplot (1,2,2)
sns.countplot( x = 'Family_size_group', data = train, hue = 'Sex', 
              order = train['Family_size_group'].value_counts().index )
plt.title('Number of male/female passengers by family size')
plt.ylabel('Number of passengers')
plt.legend( loc=(1.04,0))
plt.tight_layout()
sns.catplot(x="Pclass", y="Fare",  hue = "Survived", kind="swarm", data=train)
plt.tight_layout()
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    from collections import Counter
    
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.7 * IQR    # increased to 1.7
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers

# detect outliers from Age, SibSp, Parce, Fare
Outliers_to_drop = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
train.loc[Outliers_to_drop]
# Drop outliers
# train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) ## reduce final accuracy
# train = train.drop(train[train['Fare'] > 500].index).reset_index(drop=True) ## also reduce acc
## distribution of cat features
cat_features = train[['Pclass', 'Sex', 'Embarked']].columns
for i in cat_features:
    sns.barplot(y="Survived",x=i,data=train)
    plt.title(i+" by "+"Survived")
    plt.show()
fig = plt.figure(figsize=(12,12))
sns.heatmap(train.corr(), cmap='RdYlGn', annot=True, linewidths=1)
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['Survived'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = train.columns
spearman(train, features)
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
def drop_unnec_col(df):
    unnec_col = ['PassengerId', 'Name', 'Ticket', 'Cabin', "Embarked", 'Age']
    df = df.drop(unnec_col, axis=1)
    return df
def preprocessing(df):
    # Create family group size column
    df = add_family_group_size(df)
    # Drop unneccessary columns
    df = drop_unnec_col(df)
    # Dummy variables
    df = pd.get_dummies(df, columns=["Sex", "Family_size_group"], drop_first=True)
    print(f'{df.shape}')
    return df
train.isnull().sum()
train = preprocessing(train)
train
train.isnull().sum()
test = preprocessing(test)
test.isnull().sum()
test.fillna(test.median(), inplace=True)
test.isnull().sum()
X_train = train.drop("Survived", axis=1).to_numpy()
y_train = train["Survived"].to_numpy()
X_test = test.apply(pd.to_numeric, errors='coerce').to_numpy()
#X_test  = test.to_numpy()
X_train.shape, y_train.shape, X_test.shape
X_train.dtype
X_test.dtype
# Normalize
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
std_scaler = std_scaler.fit(X_train)
X_train_scal = std_scaler.transform(X_train)
X_train_scal
X_test_scal = std_scaler.transform(X_test)
X_test_scal
# Logistic Regression model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_scal, y_train)
y_pred = logreg.predict(X_train_scal)

acc_log = round(logreg.score(X_train_scal, y_train) * 100, 2)
acc_log
# Logistic Regression's accuracy:

from sklearn import metrics

y_pred = logreg.predict(X_train_scal)
logreg_acc = metrics.accuracy_score(y_train, y_pred)
logreg_recall = metrics.recall_score(y_train, y_pred)
logreg_precision = metrics.precision_score(y_train, y_pred)
logreg_f1 = metrics.f1_score(y_train, y_pred)

print('Accuracy    : {0:0.5f}'.format(logreg_acc))
print('Recall      : {0:0.5f}'.format(logreg_recall))
print('Precision   : {0:0.5f}'.format(logreg_precision))
print('F-Measure   : {0:0.5f}'.format(logreg_f1))
# Decision Tree model

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scal, y_train)
y_pred = decision_tree.predict(X_train_scal)
acc_decision_tree = round(decision_tree.score(X_train_scal, y_train) * 100, 2)
acc_decision_tree
# Decision Tree's accuracy:

from sklearn import metrics

y_pred = decision_tree.predict(X_train_scal)
decision_tree_acc = metrics.accuracy_score(y_train, y_pred)
decision_tree_recall = metrics.recall_score(y_train, y_pred)
decision_tree_precision = metrics.precision_score(y_train, y_pred)
decision_tree_f1 = metrics.f1_score(y_train, y_pred)

print('Accuracy    : {0:0.5f}'.format(decision_tree_acc))
print('Recall      : {0:0.5f}'.format(decision_tree_recall))
print('Precision   : {0:0.5f}'.format(decision_tree_precision))
print('F-Measure   : {0:0.5f}'.format(decision_tree_f1))
# Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train_scal, y_train)
y_pred = gaussian.predict(X_train_scal)
acc_gaussian = round(gaussian.score(X_train_scal, y_train) * 100, 2)
acc_gaussian
# Gaussian Naive Bayes's accuracy:

from sklearn import metrics

y_pred = gaussian.predict(X_train_scal)
gaussian_acc = metrics.accuracy_score(y_train, y_pred)
gaussian_recall = metrics.recall_score(y_train, y_pred)
gaussian_precision = metrics.precision_score(y_train, y_pred)
gaussian_f1 = metrics.f1_score(y_train, y_pred)

print('Accuracy    : {0:0.5f}'.format(gaussian_acc))
print('Recall      : {0:0.5f}'.format(gaussian_recall))
print('Precision   : {0:0.5f}'.format(gaussian_precision))
print('F-Measure   : {0:0.5f}'.format(gaussian_f1))
# Random Forest model
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_scal, y_train)
y_pred = random_forest.predict(X_train_scal)
random_forest.score(X_train_scal, y_train)
acc_random_forest = round(random_forest.score(X_train_scal, y_train) * 100, 2)
acc_random_forest
# Random Forest's accuracy:

from sklearn import metrics

y_pred = random_forest.predict(X_train_scal)
random_forest_acc = metrics.accuracy_score(y_train, y_pred)
random_forest_recall = metrics.recall_score(y_train, y_pred)
random_forest_precision = metrics.precision_score(y_train, y_pred)
random_forest_f1 = metrics.f1_score(y_train, y_pred)

print('Accuracy    : {0:0.5f}'.format(random_forest_acc))
print('Recall      : {0:0.5f}'.format(random_forest_recall))
print('Precision   : {0:0.5f}'.format(random_forest_precision))
print('F-Measure   : {0:0.5f}'.format(random_forest_f1))
X_train.shape
# Neural Network model

import tensorflow as tf
from tensorflow.keras import layers, models

NN = tf.keras.Sequential()

# Layers
NN.add(tf.keras.layers.Dense(10, input_shape=X_train.shape[1:], activation='relu'))
NN.add(tf.keras.layers.Dense(10, activation='relu'))
NN.add(tf.keras.layers.Dense(5, activation='relu'))
NN.add(tf.keras.layers.Dense(1, activation='sigmoid'))

NN.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
NN.summary()
NN.fit(X_train_scal, y_train, epochs=400)
y_pred = NN.predict(X_train_scal)
np.round(y_pred).astype(int)
# Neural Network's accuracy:

from sklearn import metrics

y_pred = NN.predict(X_train_scal).round().astype(int)
NN_acc = metrics.accuracy_score(y_train, y_pred)
NN_recall = metrics.recall_score(y_train, y_pred)
NN_precision = metrics.precision_score(y_train, y_pred)
NN_f1 = metrics.f1_score(y_train, y_pred)

print('Accuracy    : {0:0.5f}'.format(NN_acc))
print('Recall      : {0:0.5f}'.format(NN_recall))
print('Precision   : {0:0.5f}'.format(NN_precision))
print('F-Measure   : {0:0.5f}'.format(NN_f1))
y_test_pred = random_forest.predict(X_test_scal)
y_test_pred
y_test = pd.read_csv('../input/titanic/gender_submission.csv')
y_test = y_test['Survived'].apply(pd.to_numeric, errors='coerce').to_numpy()
y_test
y_test_pred = random_forest.predict(X_test_scal)

random_forest_acc = metrics.accuracy_score(y_test, y_test_pred)
random_forest_recall = metrics.recall_score(y_test, y_test_pred)
random_forest_precision = metrics.precision_score(y_test, y_test_pred)
random_forest_f1 = metrics.f1_score(y_test, y_test_pred)

print('Accuracy    : {0:0.5f}'.format(random_forest_acc))
print('Recall      : {0:0.5f}'.format(random_forest_recall))
print('Precision   : {0:0.5f}'.format(random_forest_precision))
print('F-Measure   : {0:0.5f}'.format(random_forest_f1))
# Average F-Measure

avg_f1 = (decision_tree_f1 + gaussian_f1 + NN_f1)/3
avg_f1
