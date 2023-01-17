import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
# Set seed value for reproducing the same results
seed = 101
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# Train data preview
train_data.head()
# Test data preview
test_data.head()
# Summary of train data
train_data.info()
# Summary of test data
test_data.info()
# Train data descriptive statistics
train_data.describe()
# Test data descriptive statistics
test_data.describe()
plt.subplots(figsize=(7, 5))
plt.boxplot(train_data['Fare'])
plt.title('Boxplot of Fare')
plt.show()
# Retrieve rows with Fare greater than 500
train_data[train_data['Fare']>500]
# Retrieve rows with Fare equal to 0
train_data[train_data['Fare']==0]
# Number of missing values in each column in train data
train_data.isnull().sum()
# Number of missing values in each column in test data
test_data.isnull().sum()
# Function to extract title from passenger's name
def extract_title(df):
    title = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    return title
# Count of each title in train data
train_data['Title'] = extract_title(train_data)
train_data['Title'].value_counts()
# Count of each title in test data
test_data['Title'] = extract_title(test_data)
test_data['Title'].value_counts()
# Function to map titles to main categories
def map_title(df):
    title_category = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
    }
    new_title = df['Title'].map(title_category)
    return new_title
# Count of each title in train data after mapping
train_data['Title'] = map_title(train_data)
train_data['Title'].value_counts()
# Count of each title in test data after mapping
test_data['Title'] = map_title(test_data)
test_data['Title'].value_counts()
# Group train data by 'Pclass', 'Title' and calculate the median age
train_data.groupby(['Pclass', 'Title']).median()['Age']
# Function to identify passengers who have the title 'Miss' and, 1 or 2 value in the 'Parch' column
def is_young(df):
    young = []
    for index, value in df['Parch'].items():
        if ((df.loc[index, 'Title'] == 'Miss') and (value == 1 or value == 2)):
            young.append(1)
        else:
            young.append(0)
    return young
# Group train data by 'Pclass', 'Title', 'Is_Young(Miss)' and calculate the median age
train_data['Is_Young(Miss)'] = is_young(train_data)
grouped_age = train_data.groupby(['Pclass', 'Title', 'Is_Young(Miss)']).median()['Age']
grouped_age
test_data['Is_Young(Miss)'] = is_young(test_data)
# Fill missing age values in train and test data
train_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
train_data['Age'].fillna(grouped_age, inplace=True)
train_data.reset_index(drop=True, inplace=True)
test_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
test_data['Age'].fillna(grouped_age, inplace=True)
test_data.reset_index(drop=True, inplace=True)
# Group train data by 'Pclass' and calculate the median fare
grouped_fare = train_data.groupby('Pclass').median()['Fare']
grouped_fare
# Fill the missing fare value in test data
test_data.set_index('Pclass', drop=False, inplace=True)
test_data['Fare'].fillna(grouped_fare, inplace=True)
test_data.reset_index(drop=True, inplace=True)
# Drop unnecessary rows and columns
train_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)
test_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)
train_data.dropna(subset=['Embarked'], inplace=True)
# Missing values in train data after data cleaning
train_data.isnull().sum()
# Missing values in test data after data cleaning
test_data.isnull().sum()
plt.subplots(figsize=(7, 5))
sns.countplot(x='Survived', data=train_data)
plt.title('Class Distribution')
plt.show()
plt.subplots(figsize=(7, 5))
sns.barplot(x='Sex', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on sex')
plt.show()
plt.subplots(figsize=(7, 5))
sns.barplot(x='Pclass', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on ticket class')
plt.show()
plt.subplots(figsize=(7, 5))
sns.barplot(x='Embarked', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on port of embarkation')
plt.show()
plt.subplots(figsize=(7, 5))
sns.barplot(x='Title', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on title')
plt.show()
# Encode 'Sex' variable values
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
test_data['Sex'] = le.transform(test_data['Sex'])
# Convert 'Embarked' and 'Title' into dummy variables
train_data = pd.get_dummies(train_data, columns=['Embarked', 'Title'])
test_data = pd.get_dummies(test_data, columns=['Embarked', 'Title'])
train_data.head()
# Pairwise correlation of columns
corr = train_data.corr()
corr
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, cmap='RdBu_r', linewidths=.5, cbar_kws={'shrink': .7})
plt.show()
# Apply feature scaling using MinMaxScaler
scaler = MinMaxScaler()
train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:])
test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])
train_data.head()
X_train, X_test, y_train = train_data.iloc[:, 2:], test_data.iloc[:, 1:], train_data['Survived']
# Function to generate submission file to get test score
def submission(preds):
    test_data['Survived'] = preds
    predictions = test_data[['PassengerId', 'Survived']]
    predictions.to_csv('submission.csv', index=False)
# Classification model
logreg = LogisticRegression()

# Parameters to tune
params = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
           'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
lr_clf = GridSearchCV(logreg, params, cv=cv, n_jobs=-1)
lr_clf.fit(X_train, y_train)
# Best parameters
lr_clf.best_params_
# Train score
lr_clf.best_score_
# Test score
y_preds = lr_clf.predict(X_test)
submission(y_preds)
# Classification model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# Test score
y_preds = gnb.predict(X_test)
submission(y_preds)
# Classification model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
# Test score
y_preds = lda.predict(X_test)
submission(y_preds)
# Classification model
knn = KNeighborsClassifier()

# Parameters to tune
params = [{'n_neighbors': range(1, 21),
           'p': [1, 2]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
knn_clf = GridSearchCV(knn, params, cv=cv, n_jobs=-1)
knn_clf.fit(X_train, y_train)
# Best parameters
knn_clf.best_params_
# Train score
knn_clf.best_score_
# Test score
y_preds = knn_clf.predict(X_test)
submission(y_preds)
# Classification model
svm = SVC(max_iter=10000)

# Parameters to tune
params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
           'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
svm_clf = GridSearchCV(svm, params, cv=cv, n_jobs=-1)
svm_clf.fit(X_train, y_train)
# Best parameters
svm_clf.best_params_
# Train score
svm_clf.best_score_
# Test score
y_preds = svm_clf.predict(X_test)
submission(y_preds)
# Classification model
dt = DecisionTreeClassifier(random_state=seed)

# Parameters to tune
params = [{'max_depth': [5, 7, 10, None],
           'min_samples_split': [2, 5, 10],
           'max_features': ['sqrt', 5, 7, 10]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
dt_clf = GridSearchCV(dt, params, cv=cv, n_jobs=-1)
dt_clf.fit(X_train, y_train)
# Best parameters
dt_clf.best_params_
# Train score
dt_clf.best_score_
# Test score
y_preds = dt_clf.predict(X_test)
submission(y_preds)
# Note: This cell will take a while to run depending on the available processing power

# Classification model
rf = RandomForestClassifier(random_state=seed)

# Parameters to tune
params = [{'n_estimators': range(50, 550, 50),
           'max_depth': [5, 7, 10, None],
           'min_samples_split': [2, 5, 10],
           'max_features': ['sqrt', 5, 7, 10]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
rf_clf = GridSearchCV(rf, params, cv=cv, n_jobs=-1)
rf_clf.fit(X_train, y_train)
# Best parameters
rf_clf.best_params_
# Train score
rf_clf.best_score_
# Test score
y_preds = rf_clf.predict(X_test)
submission(y_preds)
# Note: This cell will take a while to run depending on the available processing power

# Classification model
xgboost = xgb.XGBClassifier(random_state=seed)

# Parameters to tune
params = [{'max_depth': [3, 5, 10],
           'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
           'n_estimators': range(100, 1100, 100)}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
xgb_clf = GridSearchCV(xgboost, params, cv=cv, n_jobs=-1)
xgb_clf.fit(X_train, y_train)
# Best parameters
xgb_clf.best_params_
# Train score
xgb_clf.best_score_
# Test score
y_preds = xgb_clf.predict(X_test)
submission(y_preds)
# Models that we will input to stacking classifier
base_estimators = list()
base_estimators.append(('lda', lda))
base_estimators.append(('knn', knn_clf.best_estimator_))
base_estimators.append(('svm', svm_clf.best_estimator_))
base_estimators.append(('dt', dt_clf.best_estimator_))
base_estimators.append(('rf', rf_clf.best_estimator_))

# Stacking classifier
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
stacking_clf.fit(X_train, y_train)
# Test score
y_preds = stacking_clf.predict(X_test)
submission(y_preds)