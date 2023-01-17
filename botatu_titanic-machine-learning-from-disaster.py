import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

%matplotlib inline
sns.set()
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head()
df_test.head()
df_train.info()
df_train.describe()
sns.countplot(x='Survived', data=df_train)
df_test['Survived'] = 0
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
sns.countplot(x='Sex', data=df_train)
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)
df_train.groupby(['Sex']).Survived.sum()
print(df_train[df_train['Sex'] == 'female'].Survived.sum()/df_train[df_train['Sex'] == 'female'].Survived.count())
print(df_train[df_train['Sex'] == 'male'].Survived.sum()/df_train[df_train['Sex'] == 'male'].Survived.count())
df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train)
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train)
sns.distplot(df_train.Fare, kde=False)
df_train.groupby('Survived').Fare.hist(alpha=0.6)
df_train_drop_na = df_train.dropna()
sns.distplot(df_train_drop_na.Age, kde=False)
sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True)
sns.swarmplot(x='Survived', y='Fare', data=df_train)
df_train.groupby('Survived').Fare.describe()
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha': 0.5})
sns.pairplot(df_train_drop_na, hue='Survived')
# Remove earlier predictions in the test set
df_test = df_test.drop(['Survived'], axis=1)

# Save training set predictions
survived_train = df_train.Survived

# Concatenate training and testing sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test], sort=True)
data.info()
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

data.info()
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()
data = data[['Sex_male',  'Fare',  'Age', 'Pclass',  'SibSp']]
data.head()
data.info()
data_train = data.iloc[:891]
data_test = data.iloc[891:]
X = data_train.values
test = data_test.values
y = survived_train.values
# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred

#Submit results to Kaggle
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)
# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over differrent values of k (depth)
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)
    
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)
    
    # Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)
    
# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label='Testing accuracy')
plt.plot(dep, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test], sort=True)

data.info()
data.Name.tail()
# Extract Title from Name, store in column and plot barplot

data['Title'] =data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Mrs'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                       'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Special')
sns.countplot('Title', data=data)
plt.xticks(rotation=45)
data.tail()
data['Has_cabin'] = ~data.Cabin.isnull()

# View head of data
data.head()
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data.Embarked.fillna('S')

data.info()
data.head()
# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)
data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)
data.head()
data.drop(['Age', 'Fare'], inplace=True, axis=1)
data.head()
# Create column of number of Family members onboard (Optional)
# data['Fam_size'] = data.Parch + data.SibSp
data.drop(['Parch', 'SibSp'], inplace=True, axis=1)
data.head()
# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()
# Split into test and train sets
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values
# Set up the hyperparameter grid
dep = np.arange(1, 9)
param_grid = {'max_depth': dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is: {}".format(clf_cv.best_score_))
Y_pred = clf_cv.predict(test)
# Submit to Kaggle 
df_test['Survived'] = Y_pred
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)