import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Load in dataset
train = pd.read_csv('../input/train.csv')
train.head()
train.info()
train.describe() #Numeric
train.describe(include = ['O']) #Categoric
plt.figure(figsize=(10,8))
cor = train.corr()
sns.heatmap(cor, cmap = sns.color_palette('Blues'), annot = True)
cor
# Age Distribution Plot
plt.figure(figsize=(13,5))
age_plt = sns.distplot(train[np.isfinite(train['Age'])]['Age']) #Ignore Missing Values - Will be addressed later
age_plt.axvline(x = train.mean()['Age'], color = 'red', linewidth = 3, linestyle = 'dotted', label = 'mean')
age_plt.axvline(x = train.median()['Age'], color ='blue', linewidth = 3, linestyle='dotted', label='median')
age_plt.legend(bbox_to_anchor = (1.05, 1), loc = 2)
age_plt.set_title('Age Distribution', fontsize = 20)

# Age vs. Survival Plot
age_survival_plt = sns.FacetGrid(train, col = 'Survived',  hue_kws = {'color': ['thistle']}, aspect = 1.3, size = 5)
age_survival_plt.map(plt.hist, 'Age', bins = 20)
plt.suptitle('Does age influence survival rates?', fontsize = 22, y = 1.05) 

# Age vs. Survival Distribution
age_survival_dist = sns.FacetGrid(train, hue = 'Survived', aspect = 2.5, size = 5)
age_survival_dist.map(sns.kdeplot, 'Age', shade = True)
age_survival_dist.add_legend()
plt.suptitle('Age vs. Survival Distribution', fontsize = 20, y = 1.05)
# Contingency Table
train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Pclass vs. Survival Plot
pclass_survival_plt = sns.FacetGrid(train, col = 'Survived', row = 'Sex', hue_kws = {'color': ['cornflowerblue']}, aspect = 1.5, size = 4)
pclass_survival_plt.map(plt.hist, 'Age', bins = 20)
plt.suptitle('Does gender influence survival rates?', fontsize = 22, y = 1.05) 

# Gender vs. Survival Distribution 
sex_survival_dist = sns.FacetGrid(train, hue = 'Survived', row = 'Sex', aspect = 2.5, size = 5)
sex_survival_dist.map(sns.kdeplot, 'Age', shade = True)
sex_survival_dist.add_legend()
plt.suptitle('Sex vs. Survival Distribution', fontsize = 20, y = 1.02)
# Contingency Table
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Pclass vs. Survival Plot
pclass_survival_plt = sns.FacetGrid(train, col = 'Survived', row = 'Pclass', hue_kws = {'color': ['cadetblue']}, aspect = 1.5, size = 4)
pclass_survival_plt.map(plt.hist, 'Age', bins = 20)
plt.suptitle('Pclass vs. Survival', fontsize = 22, y = 1.02) 
# Fare Distribution Plot
plt.figure(figsize=(13,5))
fare_plt = sns.distplot(train['Fare'])
fare_plt.axvline(x = train.mean()['Fare'], color = 'red', linewidth = 3, linestyle = 'dotted', label = 'mean')
fare_plt.axvline(x = train.median()['Fare'], color ='blue', linewidth = 3, linestyle='dotted', label='median')
fare_plt.legend(bbox_to_anchor = (1.05, 1), loc = 2)
fare_plt.set_title('Fare Distribution', fontsize = 20)

# Fare vs. Survival Plot
pclass_survival_plt = sns.FacetGrid(train, col = 'Survived', row = 'Pclass', hue_kws = {'color': ['lightblue']}, aspect = 1.5, size = 3)
pclass_survival_plt.map(plt.hist, 'Fare', bins = 10)
plt.suptitle('Fare vs. Survival', fontsize = 22, y = 1.02) 
# Embarked Survival Rates
embarked_plt = sns.FacetGrid(train, aspect = 2, size = 4)
embarked_plt.map(sns.barplot, 'Embarked', 'Survived', color = 'plum')
plt.suptitle('Embarked Survival Rates', fontsize = 22, y = 1.02) 
# Embarked Gender Distribution
embarked_gender_dist = sns.FacetGrid(train, aspect = 2, size = 4)
embarked_gender_dist.map(sns.countplot, 'Embarked', hue = 'Sex', data = train)
embarked_gender_dist.add_legend()
plt.suptitle('Embarked Gender Distribution', fontsize = 22, y = 1.02) 

# Embarked Pclass Distribution
embarked_pclass_dist = sns.FacetGrid(train, aspect = 2, size = 4)
embarked_pclass_dist.map(sns.countplot, 'Embarked', hue = 'Pclass', data = train)
embarked_pclass_dist.add_legend()
plt.suptitle('Embarked Pclass Distribution', fontsize = 22, y = 1.02)
# Embarked vs. Gender vs. Pclass Plot
embarked_survival_plt = sns.FacetGrid(train, row='Embarked', size=2.5, aspect=3)
embarked_survival_plt.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'Paired')
embarked_survival_plt.add_legend()
plt.suptitle('Do people embarking at different ports have different gender or wealth status?', fontsize = 20, y = 1.05)
train[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Parch')
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'SibSp')
test = pd.read_csv('../input/test.csv')
titanic = train.append(test, ignore_index = True)
titanic.tail()
titanic = titanic.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
titanic['Title'] = titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(titanic['Title'], titanic['Sex'])
title_plt = sns.FacetGrid(titanic, aspect = 4, size = 3)
title_plt.map(sns.barplot, 'Title', 'Survived', color = 'lightblue')
plt.suptitle('Title Survival Rates', fontsize = 22, y = 1.02) 
# Classify Titles
titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')
titanic['Title'] = titanic['Title'].replace(['Mlle', 'Ms'], 'Miss')
titanic['Title'] = titanic['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royalty - Women')
titanic['Title'] = titanic['Title'].replace(['Don', 'Sir', 'Jonkheer'], 'Royalty - Men')
titanic[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
titanic = titanic.drop(['Name'], axis = 1)
# New Category: Family = Parch + SibSp
titanic['Family Size'] = titanic['Parch'] + titanic['SibSp']

# Classify Families Based on Size
titanic['Family Size'] = titanic['Family Size'].replace(0, 'None')
titanic['Family Size'] = titanic['Family Size'].replace([1,2], 'Small')
titanic['Family Size'] = titanic['Family Size'].replace([3,4], 'Medium')
titanic['Family Size'] = titanic['Family Size'].replace([5,6,7,8,9,10], 'Large')
titanic[['Family Size', 'Survived']].groupby(['Family Size'], as_index=False).mean()
titanic = titanic.drop(['Parch', 'SibSp'], axis = 1)
titanic.isnull().sum()
# Embarked
freq_port = titanic.Embarked.dropna().mode()[0] #freq_port = S
titanic['Embarked'] = titanic['Embarked'].fillna(freq_port)

# Fare
titanic['Fare'].fillna(titanic['Fare'].dropna().median(), inplace=True)
# Average Age Based on Title
ave_age = titanic[['Title', 'Age']].groupby(['Title'], as_index=False).mean()
ave_age
# Age
title_names = ave_age['Title'].tolist()

for name in title_names:
    age = ave_age['Age'].loc[ave_age['Title'] == name].item()
    titanic['Age'].loc[titanic['Title'] == name] = titanic['Age'].loc[titanic['Title'] == name].fillna(age) 
    #Bug doesn't allow inplace=True to work with loc
# Make sure all missing variables are gone
titanic.isnull().sum()
# Age
#bins = (0, 5, 12, 18, 25, 35, 60, 100)
#age_labels = ['Baby', 'Child', 'Teenager', 'Minor', 'Young Adult', 'Adult', 'Senior']
#titanic['Age'] = pd.cut(titanic['Age'], bins, labels = age_labels)

#Fare
#fare_labels = [1, 2, 3, 4, 5]
#titanic['Fare Quintile'] = pd.qcut(titanic['Fare'], 5, fare_labels)
#titanic = titanic.drop(['Fare'], axis = 1)
#titanic[['Age', 'Survived']].groupby(['Age'], as_index = False).mean()
#titanic[['Fare Quintile', 'Survived']].groupby(['Fare Quintile'], as_index = False).mean()
# Manually Mapping
#titanic['Embarked'] = titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#titanic['Family Size'] = titanic['Family Size'].map( {'None': 0, 'Small': 1, 'Medium': 2, 'Large': 3} ).astype(int)
#titanic['Sex'] = titanic['Sex'].map( {'Female': 0, 'Male': 1} ).astype(int)
#titanic['Title'] = titanic['Title'].map( {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Officer': 4, 'Royalty - Men': 5, 'Royalty - Women': 6 } ).astype(int)
# One-hot Encode with get_dummies
titanic = pd.get_dummies(titanic)
train_cleaned = titanic[0:890]
test_cleaned = titanic[891:]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
X_train = train_cleaned.drop('Survived', axis=1) #Features
y_train = train_cleaned['Survived'] #Target
X_test  = test_cleaned.drop('Survived', axis=1)
X_train.shape, y_train.shape, X_test.shape #Verify everything has the right dimensions
models = {
    'Logistic Regression' : LogisticRegression(),
    'KNN' : KNeighborsClassifier(),
    'SVM' : SVC(),
    'Naive Bayes Classifier' : GaussianNB(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Perceptron' : Perceptron()
}
def run_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    model.predict(X_test)
    return{
        'Accuracy': round(model.score(X_train, y_train) * 100, 2)
    }
results = {}
model_name = models.keys()
for name in tqdm(model_name):
    results[name] = run_model(models[name], X_train, y_train, X_test)
pd.DataFrame(results).T.sort_values(by = 'Accuracy', ascending = False)
# How to Create a Submission File
submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': DecisionTreeClassifier().fit(X_train, y_train).predict(X_test).astype(int)
    })
submission.to_csv('submission.csv', index=False)
# New Train Test Split
features = train_cleaned.drop('Survived', axis=1)
target = train_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=100)
def run_model(model, model_name, x_train, x_test, y_train, y_test):
    
    _ = model.fit(x_train, y_train)
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)   
    
    return {
            'Model Name' : model_name,
            'Train Score' : round(model.score(x_train, y_train) * 100, 2),
            'Test Score' : round(model.score(x_test, y_test) * 100, 2)
    }

results = {}
model_name = models.keys()
for name in tqdm(model_name):
    results[name] = run_model(models[name], name, X_train, X_test, y_train, y_test)
    
pd.DataFrame(results).T
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
logr = LogisticRegression()
parameters = {'penalty': ['l1', 'l2'], 
              'C' : range(20,40)
             }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(logr, parameters, cv = 10, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
logr = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
logr.fit(X_train, y_train)
logr.predict(X_test)
round(logr.score(X_train, y_train) * 100, 2)
rfc = RandomForestClassifier()
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy'],
              'max_depth': range(1,10), 
              'min_samples_split': range(2,5),
              'min_samples_leaf': range(1,5),
              'n_estimators': range(2,10)
             }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(rfc, parameters, cv = 10, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)
rfc.predict(X_test)
round(rfc.score(X_train, y_train) * 100, 2)
submission = pd.DataFrame({ 
        'PassengerId': test['PassengerId'],
        'Survived': logr.predict(test_cleaned.drop('Survived', axis=1)).astype(int)
    })
submission.to_csv('submission.csv', index=False)