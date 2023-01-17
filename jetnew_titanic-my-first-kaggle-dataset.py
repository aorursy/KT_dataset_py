# Libraries for numerical analysis
import numpy as np # linear algebra and matrix operations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for data mungling
from sklearn import preprocessing # preprocessing.scale() performs mean normalization and feature scaling across the dataset

# Common Machine Learning Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.impute import SimpleImputer # SimpleImputer fills empty cells with the mean
from sklearn.model_selection import cross_validate # To train with cross validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Others
import os
import warnings
warnings.filterwarnings('ignore') # Had a lot of future warning messages

# Import datasets
df_train = pd.read_csv('../input/train.csv') # Training set (The data we are given)
df_test = pd.read_csv('../input/test.csv') # Test set (The data we need to test for results)
# Visualise numerical features
display(df_train.describe())
# Visualise categorical features
display(df_train.describe(include=['O']))
sns.heatmap(df_train.corr(method='pearson'),annot=True,cmap="YlGnBu")
# Graph individual features by survival
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=df_train)
fig, ax = plt.subplots(1,2)
sns.barplot(x = 'SibSp', y = 'Survived', order=[1,2,3,4,5,6,7], data=df_train, ax=ax[0])
sns.barplot(x = 'Parch', y = 'Survived', order=[1,2,3,4,5,6], data=df_train, ax=ax[1])
y = df_train['Survived']
test_index = df_test['PassengerId']
# For simultaneous data cleaning
combine = [df_train, df_test]
# For feature: Name
# Convert: Name -> Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # regex

# Convert: rare titles -> Rare, Mlle -> Miss, Ms -> Miss, Mme -> Mrs
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Display the available types in Title feature.
sns.countplot(x = 'Title', order=df_train['Title'].unique(), data=df_train)
# For feature: Sex
# Convert: female -> 1, male -> 0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# For feature: Embarked
# Fill in missing values with highest frequency port
freq_port = df_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
# Convert: S -> 0, C -> 1, Q -> 2
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# For feature: Title
# Convert: Mr -> 1, Miss -> 2, Mrs -> 3, Master -> 4, Rare -> 5
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Display the values in each feature.
print("Values in Sex:", df_train['Sex'].unique())
print("Values in Embarked:", df_train['Embarked'].unique())
print("Values in Title:", sorted(df_train['Title'].unique()))
sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x = 'Sex', y = 'Survived', order=[1,0], data=df_train)
# Fill in missing values with median
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

# Split Fare into 4 bands in FareBand to find out where the bins are
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# Convert Ranges of fare prices into bins
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

sns.barplot(x = 'Fare', y = 'Survived',order=[0,1,2,3],data=df_train)
# We converted Fare into 4 bins, so FareBand is no longer useful to us, hence we drop it
df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]
sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])
for dataset in combine:
    dataset.loc[(dataset.Age.isnull()), 'Age'] = dataset.Age.median()
df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
sns.barplot(x = 'Age', y = 'Survived',order=[0,1,2,3,4],data=df_train)
df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]
df_train.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

sns.barplot(x = 'FamilySize', y = 'Survived', order=[1,2,3,4,5,6,7], data=df_train)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

sns.barplot(x = 'IsAlone', y = 'Survived',order=[0,1],data=df_train)
display(df_train.corr(method='pearson')[['Survived']])
sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])

df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

display(df_train.head())
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

df_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(5)
df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]
display(combine[0].head())
display(combine[1].head())
# df_train_X is a list of features used for model training.
df_train_X = combine[0].drop(["Survived", "PassengerId"], axis=1)
# train_y is the training output.
train_y = combine[0]["Survived"]
test_test_X = combine[1].drop("PassengerId", axis=1).copy()
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = {}

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, df_train_X, train_y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    # Fit, predict test input and evaluate using F1 score.
    alg.fit(df_train_X, train_y)
    MLA_compare.loc[row_index, 'F1 Score'] = metrics.f1_score(train_y, alg.predict(df_train_X))
    MLA_predict[MLA_name] = alg.predict(test_test_X)
    
    row_index+=1
    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['F1 Score'], ascending = False, inplace = True)
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='F1 Score', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm F1 Score \n')
plt.xlabel('F1 Score (%)')
plt.ylabel('Algorithm')
best_model = MLA_compare.loc[MLA_compare['F1 Score'].idxmax()]['MLA Name']
best_model_score = round(MLA_compare.loc[MLA_compare['F1 Score'].idxmax()]['F1 Score'],3)
print("Best model:",best_model)
print("F1 Score:",best_model_score)
# predict against test set
predictions = MLA_predict[best_model]
predictions = predictions.ravel() # To reduce ND-array to 1D-array
data_to_submit = pd.DataFrame({
    'PassengerId': test_index,
    'Survived': predictions
})
# output results to results.csv
data_to_submit.to_csv("results.csv", index=False)