# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt
# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
df = pd.read_csv('../input/train.csv')
df_raw = df.copy()
df.head()
draw_missing_data_table(df)
# Dropping columns not being used
df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Making some columns categorical
df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])
df.dtypes
# Extract titles from name
df['Title']=0
for i in df:
    df['Title']=df_raw['Name'].str.extract('([A-Za-z]+)\.', expand=False)  # Use REGEX to define a search pattern
df.head()
df['Title'].unique()
plt.figure(figsize=(15,10))
sns.barplot(data=df, x='Title', y='Age')
# Means by title
df_raw['Title'] = df['Title']  # To simplify data handling
means = df_raw.groupby('Title')['Age'].mean()
map_means = means.to_dict()
map_means
# Drop Name now that we have extracted the title
df.drop('Name', 1, inplace=True)
# Impute ages based on titles
idx_nan_age = df.loc[np.isnan(df['Age'])].index
df.loc[idx_nan_age,'Age'] = df['Title'].loc[idx_nan_age].map(map_means)
# Identify imputed data
df['Imputed'] = 0
df.at[idx_nan_age.values, 'Imputed'] = 1
df.head()
df.groupby('Title')['Embarked'].count()
# Map of aggregated titles:
titles_dict = {'Capt': 'Other',
               'Major': 'Other',
               'Jonkheer': 'Other',
               'Don': 'Other',
               'Sir': 'Other',
               'Dr': 'Other',
               'Rev': 'Other',
               'Countess': 'Other',
               'Dona': 'Other',
               'Mme': 'Mrs',
               'Mlle': 'Miss',
               'Ms': 'Miss',
               'Mr': 'Mr',
               'Mrs': 'Mrs',
               'Miss': 'Miss',
               'Master': 'Master',
               'Lady': 'Other'}
# Group titles
df['Title'] = df['Title'].map(titles_dict)
df.groupby('Title')['Embarked'].count()
percent_survived =  df['Survived'].mean()
sns.barplot(x='Pclass', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 4], [percent_survived, percent_survived], c='black')
sns.barplot(x='Sex', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 2], [percent_survived, percent_survived], c='black')
sns.barplot(x='Embarked', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 3], [percent_survived, percent_survived], c='black')
# Remove 2 rows where Emabarked is null
df = df.loc[~df.isna().values[:, 7]]
sns.barplot(x='Title', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 5], [percent_survived, percent_survived], c='black')
plt.figure(figsize=(25,10))
sns.barplot(x='Age', y='Survived', data=df, ci=None)
plt.plot([-1, 80], [percent_survived, percent_survived], c='black')
plt.xticks(rotation=90)
df['Agebin5'] = 0
for i in df:
    df['Agebin5'] = df['Age'] - (df['Age'] % 5)
df['Agebin10'] = 0
for i in df:
    df['Agebin10'] = df['Age'] - (df['Age'] % 10)
sns.barplot(x='Agebin5', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 16], [percent_survived, percent_survived], c='black')
sns.barplot(x='Agebin10', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 16], [percent_survived, percent_survived], c='black')
# Bin data
df['Age'] = pd.cut(df['Age'], bins=[0, 10, 200], labels=['Child','Adult'])
df['Age'].head()
# Remove Agebins used for plotting
df.drop(['Agebin5', 'Agebin10'], axis=1, inplace=True)
sns.barplot(x='SibSp', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 16], [percent_survived, percent_survived], c='black')
sns.barplot(x='Parch', y='Survived', data=df, capsize=0.05)
plt.plot([-1, 16], [percent_survived, percent_survived], c='black')
sns.boxplot(x='Survived', y='Fare', data=df)
sns.barplot(x='Survived', y='Fare', hue='Pclass', data=df, capsize=0.05)
plt.plot([-1, 2], [percent_survived, percent_survived], c='black')
df = pd.get_dummies(df, drop_first=True)
df.head()
draw_missing_data_table(df)
# Get training and test sets
from sklearn.model_selection import train_test_split

X = df[df.loc[:, df.columns != 'Survived'].columns]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# Apply Box-Cox transformation
from scipy.stats import boxcox

X_train_transformed = X_train.copy()
X_train_transformed['Fare'] = boxcox(X_train_transformed['Fare'] + 1)[0]
X_test_transformed = X_test.copy()
X_test_transformed['Fare'] = boxcox(X_test_transformed['Fare'] + 1)[0]
# Rescale data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
X_test_transformed_scaled = scaler.transform(X_test_transformed)
# Get polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2).fit(X_train_transformed)
X_train_poly = poly.transform(X_train_transformed_scaled)
X_test_poly = poly.transform(X_test_transformed_scaled)
# Select features using chi-squared test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## Get score using original model
logreg = LogisticRegression(C=1)
logreg.fit(X_train, y_train)
scores = cross_val_score(logreg, X_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score = np.mean(scores)

## Get score using models with feature selection
for i in range(1, X_train_poly.shape[1]+1, 1):
    # Select i features
    select = SelectKBest(score_func=chi2, k=i)
    select.fit(X_train_poly, y_train)
    X_train_poly_selected = select.transform(X_train_poly)

    # Model with i features selected
    logreg.fit(X_train_poly_selected, y_train)
    scores = cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
    print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i, 
                                                                     np.mean(scores), 
                                                                     np.std(scores)))
    
    # Save results if best score
    if np.mean(scores) > highest_score:
        highest_score = np.mean(scores)
        std = np.std(scores)
        k_features_highest_score = i
    elif np.mean(scores) == highest_score:
        if np.std(scores) < std:
            highest_score = np.mean(scores)
            std = np.std(scores)
            k_features_highest_score = i
        
# Print the number of features
print('Number of features when highest score: %i' % k_features_highest_score)
# Select features
select = SelectKBest(score_func=chi2, k=k_features_highest_score)
select.fit(X_train_poly, y_train)
X_train_poly_selected = select.transform(X_train_poly)
# Fit model
logreg = LogisticRegression(C=1)
logreg.fit(X_train_poly_selected, y_train)
# Model performance
scores = cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# Plot learning curves
title = "Learning Curves (Logistic Regression)"
cv = 10
plot_learning_curve(logreg, title, X_train_poly_selected, 
                    y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);

# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
cv = 10
plot_validation_curve(estimator=logreg, title=title, X=X_train_poly_selected, y=y_train, 
                      param_name=param_name, ylim=(0.7, 1.01), param_range=param_range);
# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = range(1,21)
cv = 10
plot_validation_curve(estimator=logreg, title=title, X=X_train_poly_selected, y=y_train, 
                      param_name=param_name, ylim=(0.8, .87), param_range=param_range);
# Fit model
logreg = LogisticRegression(C=10)
logreg.fit(X_train_poly_selected, y_train)
df = pd.read_csv('../input/test.csv')
df_raw = df.copy()
passenger_id = df['PassengerId'].values
df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])
df['Title']=0
for i in df:
    df['Title']=df_raw['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df.drop('Name', 1, inplace=True)
idx_nan_age = df.loc[np.isnan(df['Age'])].index
df.loc[idx_nan_age,'Age'] = df['Title'].loc[idx_nan_age].map(map_means)
df['Imputed'] = 0
df.at[idx_nan_age.values, 'Imputed'] = 1
df['Title'] = df['Title'].map(titles_dict)
df['Age'] = pd.cut(df['Age'], bins=[0, 10, 200], labels=['Child','Adult'])

draw_missing_data_table(df)
# There is one missing value in 'Fare'
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())  
# And 2 in Title
df['Title'] = df['Title'].fillna('Other')
draw_missing_data_table(df)
df = pd.get_dummies(df, drop_first=True)

X = df[df.loc[:, df.columns != 'Survived'].columns]

X_transformed = X.copy()
X_transformed['Fare'] = boxcox(X_transformed['Fare'] + 1)[0]

scaler = MinMaxScaler()
X_transformed_scaled = scaler.fit_transform(X_transformed)

poly = PolynomialFeatures(degree=2).fit(X_transformed)
X_poly = poly.transform(X_transformed_scaled)

X_poly_selected = select.transform(X_poly)
# Make predictions
predictions = logreg.predict(X_poly_selected)
# Generate submission file
submission = pd.DataFrame({ 'PassengerId': passenger_id,
                            'Survived': predictions})
submission.to_csv("submission.csv", index=False)