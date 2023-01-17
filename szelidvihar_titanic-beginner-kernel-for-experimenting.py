# Settings
show_graphs = True
add_interactions = False
model_tuning = False
feature_selection = False 
# The features with 1 will be used as predictors directly in the models.  

used_features = {
    'PassengerId': 0,
    'Pclass': 1,
    'Name': 0,
    'LastName': 0,
    'Title': 0,                 
    'Sex': 1,
    'Sex-female x Pclass-1-2': 1,
    'Sex-male x Pclass-3': 0,
    'SibSp': 0, 
    'Parch': 0,
    'FamilySize': 0,
    'FamilySizeBin': 0,
    'IsAloneF': 0,
    'Age': 0,
    'AgeBin': 1,
    'IsChild': 0,
    'IsChild x Pclass-1-2': 1,
    'Cabin': 0, 
    'HasCabin': 0,
    'CabinType': 0,
    'Embarked': 0,
    'Ticket': 0,
    'TicketSize': 1,
    'TicketSizeBin': 0,
    'IsAloneT': 1,
    'Fare': 0,
    'FareOrig': 0,
    'FareBin': 1,
    'NameFareSize': 0,
    'Group': 0,
    'GroupSurvived': 1,
    'GroupSize': 0
}
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

# Utils
import os
import scipy
from itertools import compress
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
# Functions
def gridgraph(df, x, col=None, row=None, hue=None, fun=sns.distplot, **fun_kwargs):
    grid = sns.FacetGrid(df, col=col, row=row, hue=hue, size=4)
    grid = grid.map(fun, x, **fun_kwargs)
    grid.add_legend()
    if fun_kwargs['kde']:
        for ax in grid.axes.flat:
            drawmedian(ax)
    
def catgroup(df, catcol, target, fun="mean"):
    print(df.groupby(catcol, as_index=False)[target].agg(fun))
    
def drawmedian(ax):
    for line in ax.get_lines():
        x, y = line.get_data()
        cdf = scipy.integrate.cumtrapz(y, x, initial=0)
        middle = np.abs(cdf-0.5).argmin()

        median_x = x[middle]
        median_y = y[middle]

        ax.vlines(median_x, 0, median_y)
df_train = pd.read_csv('../input/train.csv')
df_valid = pd.read_csv('../input/test.csv')
yt = df_train['Survived']

# Create unified data
df_train['Data'] = 'T'
df_valid['Data'] = 'V'
df_full = pd.concat([df_train, df_valid], sort=False, ignore_index=True) 
mask_train = df_full['Data'] == 'T' 
mask_valid = df_full['Data'] == 'V' 
# Check datatypes, missing values
df_full.info()
# Check stats of the columns.
df_full.describe(include='all')
catgroup(df_full.loc[mask_train], 'Pclass', 'Survived')
if show_graphs:
    sns.barplot(x='Pclass', y='Survived', order=[1,2,3], data=df_full[mask_train], palette='colorblind')
df_full['LastName'] = df_full['Name'].str.extract('^([^,]+),', expand=False)
df_full['Title'] = df_full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Change rare titles to more common categories
dict_replace = {
    "Capt": "Officer",
    "Col": "Officer",
    "Countess": "Noble",
    "Don": "Noble",
    "Dona": "Noble",
    "Dr": "Noble",
    "Jonkheer": "Noble",
    "Lady": "Noble",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Rev": "Noble",
    "Sir": "Noble",
}

df_full['Title'] = df_full['Title'].replace(dict_replace)
catgroup(df_full.loc[mask_train], 'Sex', 'Survived')
df_full['Sex'] = df_full['Sex'].map({'male': 0, 'female': 1})
if show_graphs:
    gridgraph(df_full[mask_train], 'Survived', col='Pclass', row='Sex', bins=2, kde=False)
mask_female = df_full['Sex'] == 1
mask_class12 = df_full['Pclass'].isin([1, 2])
df_full['Sex-female x Pclass-1-2'] = (mask_female & mask_class12).astype(int)
catgroup(df_full.loc[mask_train], 'Sex-female x Pclass-1-2', 'Survived')
mask_male = df_full['Sex'] == 0
mask_class3 = df_full['Pclass'] == 3
df_full['Sex-male x Pclass-3'] = (mask_male & mask_class3).astype(int)
catgroup(df_full.loc[mask_train], 'Sex-male x Pclass-3', 'Survived')
df_full['FamilySize'] = df_full['SibSp'] + df_full['Parch'] + 1
catgroup(df_full[mask_train], 'FamilySize', 'Survived')
if show_graphs:
    sns.barplot(x='FamilySize', y='Survived', data=df_full[mask_train], palette='colorblind')
# Create categories for FamilySize
df_full['FamilySizeBin'] = pd.cut(df_full['FamilySize'], [0, 1, 4, 20], labels=["alone", "normal", "big"])
catgroup(df_full.loc[mask_train], 'FamilySizeBin', 'Survived')
df_full['FamilySizeBin'] = df_full['FamilySizeBin'].map({'alone': 0, 'normal': 1, 'big': 2})
df_full['IsAloneF'] = (df_full['FamilySizeBin'] == 0).astype(int)
catgroup(df_full.loc[mask_train], 'IsAloneF', 'Survived')
# Check who has missing age.
mask_noage = df_full['Age'].isnull()
df_noage = df_full.loc[mask_noage]

df_noage.groupby(['Title', 'Pclass'], as_index=False)['Name'].count()
# Check Age distributions.
if show_graphs:
    bins = np.linspace(0, 100, 20)
    gridgraph(df_full.loc[mask_train], 'Age', col='Pclass', hue='Survived', kde=False, bins=bins)
# Check Age distributions.
if show_graphs:
    bins = np.linspace(0, 100, 20)
    gridgraph(df_full, 'Age', row='Title', kde=True, bins=bins)
# Impute Age
df_medians = df_full.groupby('Title')['Age'].median()
for idx, median in df_medians.iteritems():
    mask_group = df_full['Title'] == idx
    df_full.loc[mask_group & mask_noage, 'Age'] = median
df_full['AgeBin'] = pd.qcut(df_full['Age'], 4, labels=False).astype(int)
if show_graphs:
    sns.barplot(x='AgeBin', y='Survived', data=df_full[mask_train], palette='colorblind')
df_full['IsChild'] = (df_full['Age'] < 16).astype(int)
catgroup(df_full.loc[mask_train], 'IsChild', 'Survived')
mask_class12 = df_full['Pclass'].isin([1, 2])
df_full['IsChild x Pclass-1-2'] = df_full['IsChild'] * mask_class12.astype(int)
catgroup(df_full.loc[mask_train], 'IsChild x Pclass-1-2', 'Survived')
df_full['HasCabin'] = df_full['Cabin'].notnull().astype(int)
mask_class1 = (df_full['Pclass'] == 1).astype(int)

print("Correlation with 1st class: ", df_full['HasCabin'].corr(mask_class1))
catgroup(df_full.loc[mask_train], 'HasCabin', 'Survived')
df_full['CabinType'] = df_full['Cabin'].str[0]
catgroup(df_full.loc[mask_train], 'CabinType', 'Survived')
# See who is missing Embarked
mask_noembarked = df_full['Embarked'].isnull()
df_full[mask_noembarked]
# Impute missing Embarked with most frequent value ('S')
df_full['Embarked'].fillna(df_full['Embarked'].mode()[0], inplace=True)
if show_graphs:
    sns.barplot(x='Embarked', y='Survived', data=df_full[mask_train], palette='colorblind')
# Check survival rate in function of group size.
df_ticket = df_full.loc[mask_train].groupby('Ticket', as_index=False)['Survived', 'Name'].agg({'Survived': 'mean', 'Name': 'count'})
df_ticket = df_ticket.groupby('Name', as_index=False)['Survived'].mean()
df_ticket = df_ticket.sort_values(by='Survived')
df_ticket
df_ticket = df_full.groupby('Ticket')['Name'].count()
df_full['TicketSize'] = df_full['Ticket'].map(df_ticket)
print('Correlation: ', df_full[['TicketSize', 'FamilySize']].corr().values[0, 1])
(df_full['TicketSize'] - df_full['FamilySize']).hist(bins=20)
df_full['TicketSizeBin'] = pd.cut(df_full['TicketSize'], [0, 1, 4, 20], labels=["alone", "normal", "big"])
catgroup(df_full.loc[mask_train], 'TicketSizeBin', 'Survived')
df_full['TicketSizeBin'] = df_full['TicketSizeBin'].map({'alone': 0, 'normal': 1, 'big': 2})
df_full['IsAloneT'] = (df_full['TicketSizeBin'] == 0).astype(int)
catgroup(df_full.loc[mask_train], 'IsAloneT', 'Survived')
fare_scaler = 'TicketSize'
df_full['Fare'] = df_full['Fare'] / df_full[fare_scaler]
mask_zerofare = df_full['Fare'] == 0
df_full.loc[mask_zerofare]
df_full['Fare'].fillna(df_full['Fare'].median(), inplace=True)
df_full['FareOrig'] = df_full['Fare'] * df_full[fare_scaler] 
df_full['FareBin'] = pd.qcut(df_full['Fare'], 4, labels=False).astype(int)
# Check survival rate in function of group size.
df_familygroup = df_full.loc[mask_train].groupby(['LastName', 'FareOrig'], as_index=False)['Survived', 'Name'].agg({'Survived': 'mean', 'Name': 'count'})
df_familygroup = df_familygroup.groupby('Name', as_index=False)['Survived'].mean()
df_familygroup = df_familygroup.sort_values(by='Survived')
df_familygroup
df_familygroup = df_full.groupby(['LastName', 'FareOrig'])['Name'].count()
df_full['NameFareSize'] = df_full[['LastName', 'FareOrig']].apply(lambda row: df_familygroup[(row[0], row[1])], axis=1)
df_corr = df_full[['NameFareSize', 'TicketSize', 'FamilySize']].corr()
print("Correlation Familysize - NameFaresize = ", df_corr.loc['FamilySize', 'NameFareSize'])
print("Correlation Ticketsize - NameFaresize = ", df_corr.loc['TicketSize', 'NameFareSize'])
(df_full['NameFareSize'] - df_full['FamilySize']).hist(bins=20)
df_full['Group'] = ''
df_groups = df_full.groupby(['LastName', 'FareOrig'])

for group, df_group in df_groups:    
    for idx, row in df_group.iterrows():
        group_members = df_group.drop(idx)['PassengerId'].tolist()
        df_full.at[idx, 'Group'] = group_members
df_groups = df_full.groupby('Ticket')

for group, df_group in df_groups:    
    for idx, row in df_group.iterrows():
        group_members = df_group.drop(idx)['PassengerId'].tolist()
        df_full.at[idx, 'Group'].extend(group_members)
df_full['Group'] = df_full['Group'].map(set)
def group_survived(group):
    mask_group = df_full['PassengerId'].isin(group)
    s = df_full.loc[mask_group, 'Survived'].max()
    return s if pd.notnull(s) else 0.5 

df_full['GroupSurvived'] = df_full['Group'].apply(group_survived)
df_full['GroupSize'] = df_full['Group'].str.len() + 1
# Check survival rate in function of group size.
df_fullgroup = df_full.loc[mask_train].groupby('GroupSize', as_index=False)['Survived'].mean()
df_fullgroup = df_fullgroup.sort_values(by='Survived')
df_fullgroup
df_corr = df_full[['NameFareSize', 'TicketSize', 'FamilySize', 'GroupSize']].corr()
print("Correlation Familysize - GroupSize = ", df_corr.loc['FamilySize', 'GroupSize'])
print("Correlation Ticketsize - Groupsize = ", df_corr.loc['TicketSize', 'GroupSize'])
print("Correlation NameFareSize - Groupsize = ", df_corr.loc['NameFareSize', 'GroupSize'])
(df_full['GroupSize'] - df_full['TicketSize']).hist(bins=20)
list_drop_features = [name for name, include in used_features.items() if not include]

df_full.drop(columns=list_drop_features, inplace=True)
df_full.loc[mask_train].corr()
#sns.heatmap(df_full[mask_train], annot=True)
base_columns = ['Survived', 'Data']
data_columns = [col for col in df_full.columns if col not in base_columns]

scaler = StandardScaler()
df_full.loc[mask_train, data_columns] = scaler.fit_transform(df_full.loc[mask_train, data_columns])
df_full.loc[mask_valid, data_columns] = scaler.transform(df_full.loc[mask_valid, data_columns])
Xt = df_full.loc[mask_train].drop(columns=base_columns)
Xv = df_full.loc[mask_valid].drop(columns=base_columns)
# Logistic Regression LASSO
if feature_selection:
    threshold_pct = 0.1
    lasso = LogisticRegression(penalty='l1', C=10, random_state=0, solver='saga', max_iter=200)
    lasso.fit(Xt, yt)
    print("Lasso accuracy on training data: ", lasso.score(Xt, yt))
    
    # Select features
    coefs = np.absolute(lasso.coef_.flatten())
    plt.hist(coefs, bins=20)
    mask_features = coefs > (np.max(coefs) * threshold_pct)
    new_columns = Xt.columns[mask_features]
    df_features = pd.DataFrame({'Features': new_columns, 'Strength': coefs[mask_features]}).sort_values(by='Strength', ascending=False)
    print(df_features)
    
    # Drop features with low importance
    Xt = Xt[new_columns]
    Xv = Xv[new_columns]
# Models
models = {}
models['Logistic Regression'] = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='saga', max_iter=300)
models['SVC_rbf'] = SVC(probability=True, kernel='rbf', gamma='scale', random_state=0)
models['SVC_lin'] = SVC(probability=True, kernel='linear', random_state=0)
models['KNN'] = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski', metric_params=None, 
                                     n_jobs=1, n_neighbors=10, p=3, weights='uniform')
dtree_params = {'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'min_impurity_decrease': 0.0, 
              'min_samples_leaf': 0.01, 'min_samples_split': 0.01, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}
models['Decision Tree'] = DecisionTreeClassifier(**dtree_params)
models['Random Forest'] = RandomForestClassifier(criterion='entropy', n_estimators=200, oob_score=True)
xgb_params = {'subsample': 0.5, 'reg_lambda': 5, 'reg_alpha': 0, 'n_estimators': 200, 'min_child_weight': 0, 'max_depth': 6, 
              'max_delta_step': 1, 'learning_rate': 1.0, 'gamma': 2, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5}
models['XGBoost'] = XGBClassifier(objective='binary:logistic', **xgb_params)
# Run CV using randomized folds (these can overlap).
scoring = ['accuracy']  # We can give multiple metrics here 
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

for mname, model in models.items():
    result = cross_validate(model, Xt, yt, scoring=scoring, cv=cv, return_train_score=False)
    print("CV results for model {}: mean {:2.4f}, std {:2.4f}".format(mname, np.mean(result['test_accuracy']), np.std(result['test_accuracy'])))
if model_tuning:
    param_grid_XGBoost = {
        'colsample_bytree': [0.1, 0.5, 1],
        'colsample_bylevel': [0.1, 0.5, 1],
        'subsample': [0.1, 0.5, 1], 
        'learning_rate': [0.05, 0.1, 0.3, 1.0],
        'max_depth': [0, 3, 6, 10], 
        'reg_alpha': [0, 0.1, 1, 5],
        'reg_lambda': [0, 0.1, 1, 5],
        'gamma': [0, 1, 2, 5], 
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [0, 1, 2, 5],
        'max_delta_step': [0, 1, 2, 5],
    }

    param_grid_DTC = {
        'criterion': ['gini', 'entropy'], 
        'splitter': ['best', 'random'], 
        'max_depth': [None, 3, 5, 7, 10], 
        'min_samples_split': [2, 0.01, 0.05, 0.1],
        'min_samples_leaf': [1, 0.01, 0.05, 0.1],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'max_features': [None, 'auto'], 
        'min_impurity_decrease': [0.0, 0.2, 0.4, 0.7],
    }
    #tune_model = RandomizedSearchCV(models['XGBoost'], param_distributions=param_grid_XGBoost, scoring='roc_auc', cv=cv, n_iter=1000)
    tune_model = GridSearchCV(models['Decision Tree'], param_grid=param_grid_DTC, scoring='roc_auc', cv=cv)
    tune_model.fit(Xt, yt)
    print('Best parameters:\n', tune_model.best_params_)
list_submit = models.keys()
dict_submissions = {}

# Loop over models
for mname in list_submit:
    model = models[mname]
    model.fit(Xt, yt)

    # Check train data score
    ytp = model.predict(Xt)
    acc = model.score(Xt, yt)
    #print("Accuracy of model ", mname, " on train data: ", acc)

    # Generate validation data score
    yvp = model.predict(Xv)
    dict_submissions[mname] = yvp
    submission = pd.DataFrame({"PassengerId": df_valid["PassengerId"], "Survived": yvp})
    submission.to_csv('submission_{}.csv'.format(mname), index=False)