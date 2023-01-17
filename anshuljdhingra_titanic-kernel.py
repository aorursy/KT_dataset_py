# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
train.head()


test = pd.read_csv('../input/test.csv')
test.head()
Passenger = train.iloc[:,0]
Passenger.head()
combined = [train,test]

train.describe()
train.info()
test.dtypes
train.isnull().sum()
test.isnull().sum()
train.groupby(['Cabin'])['Cabin'].count().sort_values(ascending = False).index[0]
def impute_cat(df_train,df_test,variable):
    most_frequent_category = train.groupby([variable])[variable].count().sort_values(ascending = False).index[0]
    df_train[variable].fillna(most_frequent_category, inplace = True)
    df_test[variable].fillna(most_frequent_category, inplace = True)
    
def impute_cat_missing(df_train,df_test,variable):
    df_train[variable].fillna('Missing', inplace = True)
    df_test[variable].fillna('Missing', inplace = True)
    
impute_cat_missing(train,test,'Cabin')
impute_cat(train,test,'Embarked')
def impute_num(train,test, variable):
    train[variable].fillna(train[variable].median(), inplace = True)
    test[variable].fillna(test[variable].median(), inplace = True)
for variable in ['Age', 'Fare']:
    impute_num(train,test,variable)
test.isnull().sum()
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
submission.isnull().sum()
train.dtypes != 'O'
train.columns

print('number of unique values for Age : ' , train.Age.nunique())
print('number of unique values for Fare : ' , train.Fare.nunique())
train.dtypes
train['Ticket'].head()
# lets convert SibSp and Parch into a single column
train.head()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test.SibSp + test.Parch +1
train['IsAlone'] = np.where(train.FamilySize >1, 0, 1)
train.head()
test['IsAlone'] = np.where(test.FamilySize >1, 0, 1)
train['Title'] = train.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]

test['Title'] = test.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]
train.Ticket.nunique()
train.info()
train['Ticket_Num'] = train.Ticket.apply( lambda s : s.split(' ')[-1])

train['Ticket_Num'] = np.where(train.Ticket_Num.str.isdigit(), train.Ticket_Num, np.nan)
train['Ticket_Cat'] = train.Ticket.apply(lambda s: s.split(' ')[0])
train['Ticket_Cat'] = np.where(train.Ticket_Cat.str.isdigit(),  np.nan, train['Ticket_Cat'])

train.head()
train.Ticket_Num.isnull().sum()
train.Ticket_Cat.isnull().sum()
import re
text = train.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
text = text.str.upper()
text.unique()
train.Ticket_Cat = train.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
train.Ticket_Cat = train.Ticket_Cat.str.upper()
test['Ticket_Num'] = test.Ticket.apply( lambda s : s.split(' ')[-1])
test['Ticket_Num'] = np.where(test.Ticket_Num.str.isdigit(), test.Ticket_Num, np.nan)
test['Ticket_Cat'] = test.Ticket.apply(lambda s: s.split(' ')[0])
test['Ticket_Cat'] = np.where(test.Ticket_Cat.str.isdigit(),  np.nan, test['Ticket_Cat'])
test.Ticket_Cat = test.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
test.Ticket_Cat = test.Ticket_Cat.str.upper()
train['Ticket_Cat'] = np.where(train['Ticket_Cat'] == 'NAN', np.nan, train['Ticket_Cat'])
test['Ticket_Cat'] = np.where(test['Ticket_Cat'] == 'NAN', np.nan, test['Ticket_Cat'])
train.Ticket_Cat.isnull().sum()
train['Cabin_Cat'] = train['Cabin'].str[0]
train['Cabin_Num'] = train['Cabin'].str.extract('(\d+)')
test['Cabin_Cat'] = test['Cabin'].str[0]
test['Cabin_Num'] = test['Cabin'].str.extract('(\d+)')
train.head()
test.head()
train['Cabin_Num'] = train['Cabin_Num'].astype('float')
def conv_to_float(data):
    for var in ['Ticket_Num',  'Cabin_Num']:
        data[var] = data[var].astype('float')
conv_to_float(train)
conv_to_float(test)
train['FareBin'] = pd.qcut(train.Fare,  5)
test['FareBin'] = pd.qcut(test.Fare, 5)
test['AgeBin']= pd.cut(test.Age.astype(int), 5)
train['AgeBin']= pd.cut(train.Age.astype(int), 5)
train.head()
train.drop(labels= ['PassengerId', 'Name', 'Cabin','Ticket'], inplace= True, axis =1)
train.head()
test.drop(labels= ['PassengerId', 'Name', 'Cabin','Ticket'], inplace= True, axis =1)
def impute_cat_missing(df_train,df_test,variable):
    df_train[variable].fillna('Missing', inplace = True)
    df_test[variable].fillna('Missing', inplace = True)
  
impute_num(train,test,'Cabin_Num')
impute_num(train,test,'Ticket_Num')
impute_cat_missing(train,test,'Cabin_Cat')
impute_cat(train,test,'Ticket_Cat')
#finding outlier in numerical columns
 
for var in ['Age', 'Fare', 'FamilySize']:
    sns.boxplot(y = var, data = train)
    plt.show()
# all of these have outliers
# let's find out their distribution
for var in ['Age', 'Fare', 'FamilySize']:
    sns.distplot(train[var], bins = 30)
    plt.show()
train.columns
# all of these do not follow gaussian distribution , hence , we will follow inter-quartile range to find out outliers
def removing_outliers(data):
    for var in ['Age', 'Fare']:
        IQR = data[var].quantile(.75) - data[var].quantile(.25)
        upper_bound = round(data[var].quantile(.75) + (IQR*3))
        lower_bound = round(data[var].quantile(.25) - (IQR*3))
        print('Extreme outliers are values for {variable} < {lowerboundary} or > {upperboundary}'.format(variable=var, lowerboundary=lower_bound, upperboundary=upper_bound))
        print('-**************************-')
        print('Removing outlier values')
        data[var] = np.where(data[var] > upper_bound, upper_bound, data[var])

removing_outliers(train)
# let's check whether top-coding worked
for var in ['Age',  'Fare']:
    print(var, ' max value: ', train[var].max())
# removing outliers in categorical columns
for var in ['Title', 'Ticket_Cat', 'Embarked', 'Cabin_Cat']:
    print(var, train[var].value_counts()/np.float(len(train)))
    print()
train.dtypes
temp = train.groupby(['Cabin_Cat'])['Cabin_Cat'].count()/np.float(len(train))
frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
test.dtypes
   
frequent_cat
def rare_imputation(variable, which='rare'): 
    # find frequent labels
    temp = train.groupby([variable])[variable].count()/np.float(len(train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], mode_label)
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], mode_label)
            
    else:
        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], 'Rare')
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], 'Rare')
       
rare_imputation('Cabin_Cat', 'frequent')
rare_imputation('Ticket_Cat', 'rare')
rare_imputation('Title', 'frequent')

for var in ['Title', 'Ticket_Cat', 'Embarked', 'Cabin_Cat']:
    print(var, train[var].value_counts()/np.float(len(train)))
    print()
    print(train.dtypes)


print(train.head())
print(train.dtypes)

def conv_to_cat(data):
    for var in ['Sex', 'Embarked', 'Title', 'Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
        data[var] = data[var].astype('category')
conv_to_cat(train)
conv_to_cat(test)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
def label_encode(data):
    for var in ['Sex', 'Embarked', 'Title', 'Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
        data[var] = label.fit_transform(data[var])
           
train_label = train.copy()
test_label = test.copy()
label_encode(train_label)
train.head()
label_encode(test_label)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[2,7,10,12,13,15,16])
train_ohe= train_label.copy()
test_ohe = test_label.copy()
train_ohe = ohe.fit_transform(train_ohe).toarray()


ohe_test = OneHotEncoder(categorical_features=[1,6,9,11,12,14,15])
test_ohe = ohe_test.fit_transform(test_ohe).toarray()
train_ohe
test_ohe
train_label.head()

test_label.head()
train.head()
train.groupby(['Survived'], as_index=False).mean()

train_label.groupby(['Survived'], as_index=False).mean()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train['Title'], train['Survived'])
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Ticket_Cat', 'Survived']].groupby(['Ticket_Cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Cabin_Cat', 'Survived']].groupby(['Cabin_Cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Title', 'Survived']].groupby(['Title'], as_index=False).sum().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.columns
for var in ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', 'FamilySize', 'IsAlone', 'Title','Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
    print(train[[var, 'Survived']].groupby([var], as_index=False).sum().sort_values(by='Survived', ascending=False))
    print('*'*40)
plt.figure(figsize=(15,9))
i=1
colour = {1 : 'magenta', 2 : 'green', 3: 'orange'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    sns.boxplot(y=var, data=train, color= colour[i])
    plt.title(var)
    i=i+1
    

plt.figure(figsize=(15,6))
i=1
colour = {1 : 'magenta', 2 : 'green', 3: 'orange'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    sns.boxplot(x='Sex', y=var, data=train, color= colour[i], hue='Survived')
    plt.title(var)
    i=i+1
plt.figure(figsize=(15,6))
i=1
j=1
colour = {1 : 'pink', 2 : 'green', 3: 'white', 4: 'black', 5:'orange', 6:'blue'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    plt.hist(x=[train[train['Survived']==1][var],train[train['Survived']==0][var]], bins =20, stacked = True, color= [colour[j], colour[j+1]], label= ['Survived', 'Dead'])
    plt.title(var)
    plt.legend()
    i=i+1
    j=j+2
train.columns
plt.figure(figsize=(19,25))
i=1
for var in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin']:
    plt.subplot(5,2,i)
    sns.barplot(x=var, y='Survived', data=train)
    plt.title(var)
    i=i+1
      
plt.figure(figsize=(19,25))
i=1
for var in ['Pclass',  'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin']:
    plt.subplot(5,2,i)
    sns.barplot(x=var, y='Survived', data=train, hue ='Sex')
    plt.title(var)
    i=i+1
g = sns.FacetGrid(data= train, col='Survived', row='Pclass')
g.map(plt.hist, 'Age')
g.map(plt.legend)
train.columns
f = sns.FacetGrid(data=train, col = 'Embarked', row= 'Survived')
f.map(sns.barplot, 'Sex', 'Fare')
plt.scatter(x = train.Age, y=train.Fare)
sns.regplot(x='Age', y='Fare', data=train)
sns.kdeplot(train['Age'], train['Fare'])
train.columns
h = sns.FacetGrid(data=train, row='Embarked', col='Pclass')
h.map(sns.barplot, 'Survived', 'Age')
sns.pairplot(data=train)
sns.lmplot('Age', 'Fare', data=train)
train.head()
train.corr() >.8
train_label.corr() >.8
# feature Selection
# lets find out the constant features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0)
sel.fit(train_label)
train.head()
train_label.head()
sum(sel.get_support())
train_label.shape
# let's find the quasi constant features
sel_quasi = VarianceThreshold(threshold=0.01) # for 99% constant features
sel_quasi.fit(train_label)
sum(sel_quasi.get_support())
# hence, none of the columns are constant and quasi constant .
# let's find out the duplicated columns
dup_train_label = train_label.T
dup_train_label.head()
dup_train_label.duplicated().sum()
# so, this shows that none of the columns are duplicated
train_label.corr()
plt.figure(figsize=(15,8))
sns.heatmap(data = train_label.corr().abs(), cmap = 'magma_r')
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
correlation(train_label, 0.8)
len(set(correlation(train_label, 0.8)))
train_label.corr()

# {'AgeBin', 'FamilySize', 'FareBin'} are correlated features
train_label.drop(labels = ['AgeBin', 'FamilySize', 'FareBin'], axis=1, inplace=True)
train_label.head()
test_label.head()
test_label.drop(labels=['AgeBin', 'FamilySize', 'FareBin'], axis=1, inplace=True)
# using Univariate roc-auc or mse for feature selection
roc_values = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(train_label.drop(labels=['Survived'], axis=1), train_label['Survived'], test_size=0.25, random_state=1)
for var in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train.loc[:,var].to_frame(), y_train)
    y_score = clf.predict(X_test.loc[:,var].to_frame())
    roc_values.append(roc_auc_score(y_test,y_score))
roc_values
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values
roc_values[roc_values <= 0.5]
train_label.drop(labels = ['Ticket_Cat'], axis=1, inplace=True)
train_label.head()
test_label.drop(labels=['Ticket_Cat'], axis=1, inplace=True)
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
X_train,X_test,y_train,y_test = train_test_split(train_label.drop(labels=['Survived'], axis=1), train_label['Survived'], test_size=0.3, random_state=1)
from sklearn.ensemble import RandomForestClassifier

sfs = SFS(estimator=RandomForestClassifier(n_jobs=-1), k_features=12,forward=True, verbose=2, scoring='roc_auc', cv=10)
sfs = sfs.fit(np.array(X_train),y_train)
sfs_2 = SFS(estimator=RandomForestClassifier(n_jobs=-1, n_estimators=30), k_features=10,forward=True, verbose=2, scoring='roc_auc', cv=10)
sfs_2 = sfs_2.fit(np.array(X_train), y_train)
sfs_2.subsets_
selected_features = X_train.columns[list(sfs_2.k_feature_idx_)]
selected_features

X_train.columns
sfs_3 = SFS(estimator=RandomForestClassifier(n_jobs=-1, n_estimators=30), k_features=10, forward=False, verbose=2, floating=False, scoring='roc_auc', cv=10)
sfs_3.fit(np.array(X_train), y_train)
selected_features_backward = X_train.columns[list(sfs_3.k_feature_idx_)]
selected_features_backward
# let's make models on these and evaluate their performance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
logreg= LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
logreg.score(X_train,y_train)
from sklearn.metrics import confusion_matrix, roc_auc_score
roc_auc_score(y_test,y_pred)
train_label.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import roc_auc_score
logreg_score = logreg.score(X_train,y_train)
logreg_score
roc_auc_score(y_test,y_pred)
# let's find out the most important features using regression coefficients
logreg.coef_
score_dict = {}
roc_auc_score_dict = {}
logreg = LogisticRegression()
svc = SVC()
linear_svc = LinearSVC()
gaussian_nb = GaussianNB()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)
xg_boost = XGBClassifier()

alg = [LogisticRegression(),
SVC(),
LinearSVC(),
GaussianNB(),
DecisionTreeClassifier(),
RandomForestClassifier(n_estimators=100),
XGBClassifier()]

for est in alg:
    est_name = est.__class__.__name__
    print(est_name)
    model = est
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score_dict[est_name] = model.score(X_train,y_train)
    roc_auc_score_dict[est_name] = roc_auc_score(y_test,y_pred)

    
score_dict
roc_auc_score_dict
score_df = pd.DataFrame.from_dict(data=score_dict, orient='index' )
score_df.columns = ['Model Score']
score_df
roc_auc_score_df = pd.DataFrame.from_dict(data=roc_auc_score_dict, orient='index')
roc_auc_score_df.columns = ['ROC-AUC Score']
roc_auc_score_df
combined_score_df = pd.concat([score_df, roc_auc_score_df], axis=1)
combined_score_df
test_label.head()
# this states that XBG classifier performs best, though scores for Decision Tree and Random Forest are better, they are clearly overfitting
from sklearn.preprocessing import StandardScaler
sc_test = StandardScaler()
scaled_test = sc.fit_transform(test_label)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(scaled_test)
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": y_pred})
submission.head()
submission.to_csv('submission.csv', index=False)
