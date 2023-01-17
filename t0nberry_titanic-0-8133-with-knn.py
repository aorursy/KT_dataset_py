import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def plot_subplots(feature, data):
    fx, axes = plt.subplots(2,1,figsize=(15,10))
    axes[0].set_title(f"{feature} vs Frequency")
    axes[1].set_title(f"{feature} vs Survival")
    fig_title1 = sns.countplot(data = data, x=feature, ax=axes[0])
    fig_title2 = sns.countplot(data = data, x=feature, hue='Survived', ax=axes[1])
    
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

full = train.append(test, sort=False)

titanic = full.iloc[0:891,:]
full.shape
full.columns
full.isnull().sum()
full.Name.head()
full['Title'] = full['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
full.Title.sample(10)
full.Title.unique().tolist()
plot_subplots('Title', full)
full.loc[full.Title=='the']
full.iloc[759,-1] = "Countess"
full.iloc[759,:]
fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.boxplot(x='Title', y='Age', data=full)
full.groupby("Title").Age.describe()
full.groupby("Title").Survived.describe()
full.Title.value_counts()
Title_Dictionary = {
                    "Mme.":        "Mrs",
                    "Mlle.":       "Mrs",
                    "Ms.":         "Mrs",
                    "Mr." :        "Mr",
                    "Mrs." :       "Mrs",
                    "Miss." :      "Mrs",
                    "Master.":     "Master",
                    "Countess":    "Lady",
                    "Dona.":       "Lady",
                    "Lady.":       "Lady"
                    }
Mapped_titles = full.Title.map(Title_Dictionary)
Mapped_titles.fillna("Rare", inplace=True)
full['Titles_mapped'] = Mapped_titles
full.Titles_mapped.value_counts()
full.Titles_mapped.unique()
plot_subplots('Titles_mapped', full)
target_columns = []
target_columns.append('Titles_mapped')
Ticket = pd.DataFrame(full.Ticket)
Ticket.sample(10)
Ticket.Ticket.value_counts()
Ticket['Count'] = Ticket.groupby('Ticket')['Ticket'].transform('count')
Ticket.sample(10)
full['Ticket_Count'] = Ticket.Count
full.Ticket_Count.head()
plot_subplots('Ticket_Count', full)
target_columns.append('Ticket_Count')
full.isnull().sum()
cabin = pd.DataFrame()
cabin['Cabin'] = full.Cabin
cabin.Cabin.value_counts()
cabin.Cabin.fillna("U", inplace=True)
import re
def findLetter(string, group):
    return re.match(r"([A-Z]{1})(\d*)", str(string)).group(group)
re.match(r"([A-Z]{1})(\d*)", 'U').group(1)
cabin.Cabin.sample(10)
cabin['Cabin_Letter']  = cabin.Cabin.apply(lambda x: findLetter(x,1))   
cabin.sample(10)
cabin['Survived'] = full['Survived']
cabin.head(10)
plot_subplots('Cabin_Letter', cabin)
target_columns.append('Cabin_Letter')
full['Cabin_Letter'] = cabin.Cabin_Letter
full.drop(columns='Cabin', inplace=True)
family = pd.DataFrame()
family["FamilySize"] = full.SibSp + full.Parch + 1
family.sample(10)
family.describe()
family.FamilySize.value_counts()
family['Survived'] = full.Survived
plot_subplots('FamilySize', family)
target_columns.append('FamilySize')
full['FamilySize'] = family.FamilySize
full[full.Fare.isnull()]
fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.barplot(x='Titles_mapped', y='Fare', data=full, hue="Embarked")
Mr_S_Fare_Mean = full[(full['Titles_mapped']=='Mr') & (full['Embarked']=='S')]['Fare'].mean()
Mr_S_Fare_Mean
full.loc[full.PassengerId==1044,'Fare'] = Mr_S_Fare_Mean
target_columns.append('Fare')
target_columns
full.Fare.value_counts()
full.sort_values('Ticket').head(10)
target_columns.remove('Fare')
full['Fare_adjusted'] = full.Fare / full.Ticket_Count
target_columns.append('Fare_adjusted')
plot_subplots('Sex', full)
target_columns.append('Sex')
full.isnull().sum()
full[full.Age.isnull() == False].groupby(['Titles_mapped', 'Pclass']).describe()
fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.boxplot(x='Titles_mapped', y='Age', data=full, hue="Pclass")
full[full.Age.isnull()].groupby(['Titles_mapped', 'Pclass']).describe()
def get_Age_mean(title, pclass):
    return full.loc[(full.Age.isnull() == False) & (full.Titles_mapped==title) & (full.Pclass == pclass)].Age.mean()
get_Age_mean('Master', 3)
full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Master'), 'Age'] = get_Age_mean('Master', 3)
for pclass in [1,2,3]:
    full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Mr'), 'Age'] = get_Age_mean('Mr', pclass)
for pclass in [1,2,3]:
    full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Mrs'), 'Age'] = get_Age_mean('Mrs', pclass)
full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Rare'), 'Age'] = get_Age_mean('Rare', 1)
target_columns.append('Age')
full[full.Age.isnull() == False].groupby(['Titles_mapped', 'Pclass']).describe()
full.Embarked.value_counts()
full[full.Embarked.isnull()]
full.Embarked.mode()[0]
full.loc[full.Embarked.isnull(), 'Embarked'] = full.Embarked.mode()[0]
full.loc[full.Embarked.isnull()]
full.isnull().sum()
target_columns.append('Embarked')
target_columns
fullfinal = full[target_columns]
fullfinal['Pclass'] = full.Pclass
fullfinal.dtypes
full['Last_Name'] = full['Name'].apply(lambda x :str.split(x,',')[0])
full.Last_Name.sample(10)
DEFAULT_SURVIVAL_VALUE = 0.5
full['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in full[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'Age',]].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        #found Family group
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
print("Number of passengers with family survival information:", full.loc[full['Family_Survival']!=0.5].shape[0])
for _, grp_df in full.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
print("Number of passenger with family/group survival information: " 
      +str(full[full['Family_Survival']!=0.5].shape[0]))
full.Family_Survival.describe()
fullfinal['Family_Survival'] = full.Family_Survival
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = fullfinal.copy()
for i in range(0,X.shape[1]):
    if X.dtypes[i]=='object':
        X[X.columns[i]] = le.fit_transform(X[X.columns[i]])
X.head()
X['Survived'] = full.Survived
plot_correlation_map(X)
full_bins = fullfinal.copy()
sns.distplot(full_bins.Age)
sns.distplot(full_bins.Fare_adjusted)
full_bins['AgeBin'] = pd.qcut(full_bins['Age'], 5)
full_bins['FareBin'] = pd.qcut(full_bins['Fare_adjusted'], 6)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
full_bins['AgeBin_Code'] = label.fit_transform(full_bins.AgeBin)
full_bins['FareBin_Code'] = label.fit_transform(full_bins.FareBin)
full_bins['CabinBin_Code'] = label.fit_transform(full_bins.Cabin_Letter)
full_bins['EmbarkedBin_Code'] = label.fit_transform(full_bins.Embarked)
full_bins.head()
full_bin_final = full_bins.drop(columns=['Titles_mapped', 'Cabin_Letter','Fare_adjusted', 'Age', 'Embarked', 'AgeBin', 'FareBin'] )
full_bin_final.head()
full_bin_final.Sex = label.fit_transform(full_bin_final.Sex)
full_bin_final.head()
full_train = full_bin_final.copy()
full_train.describe()
full_train.columns
full_train.dtypes
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
def recover_train_test_target(df):
    global combined
    
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = df.iloc[:891]
    test = df.iloc[891:]
    
    return train, test, targets
train, test, targets = recover_train_test_target(full_train)
def checkFeatureImportance(dataset):
    train, test, targets = recover_train_test_target(dataset)
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    
    features.plot(kind='barh', figsize=(10, 10))
checkFeatureImportance(full_train)
full_train.columns
full_train2 = full_train.drop(columns=['EmbarkedBin_Code', 'CabinBin_Code', 'Ticket_Count'])
checkFeatureImportance(full_train2)
full_train2.head()
full_train2.describe()
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
train = std_scaler.fit_transform(train)
test = std_scaler.fit_transform(test)
# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
                 'max_depth' : [8,10,12],
                 'n_estimators': [45,47,48,50],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)
from sklearn.neighbors import KNeighborsClassifier 

# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
                 'n_neighbors' : [6,7,8,9,10,11,12,114,16,18,20,22],
                 'algorithm': ['auto'],
                 'weights': ['uniform', 'distance'],
                 'leaf_size': list(range(1,50,5)),
                 }
    KNN = KNeighborsClassifier()
    cross_validation = StratifiedKFold(n_splits=10)

    grid_search = GridSearchCV(KNN,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)
model
def to_Kaggle_csv(model, filename):
    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv(filename, index=False)
to_Kaggle_csv(model, 'Family_Survival_KNN_GridSearch.csv')
