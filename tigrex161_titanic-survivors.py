import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
    
    return combined
train_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
combined_set = get_combined_data()
combined_set.head()
corr_matrix = train_set.corr()
corr_matrix["Survived"].sort_values(ascending = False)
# Continuous Data Plot
def cont_plot(df, feature_name, target_name, palettemap, hue_order, feature_scale): 
    df['Counts'] = "" # A trick to skip using an axis (either x or y) on splitting violinplot
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    sns.distplot(df[feature_name], ax=axis0);
    sns.violinplot(x=feature_name, y="Counts", hue=target_name, hue_order=hue_order, data=df,
                   palette=palettemap, split=True, orient='h', ax=axis1)
    axis1.set_xticks(feature_scale)
    plt.show()
    # WARNING: This will leave Counts column in dataset if you continues to use this dataset

# Categorical/Ordinal Data Plot
def cat_plot(df, feature_name, target_name, palettemap): 
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
    sns.countplot(x=feature_name, hue=target_name, data=df,
                  palette=palettemap,ax=axis1)
    plt.show()

    
survival_palette = {0: "black", 1: "orange"} # Color map for visualization
cat_plot(train_set, 'Pclass','Survived', survival_palette)
cat_plot(train_set, 'Sex','Survived', survival_palette)
age_set_nonan = train_set[['Age','Survived']].copy().dropna(axis=0)
cont_plot(age_set_nonan, 'Age', 'Survived', survival_palette, [1, 0], range(0,100,10))
cat_plot(train_set, 'SibSp', 'Survived', survival_palette)
cat_plot(train_set, 'Parch', 'Survived', survival_palette)
fare_set = train_set[['Fare','Survived']].copy() # Copy dataframe so method won't leave Counts column in train_set
cont_plot(fare_set, 'Fare', 'Survived', survival_palette, [1, 0], range(0,550,50))
fare_set_mod = train_set[['Fare','Survived']].copy()
fare_set_mod['Counts'] = "" 
fig, axis = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='Fare', y="Counts", hue='Survived', hue_order=[1, 0], data=fare_set_mod,
               palette=survival_palette, split=True, orient='h', ax=axis)
axis.set_xticks(range(0,100,10))
axis.set_xlim(-20,100)
plt.show()

emb_set_nonan = train_set[['Embarked','Survived']].copy().dropna(axis=0)
cat_plot(train_set, 'Embarked','Survived', survival_palette)
from sklearn.preprocessing import LabelEncoder
encoder1 = LabelEncoder()
combined_gendercat = combined_set["Sex"].copy()
combined_gendercat_encoded = encoder1.fit_transform(combined_gendercat)
combined_set["Sex"] = combined_gendercat_encoded
combined_set.head()

combined_set['Embarked'].fillna(combined_set['Embarked'].mode()[0], inplace= True)
combined_set['Embarked'].isna().sum()

from sklearn.preprocessing import LabelBinarizer
encoder2 = LabelBinarizer()
combined_embark = combined_set['Embarked'].copy()
combined_embark_encoded = encoder2.fit_transform(combined_embark)
combined_set['Embarked'] = combined_embark_encoded
combined_set.head()
encoder2.classes_
def get_title(dataset, feature_name):
    return dataset[feature_name].map(lambda name:name.split(',')[1].split('.')[0].strip())
combined_set['Title'] = get_title(combined_set, 'Name')
combined_set.head()
title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Mr",
    "Lady" : "Royalty",
     "Dona" : "Mrs"
}
combined_set['TitleGroup'] = combined_set.Title.map(title_dict)
combined_set.head()
titles = combined_set['TitleGroup'].copy()
combined_set['Num_titles'] = encoder2.fit(titles)
combined_set['Num_titles'] = encoder2.transform(titles)
combined_set.head()
encoder2.classes_
combined_set.drop('TitleGroup', axis = 1, inplace = True)
combined_set.head()
combined_set.iloc[:891].Age.isnull().sum()
combined_set.iloc[891:].Age.isnull().sum()
combined_set["Age"] = combined_set.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
combined_set.head() 
combined_set.drop('Name', axis=1, inplace=True)

combined_set['Age'].loc[combined_set['Age'].isna()] = combined_set['Age'].median() 
combined_set['Age'].isna().sum()
combined_set['Fare'].loc[combined_set['Fare'].isna()]
combined_set['Fare'].loc[combined_set['Fare'].isna()] = combined_set['Fare'].mean()
encoder3 = LabelBinarizer()
combined_pclass = combined_set['Pclass'].copy()
combined_pclass_encoded = encoder3.fit_transform(combined_pclass)
combined_set['Pclass'] = combined_pclass_encoded 
encoder3.classes_
combined_set.head()
def process_family():
    
    global combined_set
    # introducing a new feature : the size of families (including the passenger)
    combined_set['FamilySize'] = combined_set['Parch'] + combined_set['SibSp'] + 1
    
    # introducing other features based on the family size
    combined_set['Singleton'] = combined_set['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined_set['SmallFamily'] = combined_set['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined_set['LargeFamily'] = combined_set['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return combined_set
combined_set = process_family()
combined_set.head()
train_cabin, test_cabin = set(), set()

for c in combined_set.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
        
for c in combined_set.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')
train_cabin

test_cabin
def process_cabin():
    global combined_set    
    # replacing missing cabins with U (for Uknown)
    combined_set['Cabin'].fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined_set['Cabin'] = combined_set['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined_set['Cabin'], prefix='Cabin')    
    combined_set = pd.concat([combined_set, cabin_dummies], axis=1)

    combined_set.drop('Cabin', axis=1, inplace=True)
    return combined_set
combined_set=process_cabin()
combined_set.drop('Ticket', axis=1, inplace = True)
combined_set.drop('Title', axis=1, inplace = True)
combined_set.head()
combined_set.shape
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

def recover_train_test_target():
    global combined
    
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = combined_set.iloc[:891]
    test = combined_set.iloc[891:]
    
    return train, test, targets
train, test, targets = recover_train_test_target()
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))


logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]
for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
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



output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('gridsearch_rf.csv', index=False)







































