# Basic Libraries
import numpy as np 
import pandas as pd 

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# Visaulization
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier (machine learning algorithm) 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict

# Parameter Tuning
from sklearn.model_selection import GridSearchCV

# Settings
pd.options.mode.chained_assignment = None # Stop warning when use inplace=True of fillna
train_set =  pd.read_csv('../input/train.csv')
test_set =  pd.read_csv('../input/test.csv')
train_set.head()
test_set.head()
len(train_set)
train_set.describe()
train_set.isnull().sum()
len(test_set)
test_set.describe()
test_set.isnull().sum()
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
train_set.describe()
test_set.describe()
combined_set = [train_set, test_set] # combined 2 datasets for more efficient processing

for dataset in combined_set:
    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)

train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)
train_set.isnull().sum()
test_set.isnull().sum()
age_bins = [0,15,35,45,60,200]
age_labels = ['15-','15-35','35-45','40-60','60+']
fare_bins = [0,10,30,60,999999]
fare_labels = ['10-','10-30','30-60','60+']

def get_title(dataset, feature_name):
    return dataset[feature_name].map(lambda name:name.split(',')[1].split('.')[0].strip())

for dataset in combined_set:
    dataset['AgeRange'] = pd.cut(dataset['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    dataset['FareRange'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
    dataset['FamilySize'] = dataset['SibSp'] + train_set['Parch']
    dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int) # NaN Cabins will become 0, otherwise 1
    dataset['Title'] = get_title(dataset, 'Name')
cat_plot(train_set, 'AgeRange','Survived', survival_palette)
cat_plot(train_set, 'FareRange','Survived', survival_palette)
cat_plot(train_set, 'FamilySize','Survived', survival_palette)
cat_plot(train_set, 'HasCabin','Survived', survival_palette)
fig, axis = plt.subplots(1,1,figsize=(12,5))
sns.countplot(x='Title', hue='Survived', data=train_set,
                  palette=survival_palette,ax=axis)
plt.show()

print(train_set['Title'].value_counts())
for dataset in combined_set:
    dataset['Family'] = ''
    dataset.loc[dataset['FamilySize'] == 0, 'Family'] = 'alone'
    dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] <= 3), 'Family'] = 'small'
    dataset.loc[(dataset['FamilySize'] > 3) & (dataset['FamilySize'] <= 6), 'Family'] = 'medium'
    dataset.loc[dataset['FamilySize'] > 6, 'Family'] = 'large'
cat_plot(train_set, 'Family','Survived', survival_palette)
title_dict = {
                "Mr" :        "Mr",
                "Miss" :      "Miss",
                "Mrs" :       "Mrs",
                "Master" :    "Master",
                "Dr":         "Scholar",
                "Rev":        "Religious",
                "Col":        "Officer",
                "Major":      "Officer",
                "Mlle":       "Miss",
                "Don":        "Noble",
                "the Countess":"Noble",
                "Ms":         "Mrs",
                "Mme":        "Mrs",
                "Capt":       "Noble",
                "Lady" :      "Noble",
                "Sir" :       "Noble",
                "Jonkheer":   "Noble"
            }

for dataset in combined_set:
    dataset['TitleGroup'] = dataset.Title.map(title_dict)
print(test_set[test_set['TitleGroup'].isnull() == True])
test_set.at[414, 'TitleGroup'] = 'Noble' # A record with Dona title
fig, axis = plt.subplots(1,1,figsize=(12,5))
sns.countplot(x='TitleGroup', hue='Survived', data=train_set,
              palette=survival_palette,ax=axis)
axis.set_ylim(0, 200)
plt.show()
X_train = train_set.drop(['Survived','PassengerId','Name','Age','Fare','Ticket','Cabin','SibSp','Parch','Title','FamilySize'], axis=1)
X_test = test_set.drop(['PassengerId','Name','Age','Fare','Ticket','Cabin','SibSp','Parch','Title','FamilySize'], axis=1)

y_train = train_set['Survived']  # Relocate Survived target feature to y_train
X_train_analysis = X_train.copy()
X_train_analysis['Sex'] = X_train_analysis['Sex'].map({'male': 0, 'female': 1}).astype(int)
X_train_analysis['Embarked'] = X_train_analysis['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
X_train_analysis['Family'] = X_train_analysis['Family'].map({'alone': 0, 'small': 1, 'medium': 2, 'large': 3}).astype(int)

agerange_dict = dict(zip(age_labels, list(range(len(age_labels)))))
X_train_analysis['AgeRange'] = X_train_analysis['AgeRange'].map(agerange_dict).astype(int)

farerange_dict = dict(zip(fare_labels, list(range(len(fare_labels)))))
X_train_analysis['FareRange'] = X_train_analysis['FareRange'].map(farerange_dict).astype(int)

titlegroup_labels = list(set(title_dict.values()))
titlegroup_dict = dict(zip(titlegroup_labels, list(range(len(titlegroup_labels)))))
X_train_analysis['TitleGroup'] = X_train_analysis['TitleGroup'].map(titlegroup_dict).astype(int)
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation between Features', y=1.05, size = 15)
sns.heatmap(X_train_analysis.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
rforest_checker = RandomForestClassifier(random_state = 0)
rforest_checker.fit(X_train_analysis, y_train)
importances_df = pd.DataFrame(rforest_checker.feature_importances_, columns=['Feature_Importance'],
                              index=X_train_analysis.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)
my_imp_dict = {'Feature Importance' : pd.Series([0.360313, 0.113686, 0.109495, 0.103845, 0.100966, 0.099818, 0.056429, 0.055449],
             index=['TitleGroup', 'Family', 'Pclass', 'Sex','FareRange', 'AgeRange', 'HasCabin', 'Embarked'])}
my_imp_df = pd.DataFrame(my_imp_dict)
print(my_imp_df)
X_train = X_train.drop(['HasCabin','Embarked'], axis=1)
X_test = X_test.drop(['HasCabin','Embarked'], axis=1)
# TitleGroup, Family, Pclass, Sex, FareRange, AgeRange
X_train = pd.get_dummies(X_train, columns=['TitleGroup','Family','Pclass','Sex','AgeRange','FareRange'])
X_test = pd.get_dummies(X_test, columns=['TitleGroup','Family','Pclass','Sex','AgeRange','FareRange'])
X_train = X_train.drop(['Pclass_1','Sex_female','TitleGroup_Master','AgeRange_15-','FareRange_10-','Family_alone'], axis=1)
X_test = X_test.drop(['Pclass_1','Sex_female','TitleGroup_Master','AgeRange_15-','FareRange_10-','Family_alone'], axis=1)
X_train.head()
# Train the Classifier
## classifier = LogisticRegression()
## classifier.fit(X_train, y_train)

# Predict the Test Data
## y_pred = classifier.predict(X_test)

# Submit Prediction Result
## passengerId = test_set['PassengerId']
## submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
## submission.to_csv('submission.csv', index=False)
## classifier = LogisticRegression()

# First we need to train the classifier as usual
## classifier.fit(X_train, y_train)

# estimator = the classifier algorithm to use, cv = number of cross validation split
## acc_logreg = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# You can check the accuracy score for each split. In this case, 10 accuracy scores
## print(acc_logreg)

# Get mean of accuracy score of all cross validations
## acc_logreg.mean() 

# Standard deviation = differences of the accuracy score in each cross validations. the less = less variance = the better
## acc_logreg.std() 
params_logreg = [{'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1','l2']}]
grid_logreg = GridSearchCV(estimator = LogisticRegression(),
                           param_grid = params_logreg,
                           scoring = 'accuracy',
                           cv = 10)
grid_logreg = grid_logreg.fit(X_train, y_train)
best_acc_logreg = grid_logreg.best_score_
best_params_logreg = grid_logreg.best_params_
"""
X_train_norm = X_train.copy()
X_train_norm = X_train_norm.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
X_train_norm.head()
"""
params_ksvm = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
               {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]},
               {'C': [0.1, 1, 10, 100], 'kernel': ['poly'],
                'degree': [1, 2, 3],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grid_ksvm = GridSearchCV(estimator = SVC(random_state = 0),
                         param_grid = params_ksvm,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs=-1)
grid_ksvm = grid_ksvm.fit(X_train, y_train)  # Replace X_train with X_train_norm here if you need
best_acc_ksvm = grid_ksvm.best_score_
best_params_ksvm = grid_ksvm.best_params_
params_dtree = [{'min_samples_split': [5, 10, 15, 20],
                 'min_samples_leaf': [1, 2, 3],
                 'max_features': ['auto', 'log2']}]
grid_dtree = GridSearchCV(estimator = DecisionTreeClassifier(criterion = 'gini', 
                                                             random_state = 0),
                            param_grid = params_dtree,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_dtree = grid_dtree.fit(X_train, y_train)
best_acc_dtree = grid_dtree.best_score_
best_params_dtree = grid_dtree.best_params_
params_rforest = [{'n_estimators': [200, 300],
                   'max_depth': [5, 7, 10],
                   'min_samples_split': [2, 4]}]
grid_rforest = GridSearchCV(estimator = RandomForestClassifier(criterion = 'gini', 
                                                               random_state = 0, n_jobs=-1),
                            param_grid = params_rforest,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_rforest = grid_rforest.fit(X_train, y_train)
best_acc_rforest = grid_rforest.best_score_
best_params_rforest = grid_rforest.best_params_
""" params_rforest = [{'n_estimators': [100, 200, 500, 800], 
                   'min_samples_split': [5, 10, 15, 20],
                   'min_samples_leaf': [1, 2, 3],
                   'max_features': ['auto', 'log2']}] """
grid_score_dict = {'Best Score': [best_acc_logreg,best_acc_ksvm,best_acc_dtree,best_acc_rforest],
                   'Optimized Parameters': [best_params_logreg,best_params_ksvm,best_params_dtree,best_params_rforest],
                  }
pd.DataFrame(grid_score_dict, index=['Logistic Regression','Kernel SVM','Decision Tree','Random Forest'])
best_params_dtree
best_params_rforest
logreg = LogisticRegression(C = 1, penalty = 'l1')
logreg.fit(X_train, y_train)
y_pred_train_logreg = cross_val_predict(logreg, X_train, y_train)
y_pred_test_logreg = logreg.predict(X_test)
ksvm = SVC(C = 1, gamma = 0.2, kernel = 'rbf', random_state = 0)
ksvm.fit(X_train, y_train)   # Replace X_train with X_train_norm here if you need
y_pred_train_ksvm = cross_val_predict(ksvm, X_train, y_train)
y_pred_test_ksvm = ksvm.predict(X_test)
dtree = DecisionTreeClassifier(criterion = 'gini', max_features='auto', min_samples_leaf=1, min_samples_split=5, random_state = 0)
dtree.fit(X_train, y_train)
y_pred_train_dtree = cross_val_predict(dtree, X_train, y_train)
y_pred_test_dtree = dtree.predict(X_test)
rforest = RandomForestClassifier(max_depth = 7, min_samples_split=4, n_estimators = 200, random_state = 0) # Grid Search best parameters
rforest.fit(X_train, y_train)
y_pred_train_rforest = cross_val_predict(rforest, X_train, y_train)
y_pred_test_rforest = rforest.predict(X_test)
second_layer_train = pd.DataFrame( {'Logistic Regression': y_pred_train_logreg.ravel(),
                                    'Kernel SVM': y_pred_train_ksvm.ravel(),
                                    'Decision Tree': y_pred_train_dtree.ravel(),
                                    'Random Forest': y_pred_train_rforest.ravel()
                                    } )
second_layer_train.head()

X_train_second = np.concatenate(( y_pred_train_logreg.reshape(-1, 1), y_pred_train_ksvm.reshape(-1, 1), 
                                  y_pred_train_dtree.reshape(-1, 1), y_pred_train_rforest.reshape(-1, 1)),
                                  axis=1)
X_test_second = np.concatenate(( y_pred_test_logreg.reshape(-1, 1), y_pred_test_ksvm.reshape(-1, 1), 
                                 y_pred_test_dtree.reshape(-1, 1), y_pred_test_rforest.reshape(-1, 1)),
                                 axis=1)

xgb = XGBClassifier(
        n_estimators= 800,
        max_depth= 4,
        min_child_weight= 2,
        gamma=0.9,                        
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(X_train_second, y_train)

y_pred = xgb.predict(X_test_second)
passengerId = np.array(test_set['PassengerId']).astype(int)
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
# Check if dataframe has 418 entries and 2 columns or not
print(submission.shape)

submission.to_csv('submission.csv', index=False)