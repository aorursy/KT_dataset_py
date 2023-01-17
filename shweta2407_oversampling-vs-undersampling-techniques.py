# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, TomekLinks, OneSidedSelection, NeighbourhoodCleaningRule
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV,RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# READ DATA
train_data = pd.read_csv('../input/highly-unbalanced-multiclass6-dataset/spenddata.csv')
train_data = train_data[train_data.columns[1:]]
train_data.head()
plt.figure(figsize=(15,6))
sns.countplot(train_data.pov6, palette='Set2')
plt.show()
train_data.info()
train_data.pov6
# READ TEST DATA
test_data = pd.read_csv('../input/highly-unbalanced-multiclass6-dataset/testdata.csv')
test_data = test_data[test_data.columns[1:]]
test_data.info()
train_data.drop_duplicates()
# STORE ALL THE COLUMNS THAT HAS NULL VALUES 
train_null_columns = []

# ITERATE THROUGH TRAINING DATA COLUMNS AND CHECK WHETHER IT HAS NULL VALUES OR NOT
for index, rows in pd.DataFrame(train_data.isna().any()).iterrows():
    if rows[0] == True:
        # COLUMNS WITH NULL VALUES MORE THAN 11000
        if train_data[index].isna().sum() >= 11000:
            train_null_columns.append(index)
        
train_null_columns
# STORE ALL THE COLUMNS THAT HAS NULL VALUES 
test_null_columns = []

# ITERATE THROUGH TEST DATA COLUMNS AND CHECK WHETHER IT HAS NULL VALUES OR NOT
for index, rows in pd.DataFrame(test_data.isna().any()).iterrows():
    if rows[0] == True:
        # COLUMNS WITH NULL VALUES MORE THAN 11000
        if test_data[index].isna().sum() >= 4000:
            test_null_columns.append(index)
        
test_null_columns    
# COLUMNS THAT HAVE MOSTLY NULL VALUES IN TRAIN DATA BUT NOT IN TEST DATA
set(train_null_columns) - set(test_null_columns)
# COLUMNS THAT HAVE MOSTLY NULL VALUES TEST DATA BUT NOT IN TRAIN DATA
set(test_null_columns) - set(train_null_columns)
test_data['totshopping.rep'].isna().sum()
train_data['totshopping.rep'].isna().sum()
# TOTAL NULL COLUMNS TO BE DELETED
tot_cols = train_null_columns + list(set(test_null_columns) - set(train_null_columns))

print('TOTAL COLUMNS TO BE DELETED', len(tot_cols))

# DELETE THE NULL VALUED COLUMNS FROM TRAINING DATA 
filtered_data = train_data.drop(tot_cols, axis=1)

print('COLUMNS BEFORE', len(train_data.columns))
print('COLUMNS AFTER REMOVAL', len(filtered_data.columns))
# DELETE THE NULL VALUED COLUMNS FROM TEST DATA 
filtered_test_data = test_data.copy()

for col in tot_cols:
    try:
        filtered_test_data.drop([col], axis=1, inplace=True)
    except:
        print('{} COLUMN not in TEST DATA '.format(col))

print()
print('COLUMNS BEFORE', len(test_data.columns))
print('COLUMNS AFTER REMOVAL', len(filtered_test_data.columns))
if set(filtered_data.columns.tolist()) == set(filtered_test_data.columns.tolist()):
    print('THEY BOTH HAVE SAME COLUMNS')
else:
    print(set(filtered_data.columns.tolist()) - set(filtered_test_data.columns.tolist()))
    print(set(filtered_test_data.columns.tolist()) - set(filtered_data.columns.tolist()))
# MAKE LABEL
label = pd.DataFrame(train_data.pov6)
# SELECT COLUMNS WITH OBJECT DATATYPE
filtered_data.columns[filtered_data.dtypes=='object']
filtered_data['var9'].head()
filtered_data['respondent.id'].head()
# ENCODE STRING COLUMNS 
pd.get_dummies(filtered_data.var9)
# CONCAT THE DUMMY DATA WITH THE TRAINING DATA
filtered_data = pd.concat([filtered_data, pd.get_dummies(filtered_data.var9)], axis=1 )

# REMOVE UNWANTED STRING COLUMN
filtered_data.drop(['respondent.id','var9'], axis=1, inplace=True)
len(filtered_data.columns)
# CONCAT THE DUMMY DATA WITH THE TEST DATA
filtered_test_data = pd.concat([filtered_test_data, pd.get_dummies(filtered_test_data.var9)], axis=1 )

# REMOVE UNWANTED STRING COLUMN
filtered_test_data.drop(['respondent.id','var9'], axis=1, inplace=True)
len(filtered_test_data.columns)
# BEFORE REMOVING NULL VALUES
filtered_data.head()
# FILL ALL THE NULL VALUES WITH MEAN
filtered_data.fillna(filtered_data.mean(), inplace=True)
filtered_data.head()
# FILL ALL THE NULL VALUES WITH MEAN
filtered_test_data.fillna(filtered_test_data.mean(), inplace=True)
filtered_test_data.head()
# INITIALISE SCALER
scaler = MinMaxScaler()

# CREATE DATAFRAME OF THE SCALED DATA
scaled_train_data = pd.DataFrame(scaler.fit_transform(filtered_data), index=filtered_data.index, columns=filtered_data.columns)
scaled_train_data.head()
# CREATE DATAFRAME OF THE SCALED DATA
scaled_test_data = pd.DataFrame(scaler.fit_transform(filtered_test_data), index=filtered_test_data.index, columns=filtered_test_data.columns)
scaled_test_data.head()
# PLOT DISTRIBUTION OF THE TARGET CLASSES
plt.figure(figsize=(15,6))
sns.countplot(label.pov6, palette='Set2')
plt.show()

# SUMMARIZE DISTRIBUTION
counter = Counter(label.pov6)
for k,v in counter.items():
    per = v / len(label) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# PERFORM OVERSAMPLING USING SMOTE ( SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE )
oversample = SMOTE()
X_over, y_over = oversample.fit_resample(scaled_train_data, label)

# PLOT THE NEW DATA
plt.figure(figsize=(10,6))
sns.countplot(y_over.pov6, palette='Wistia')
plt.show()

# SUMMARIZE DISTRIBUTION
counter = Counter(y_over.pov6)
for k,v in counter.items():
    per = v / len(y_over) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
X_under, y_under = undersample.fit_resample(scaled_train_data, label)

# PLOT THE NEW DATA
plt.figure(figsize=(10,6))
sns.countplot(y_under.pov6, palette='autumn')
plt.show()

# SUMMARIZE DISTRIBUTION
counter = Counter(y_under.pov6)
for k,v in counter.items():
    per = v / len(y_under) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# CREATE PIPELINE OF ALL THE POSSIBLE CLASSIFIERS
est =[]

est.append(('SVC', Pipeline([('SVC', SVC(gamma='scale', class_weight='balanced'))])))
est.append(('GradientBoosting', Pipeline([('GradientBoosting',GradientBoostingClassifier())])))
est.append(('AdaBoost', Pipeline([ ('AdaBoost', AdaBoostClassifier())])))
est.append(('ExtraTree', Pipeline([('ExtraTrees', ExtraTreesClassifier())])))
est.append(('RandomForest', Pipeline([('RandomForest', RandomForestClassifier())]))) 
est.append(('Bagging', Pipeline([('Bagging', BaggingClassifier())])))
est.append(('KNeighbors', Pipeline([('KNeighbors', KNeighborsClassifier())])))
est.append(('DecisionTree', Pipeline([('DecisionTree', DecisionTreeClassifier())])))
# est.append(('XGB', Pipeline([('XGB', XGBClassifier())])))
import warnings
warnings.filterwarnings(action='ignore')

seed = 4
splits = 5

# CALCULATE F1 SCORE WITH MORE WEIGHT ON RECALL
f_score = make_scorer(fbeta_score, beta=2, average='macro')
models_score =[]

for i in est:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    results = cross_val_score(i[1], X_under, y_under, cv=cv, scoring=f_score)
    models_score.append({i[0] : '{} +/- {}'.format(results.mean(), results.std())})
print("F1 SCORES OF DIFFERENT CLASSIFIERS")
models_score
model = GradientBoostingClassifier()
model.fit(X_under, y_under.pov6)
# PREDICT ON THE TEST DATA
predict_under = model.predict(scaled_test_data)
model_ = GradientBoostingClassifier()
model_.fit(X_over, y_over.pov6)
# PREDICT ON THE TEST DATA
predict_over = model_.predict(scaled_test_data)
l = [predict_under, predict_over, train_data.pov6]
pal = ["autumn",'Wistia','Set2']
titles =['UNDERSAMPLED', 'OVERSAMPLED', 'TRAINING']

plt.figure(figsize=(25,6))
plt.rc('font', size=15)
for i in range(3):
    plt.subplot(1,3, i+1)
    sns.countplot(l[i], palette=pal[i])
    plt.title(titles[i])
    plt.xlabel('CLASSES')
    plt.ylabel('COUNT')
plt.show()

# SUMMARIZE DISTRIBUTION
print('DISTRIBUTION OF PREDICTED UNDERSAMPLED CLASSES')
counter = Counter(predict_under)
for k,v in counter.items():
    per = v / len(predict_under) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    
print()
print('DISTRIBUTION OF PREDICTED OVERSAMPLED CLASSES')
counter = Counter(predict_over)
for k,v in counter.items():
    per = v / len(predict_over) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# NEAR MISS VERSION 2
undersample = NearMiss(version=2, n_neighbors=3)
# transform the dataset
X_near2, y_near2 = undersample.fit_resample(scaled_train_data, label)

# NEAR MISS VERSION 3
undersample = NearMiss(version=3, n_neighbors=3)
# transform the dataset
X_near3, y_near3 = undersample.fit_resample(scaled_train_data, label)

# CONDENSED NEAREST NEIGHBOUR
undersample = CondensedNearestNeighbour(n_neighbors=1)
# transform the dataset
X_cnn, y_cnn = undersample.fit_resample(scaled_train_data, label)

# TOMEK LINKS
undersample = TomekLinks()
# transform the dataset
X_tomek, y_tomek = undersample.fit_resample(scaled_train_data, label)

# EDITED NEAREST NEIGHBOURS
undersample = EditedNearestNeighbours(n_neighbors=3)
# transform the dataset
X_enn, y_enn = undersample.fit_resample(scaled_train_data, label)

# ONE SIDED SELECTION 
undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
# transform the dataset
X_oss, y_oss = undersample.fit_resample(scaled_train_data, label)

# NEIGHBOURHOOD CLEANING RULE
undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
# transform the dataset
X_ncr, y_ncr = undersample.fit_resample(scaled_train_data, label)
model = GradientBoostingClassifier()

# FIT ON NEAR MISS VERSION 2
model.fit(X_near2, y_near2)
# PREDICT ON THE TEST DATA
predict_near2 = model.predict(scaled_test_data)

# NEAR MISS VERSION 3
model.fit(X_near3, y_near3)
predict_near3 = model.predict(scaled_test_data)

# CONDENSED NEAREST NEIGHBOUR
model.fit(X_cnn, y_cnn)
predict_cnn = model.predict(scaled_test_data)

# TOMEK LINKS
model.fit(X_tomek, y_tomek)
predict_tomek = model.predict(scaled_test_data)

# EDITED NEAREST NEIGHBOURS
model.fit(X_enn, y_enn)
predict_enn = model.predict(scaled_test_data)

# ONE SIDED SELECTION
model.fit(X_oss, y_oss)
predict_oss = model.predict(scaled_test_data)

# NEIGHBOURHOOD CLEANING RULE
model.fit(X_ncr, y_ncr)
predict_ncr = model.predict(scaled_test_data)
pred_list = [predict_near2, predict_near3, predict_cnn, predict_tomek, predict_enn, predict_oss, predict_ncr]
pal = ['autumn','Blues','Reds_r', 'icefire', 'Wistia','Purples', 'Greens']
titles =['NEAR_2', 'NEAR_3', 'Condensed Nearest Neighbour', 'TOMEK_LINKS', 'Edited Nearest Neighbour', 'One Sided Selection', 'Neighborhood Cleaning Rule']

plt.figure(figsize=(25,25))
plt.rc('font', size=15)
for i in range(len(pred_list)):
    plt.subplot(3, 3, i+1)
    sns.countplot(pred_list[i], palette=pal[i])
    plt.title(titles[i])
    plt.xlabel('CLASSES')
    plt.ylabel('COUNT')
plt.show()
l = [predict_under, predict_near2]
pal = ['Set1','Set2']
titles =['NearMiss 1', 'NearMiss 2']

plt.figure(figsize=(25,6))
plt.rc('font', size=15)
for i in range(2):
    plt.subplot(1,2, i+1)
    sns.countplot(l[i], palette=pal[i])
    plt.title(titles[i])
    plt.xlabel('CLASSES')
    plt.ylabel('COUNT')
plt.show()

print('NearMiss Version 1')
counter = Counter(predict_under)
for k,v in counter.items():
    per = v / len(predict_under) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

print()
print()
print('NearMiss Version 2')
counter = Counter(predict_near2)
for k,v in counter.items():
    per = v / len(predict_near2) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
