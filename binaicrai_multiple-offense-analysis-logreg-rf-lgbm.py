import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import sklearn

from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score





from IPython.display import Image  

from sklearn.tree import export_graphviz
traindf = pd.read_csv('../input/novartis-data/Train.csv')

traindf.head()
testdf = pd.read_csv('../input/novartis-data/Test.csv')

testdf.head()
ss = pd.read_csv('../input/novartis-data/sample_submission.csv')

ss.head()
traindf.dtypes
testdf.dtypes
traindf['DATE'] =pd.to_datetime(traindf.DATE, format = '%d-%b-%y')

testdf['DATE'] = pd.to_datetime(testdf.DATE, format = '%d-%b-%y')

print('train set ' , traindf.DATE.dtypes)

print('test set ',testdf.DATE.dtypes)
# traindf['MULTIPLE_OFFENSE'] = pd.Categorical(traindf.MULTIPLE_OFFENSE)

# traindf['MULTIPLE_OFFENSE'].dtypes
# traindf.dtypes
NA_col = pd.DataFrame(traindf.isna().sum(), columns = ['NA_Count'])

NA_col['%_of_NA'] = (NA_col.NA_Count/len(traindf))*100

NA_col.sort_values(by = ['%_of_NA'], ascending = False, na_position = 'first')
NA_row = pd.DataFrame(traindf.isna().sum(axis=1), columns = ['NA_rw_count'])

NA_row['%_of_rw_NA'] = (NA_row.NA_rw_count/len(traindf))*100

NA_row.sort_values(by = ['%_of_rw_NA'], ascending = False, na_position = 'first').head(10)
NA_col = pd.DataFrame(testdf.isna().sum(), columns = ['NA_Count'])

NA_col['%_of_NA'] = (NA_col.NA_Count/len(testdf))*100

NA_col.sort_values(by = ['%_of_NA'], ascending = False, na_position = 'first')
NA_row = pd.DataFrame(testdf.isna().sum(axis=1), columns = ['NA_rw_count'])

NA_row['%_of_rw_NA'] = (NA_row.NA_rw_count/len(testdf))*100

NA_row.sort_values(by = ['%_of_rw_NA'], ascending = False, na_position = 'first').head()
cor = traindf[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7',

       'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15']]
plt.figure(figsize = (18,14))

sns.heatmap(cor.corr(), vmin=cor.values.min(), vmax=1, annot=True, annot_kws={"size":14}, square = False)

plt.show()
sns.catplot('MULTIPLE_OFFENSE', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['MULTIPLE_OFFENSE'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of MULTIPLE_OFFENSE', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_1', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_1'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_1', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_2', data= traindf, kind='count', alpha=0.7, height=4, aspect= 4)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_2'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_2', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_3', data= traindf, kind='count', alpha=0.7, height=4, aspect= 4)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_3'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_3', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_4', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_4'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_4', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_5', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_5'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_5', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_6', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_6'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_6', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_7', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_7'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_7', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_8', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_8'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_8', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_9', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_9'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_9', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_10', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_10'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_10', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_11', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3.8)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_11'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=10, color='blue',ha='center',  va='bottom', rotation = 90)

plt.title('Frequency plot of X_11', fontsize = 14, color = 'black')



plt.tick_params(axis='x', rotation = 90,  labelsize = 8)

plt.show()
sns.catplot('X_12', data= traindf, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_12'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_12', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_13', data= traindf, kind='count', alpha=0.7, height=4, aspect= 4)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_13'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_13', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_14', data= traindf, kind='count', alpha=0.7, height=4, aspect= 4)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_14'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_14', fontsize = 14, color = 'black')

plt.show()
sns.catplot('X_15', data= traindf, kind='count', alpha=0.7, height=4, aspect= 4)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = traindf['X_15'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of X_15', fontsize = 14, color = 'black')

plt.show()
train_X_10_12 = traindf[["DATE", "X_10","X_12","MULTIPLE_OFFENSE"]]

train_X_10_12.head()
null = train_X_10_12[train_X_10_12['X_12'].isna()]

null.DATE.min()
train_X_10_12 = train_X_10_12[train_X_10_12["DATE"].isin(pd.date_range("2016-07-01", "2030-01-01"))]

train_X_10_12 = train_X_10_12.set_index('DATE')

train_X_10_12.head()
train_X_10_12 = train_X_10_12 .assign(missing= np.nan)

train_X_10_12.missing[train_X_10_12['X_12'].isna()] = train_X_10_12.X_10

train_X_10_12.info()
train_X_10_12.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))
train_X_10_12.X_12.value_counts()
test_X_10_12 = testdf[["DATE", "X_10","X_12"]]

test_X_10_12.head()
null = test_X_10_12[test_X_10_12['X_12'].isna()]

null.DATE.min()
test_X_10_12 = test_X_10_12[test_X_10_12["DATE"].isin(pd.date_range("2016-07-01", "2030-01-01"))]

test_X_10_12 = test_X_10_12.set_index('DATE')

test_X_10_12.head()
test_X_10_12 = test_X_10_12 .assign(missing= np.nan)

test_X_10_12.missing[test_X_10_12['X_12'].isna()] = test_X_10_12.X_10

test_X_10_12.info()
test_X_10_12.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10))
test_X_10_12.X_12.value_counts()
traindf['X_12'] = traindf['X_12'].fillna(1)

traindf.isna().sum()
testdf['X_12'] = testdf['X_12'].fillna(1)

testdf.isna().sum()
for i in traindf.columns:

    if traindf[i].nunique() == 1:

        print('Train: With only 1 unique value: ', i)

    if traindf[i].nunique() == traindf.shape[0]:

        print('Train: With all unique value: ', i)



for i in testdf.columns:

    if testdf[i].nunique() == 1:

        print('Test: With only 1 unique value: ', i)

    if testdf[i].nunique() == testdf.shape[0]:

        print('Test: With all unique value: ', i)
train = traindf.copy().drop('INCIDENT_ID', axis = 1)

test = testdf.copy().drop('INCIDENT_ID', axis = 1)
train['Year'] = train['DATE'].dt.year

train['Month'] = train['DATE'].dt.month

train['Day'] = train['DATE'].dt.day_name()

train.head()
test['Year'] = test['DATE'].dt.year

test['Month'] = test['DATE'].dt.month

test['Day'] = test['DATE'].dt.day_name()

test.head()
train = train.drop('DATE', axis = 1)

test = test.drop('DATE', axis = 1)
col_train = train.columns

col_test = test.columns
# Changing variables with 10 or less unique values

l1 = []

for i in col_train:

    if train[i].nunique() <= 10:

        l1.append(i)

               

l1.remove('MULTIPLE_OFFENSE')
l2 = []

for i in col_test:

    if test[i].nunique() <= 10:

        l2.append(i)
# Checking the columns in train and test are same or not

df = pd.DataFrame(l1, columns = ['train'])

df['test'] = pd.DataFrame(l2)

df
train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)

test[l2] = test[l2].apply(lambda x: x.astype('category'), axis=0)

print('train dtypes:')

print(train[l1].dtypes)

print('======================================')

print('test dtypes:')

print(test[l1].dtypes)
cols = train.drop('MULTIPLE_OFFENSE', axis=1).columns

num_cols = train._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))

cat_cols
for usecol in cat_cols:

    train[usecol] = train[usecol].astype('str')

    test[usecol] = test[usecol].astype('str')

    

    #Fit LabelEncoder

    le = LabelEncoder().fit(

            np.unique(train[usecol].unique().tolist()+ test[usecol].unique().tolist()))



    #At the end 0 will be used for dropped values

    train[usecol] = le.transform(train[usecol])+1

    test[usecol]  = le.transform(test[usecol])+1

    

    train[usecol] = train[usecol].replace(np.nan, '').astype('int').astype('category')

    test[usecol]  = test[usecol].replace(np.nan, '').astype('int').astype('category')
train.MULTIPLE_OFFENSE.value_counts()
# Separating majority and minority classes

train_majority = train[train.MULTIPLE_OFFENSE==1]

train_minority = train[train.MULTIPLE_OFFENSE==0]
# Resampling the minority levels to match the majority level

# Upsample minority class

from sklearn.utils import resample

train_minority_upsampled = resample(train_minority, 

                                 replace=True,       # sample with replacement

                                 n_samples=22788,    # to match majority class

                                 random_state= 303)  # reproducible results

 

# Combine majority class with upsampled minority class

train_upsampled = pd.concat([train_majority, train_minority_upsampled])

 

# Display new class counts

train_upsampled.MULTIPLE_OFFENSE.value_counts()
# Resampling the majority levels to match the minority level

# Downsample majority class

from sklearn.utils import resample

train_majority_downsampled = resample(train_majority, 

                                 replace=True,       # sample with replacement

                                 n_samples=1068,    # to match minority class

                                 random_state= 303)  # reproducible results

 

# Combine majority class with upsampled minority class

train_downsampled = pd.concat([train_majority_downsampled, train_minority])

 

# Display new class counts

train_downsampled.MULTIPLE_OFFENSE.value_counts()
X = train.drop('MULTIPLE_OFFENSE', axis = 1)

y = train['MULTIPLE_OFFENSE']
Xs = train_upsampled.drop('MULTIPLE_OFFENSE', axis = 1)

ys = train_upsampled['MULTIPLE_OFFENSE']
Xsd = train_downsampled.drop('MULTIPLE_OFFENSE', axis = 1)

ysd = train_downsampled['MULTIPLE_OFFENSE']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, random_state = 42)
Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs, ys, test_size=0.3, random_state = 42)
Xsd_train, Xsd_val, ysd_train, ysd_val = train_test_split(Xsd, ysd, test_size=0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_val)
confusion_matrix(y_val, predictions)
recall_score(y_val, predictions)
test['MULTIPLE_OFFENSE'] = logreg.predict(test)
test.head()
test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission.head()
# submission.to_csv("submission_logreg.csv",index=False)
from sklearn.ensemble import RandomForestClassifier

rfc0 = RandomForestClassifier()

rfc0.fit(X = X_train,y = y_train)
pred_rf = rfc0.predict(X_val)
confusion_matrix(y_val, pred_rf)
recall_score(y_val, pred_rf)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)
test['MULTIPLE_OFFENSE'] = rfc0.predict(test)
test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rf = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rf.head()
# submission_rf.to_csv("submission_rf.csv",index=False)
rfc0_up = RandomForestClassifier()

rfc0_up.fit(X = Xs_train,y = ys_train)
pred_rf_up = rfc0_up.predict(Xs_val)

confusion_matrix(ys_val, pred_rf_up)
recall_score(ys_val, pred_rf_up)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = rfc0_up.predict(test)
test['INCIDENT_ID'] = testdf['INCIDENT_ID']

submission_rf_up = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rf_up.head()
# submission_rf_up.to_csv("submission_rf_up.csv",index=False)
rfcv0 = RandomForestClassifier(random_state=42)
param_grid = { 

    'n_estimators': [100, 150, 200, 250],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,9,10],

    'criterion' :['gini', 'entropy']

}
cv_rfc = GridSearchCV(estimator=rfcv0, param_grid=param_grid, cv= 5)

cv_rfc.fit(X_train, y_train)
cv_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=10, criterion='entropy')
rfc1.fit(X_train, y_train)
pred_rfgs = rfc1.predict(X_val)
confusion_matrix(y_val, pred_rfgs)
recall_score(y_val, pred_rfgs)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)
test['MULTIPLE_OFFENSE'] = rfc1.predict(test)
test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rfgs1 = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rfgs1.head()
# submission_rfgs1.to_csv("submission_rfgs.csv",index=False)
param_grid_2 = { 

    'n_estimators': [80,100,110,120,130],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [10,12,14,16],

    'criterion' :['gini']

}
cv_rfc2 = GridSearchCV(estimator=rfcv0, param_grid=param_grid_2, cv= 5)

cv_rfc2.fit(X_train, y_train)
cv_rfc2.best_params_
rfc2=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 80, max_depth=16, criterion='gini')
rfc2.fit(X_train, y_train)
pred_rfgs2 = rfc2.predict(X_val)
confusion_matrix(y_val, pred_rfgs2)
recall_score(y_val, pred_rfgs2)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)
test['MULTIPLE_OFFENSE'] = rfc2.predict(test)
test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rfgs2 = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rfgs2.head()
# submission_rfgs2.to_csv("submission_rfgs2.csv",index=False)
cv_rfc_up = GridSearchCV(estimator=rfcv0, param_grid=param_grid, cv= 5)

cv_rfc_up.fit(Xs_train, ys_train)
cv_rfc_up.best_params_
rfc_up1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=10, criterion='entropy')

rfc_up1.fit(Xs_train, ys_train)
pred_rfgs_up1 = rfc_up1.predict(Xs_val)

confusion_matrix(ys_val, pred_rfgs_up1)
recall_score(ys_val, pred_rfgs_up1)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = rfc_up1.predict(test)

test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rfgs_up1 = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rfgs_up1.head()
# submission_rfgs_up1.to_csv("submission_rfgs_up1.csv",index=False)
cv_rfc_up2 = GridSearchCV(estimator=rfcv0, param_grid=param_grid_2, cv= 5)

cv_rfc_up2.fit(Xs_train, ys_train)
cv_rfc_up2.best_params_
rfc_up2=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 120, max_depth=16, criterion='gini')

rfc_up2.fit(Xs_train, ys_train)
pred_rfgs_up2_train = rfc_up2.predict(Xs_train)

pred_rfgs_up2 = rfc_up2.predict(Xs_val)

confusion_matrix(ys_val, pred_rfgs_up2)
recall_score(ys_val, pred_rfgs_up2)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = rfc_up2.predict(test)

test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rfgs_up2 = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rfgs_up2.head()
# submission_rfgs_up2.to_csv("submission_rfgs_up2.csv",index=False)
param_grid_3 = { 

    'n_estimators': [80,100,110,120,130,140,150,180],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [8,10,12,14,16],

    'criterion' :['gini', 'entropy']

}
cv_rfc_down = GridSearchCV(estimator=rfcv0, param_grid=param_grid_3, cv= 5)

cv_rfc_down.fit(Xsd_train, ysd_train)
cv_rfc_down.best_params_
rfc_down=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 130, max_depth=12, criterion='gini')

rfc_down.fit(Xsd_train, ysd_train)
pred_rfgs_down = rfc_down.predict(Xsd_val)

confusion_matrix(ysd_val, pred_rfgs_down)
recall_score(ysd_val, pred_rfgs_down)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = rfc_down.predict(test)

test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_rfgs_down = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_rfgs_down.head()
# submission_rfgs_down.to_csv("submission_rfgs_down.csv",index=False)
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators = [('rf1', rfc_up2), ('rf2', rfc_down)], voting = 'hard')
model.fit(X_train, y_train)
ensemble_pred = model.predict(X_val)

confusion_matrix(y_val, ensemble_pred)
recall_score(y_val, ensemble_pred)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = model.predict(test)

test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_ensemble = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_ensemble.head()
# submission_ensemble.to_csv("submission_ensemble.csv",index=False)
!pip install lightgbm
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

clf = lgb.LGBMClassifier(silent=True, random_state = 333, metric='recall', n_jobs=4)
params ={'cat_smooth' : sp_randint(1, 100), 'min_data_per_group': sp_randint(1,1000), 'max_cat_threshold': sp_randint(1,100)}
fit_params={"eval_metric" : 'recall', 

            "eval_set" : [(Xs_train, ys_train),(Xs_val,ys_val)],

            'eval_names': ['train','valid'],

            'verbose': 200,

            'categorical_feature': 'auto'}
gs = RandomizedSearchCV( estimator=clf, param_distributions=params, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
gs.best_params_, gs.best_score_
clf2 = lgb.LGBMClassifier(**clf.get_params())

clf2.set_params(**gs.best_params_)
params_2 = {'learning_rate': [0.08, 0.09, 0.5, 0.85, 0.9],   

            'num_iterations': sp_randint(500,3000)}
gs2 = RandomizedSearchCV( estimator=clf2, param_distributions=params_2, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs2.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))
gs2.best_params_, gs2.best_score_
clf3 = lgb.LGBMClassifier(**clf2.get_params())

clf3.set_params(**gs2.best_params_)
params_3 = {'colsample_bytree': sp_uniform(loc=0.4, scale=0.6), 'num_leaves': sp_randint(500, 5000), 

            'min_child_samples': sp_randint(100,500), 'min_child_weight': [1e-2, 1e-1, 1, 1e1]}
gs3 = RandomizedSearchCV(estimator=clf3, param_distributions=params_3, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs3.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs3.best_score_, gs3.best_params_))
gs3.best_params_, gs3.best_score_
clf4 = lgb.LGBMClassifier(**clf3.get_params())

clf4.set_params(**gs3.best_params_)
params_4 = {'max_bin': sp_randint(100, 1500), 'max_depth': sp_randint(1, 15), 

            'min_data_in_leaf': sp_randint(500,3500)}
gs4 = RandomizedSearchCV(estimator=clf4, param_distributions=params_4, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs4.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs4.best_score_, gs4.best_params_))
gs4.best_params_, gs4.best_score_
clf5 = lgb.LGBMClassifier(**clf4.get_params())

clf5.set_params(**gs4.best_params_)
params_5 = {'reg_lambda': sp_randint(1, 30), 'boosting': ['goss', 'dart']}
gs5 = RandomizedSearchCV(estimator=clf5, param_distributions=params_5, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs5.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs5.best_score_, gs5.best_params_))
gs5.best_params_, gs5.best_score_
clf6 = lgb.LGBMClassifier(**clf5.get_params())

clf6.set_params(**gs5.best_params_)
params_6 = {'bagging_fraction': [0.2, 0.4, 0.6, 0.8, 1], 'feature_fraction': [0.2, 0.4, 0.6, 0.8, 1]}
gs6 = RandomizedSearchCV(estimator=clf6, param_distributions=params_6, scoring='recall',

                        cv=5, refit=True,random_state=333,verbose=True)
gs6.fit(Xs_train, ys_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs6.best_score_, gs6.best_params_))
gs6.best_params_, gs6.best_score_
clf_final = lgb.LGBMClassifier(**clf6.get_params())



clf_final.fit(Xs_train, ys_train, **fit_params)
feat_imp = pd.Series(clf_final.feature_importances_, index=train_upsampled.drop(['MULTIPLE_OFFENSE'], 

                                                                                axis=1).columns)

feat_imp.nlargest(20).plot(kind='barh', figsize=(8,10))
lgbm_pred = clf_final.predict(Xs_val, pred_contrib=False)

confusion_matrix(ys_val, lgbm_pred)
final_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, **gs4.best_params_, **gs5.best_params_, 

               **gs6.best_params_, 'scoring':'recall', 'metric':'recall', 'objective': 'binary'}

final_params
recall_score(ys_val, lgbm_pred)
test = test.drop(['MULTIPLE_OFFENSE', 'INCIDENT_ID'], axis = 1)

test['MULTIPLE_OFFENSE'] = clf_final.predict(test, pred_contrib=False)

test['INCIDENT_ID'] = testdf['INCIDENT_ID']
submission_lgbm = test[['INCIDENT_ID', 'MULTIPLE_OFFENSE']]

submission_lgbm.head()
# submission_lgbm.to_csv("submission_lgbm2.csv",index=False)