#Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#Import data
print('Importing data...')
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
# Any results you write to the current directory are saved as output.
print('train shape:', train.shape)
print('test shape:',test.shape)
#Impute missing values (mean for numeric, mode for categorical)
for i in train.columns:
    if train[i].dtype == 'object':
      train[i] = train[i].fillna(train[i].mode().iloc[0])
    elif (train[i].dtype != 'object'):
      train[i] = train[i].fillna(np.mean(train[i]))


for i in test.columns:
    if test[i].dtype == 'object':
      test[i] = test[i].fillna(test[i].mode().iloc[0])
    elif (test[i].dtype != 'object'):
      test[i] = test[i].fillna(np.mean(test[i]))
    

print('Nulls in train set:', train.isnull().sum().sum())
print('Nulls in test set:', test.isnull().sum().sum())
## label encode categorical variables
for col in train.columns:
    if train[col].dtype == 'object':
      train[col] = train[col].astype('category')
      train[col] = train[col].cat.codes

for col in test.columns:
    if test[col].dtype == 'object':
      test[col] = test[col].astype('category')
      test[col] = test[col].cat.codes
## Creating a dummy y label and drop the target variable
train['set'] = 0
test['set'] = 1
train = train.drop('TARGET',axis=1)
## Use a sample set from both train and test and concatenate into a single dataframe
train_df = train.sample(5000, random_state=344)
test_df = test.sample(5000, random_state=433)

all_data = train_df.append(test_df)
y_label = all_data['set']
all_data = all_data.drop('set',axis=1)

#Make sure the new dataframe contains all the initial features
print('New dataframe shape:', all_data.shape)
## Find all the features with covariate shift. Print during the procedure and then save in array
model = RandomForestClassifier(n_estimators = 50, max_depth = 5, min_samples_leaf = 5)
feat_to_drop = []
for col in all_data.columns:
    score = cross_val_score(model,pd.DataFrame(all_data[col]),y_label,cv=2,scoring='roc_auc')
    if np.mean(score) > 0.8:
        feat_to_drop.append(col)
    print(col,np.mean(score))
#Print number of features with covariate shift
print('Number of features that display a covariate shift:', len(feat_to_drop))
