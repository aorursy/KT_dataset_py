# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Warning Libraries :
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/wns-inno/train_LZdllcl.csv')
test = pd.read_csv('../input/wns-inno/test_2umaH9m.csv')
test.head()
train.head()




# getting their shapes

print("Shape of train :", train.shape)
print("Shape of test :", test.shape)
train.describe()
train.info()
train.employee_id.value_counts()
train.department.value_counts()
train.region.value_counts()
train.education.value_counts()
train.gender.value_counts()   
train.recruitment_channel.value_counts()
train.no_of_trainings.value_counts()
print("Age varies from: ",train.age.min(),'yrs to',train.age.max(),'yrs')
train.previous_year_rating.value_counts()
print("The years of employment varies from: ",train.length_of_service.min(),'yrs to',train.length_of_service.max(),'yrs')
train['KPIs_met >80%'].value_counts()
train['awards_won?'].value_counts()
print("Average score ranges between: ",train.avg_training_score.min(),'to',train.avg_training_score.max())
train.is_promoted.value_counts()
sns.countplot(train['is_promoted'])
plt.show()
# Correlation matrix between numerical values

plt.figure(figsize=(10, 5))
cor = train.corr()
ax = sns.heatmap(cor,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
#A bar plot of is_promoted by Department

train.groupby(['is_promoted','department'])['employee_id'].count().plot(kind = 'bar')
plt.title('Comparing based on department and is_promoted')
plt.show()

#percentages of employess from each department who are promoted
d = train[train['is_promoted'] == 1].groupby('department').count()['employee_id']
print((d / d.sum()) * 100)
# Employees with which educational background have been promoted the most?

sns.countplot(x = 'education', hue = 'is_promoted', data = train)
plt.title('Comparing based on educational background and is_promoted')
plt.show()

#print percentages of  Below Secondary,Bachelor's & Master's & above who are promoted
train[train['is_promoted'] == 1]['education'].value_counts(normalize = True) * 100
# create subplot plot
fig, axes = plt.subplots(1, 2, figsize = (15, 5))

# violinplot plot
sns.violinplot(x = 'previous_year_rating', y = 'length_of_service', data = train, hue = 'is_promoted', split = True, ax = axes[0])
sns.violinplot(x = 'recruitment_channel', y = 'length_of_service', data = train, hue = 'is_promoted', split = True, ax = axes[1])
plt.show()
# Gender V/S is_promoted

sns.countplot(x = 'gender', hue = 'is_promoted', data = train)
plt.title('Gender V/S is_promoted')
plt.show()

#percentages of females vs. males who are promoted
train[train['is_promoted'] == 1]['gender'].value_counts(normalize = True) * 100
# recruitment_channel V/S is_promoted

sns.countplot(x = 'recruitment_channel', hue = 'is_promoted', data = train)
plt.title('recruitment_channel V/S is_promoted')
plt.show()

train[train['is_promoted'] == 1]['recruitment_channel'].value_counts(normalize = True) * 100
# create subplot plot
fig, axes = plt.subplots(1, 2, figsize = (15, 5))

# boxplot
sns.boxplot(x='previous_year_rating',y='avg_training_score',data=train,hue='is_promoted', ax = axes[0])
sns.boxplot(x='is_promoted',y='avg_training_score',data=train,hue='previous_year_rating', ax = axes[1])
plt.show()
# dependency of awards won on promotion

data = pd.crosstab(train['awards_won?'], train['is_promoted'])
data.plot(kind = 'bar', stacked = True, color = ['magenta', 'purple'])

plt.title('Dependency of Awards in determining Promotion')
plt.xlabel('Awards Won or Not')
plt.show()

train[train['is_promoted'] == 1]['awards_won?'].value_counts(normalize = True) * 100
#dependency of KPIs with Promotion

data = pd.crosstab(train['KPIs_met >80%'], train['is_promoted'])
data.plot(kind = 'bar', stacked = True, color = ['pink', 'darkred'])

plt.title('Dependency of KPIs in determining Promotion')
plt.xlabel('KPIs Met or Not')
plt.show()

print(train[train['is_promoted'] == 1]['KPIs_met >80%'].value_counts(normalize = True) * 100)
#Region and is_promoted

plt.figure(figsize=(30,12))
sns.countplot(x='region',data=train,hue='is_promoted')
plt.title('Region v/s is_promoted',fontsize = 20)
plt.show()
sns.lmplot(x='no_of_trainings',y='age',data=train,fit_reg=False,hue='is_promoted',markers=['x','o'])
plt.show()
# combining the data for data prep

test['is_promoted']=np.nan
train['data']='train'
test['data']='test'
test=test[train.columns]

combined = pd.concat([train,test], sort = False , ignore_index= True)
combined.head()
#Missing values

combined.isna().sum()
combined.education.value_counts()
#filling in education with the maximum value(mode)

combined.education.fillna("Bachelor's",inplace=True)
combined.education.value_counts()
combined.education.isna().sum()
combined.previous_year_rating.value_counts()
#filling in previous_year_rating with the median

combined.previous_year_rating.fillna(combined.previous_year_rating.median(),inplace=True)
combined.previous_year_rating.value_counts()
combined.isna().sum()
combined.info()
combined.head()
#Feature hashing region

unique_region = np.unique(combined[['region']])
print("Total unique regions:", len(unique_region))
print(unique_region)
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=6, input_type='string')
hashed_features = fh.fit_transform(combined['region'])
hashed_features = hashed_features.toarray()
hashed_features = pd.DataFrame(hashed_features)
print(hashed_features)
combined = pd.concat((combined, hashed_features),axis=1)
combined.head()
combined.drop('region',axis=1,inplace=True)
#encoding gender and education

combined['gender'] = combined['gender'].map( {'f': 0, 'm': 1} ).astype(int)  

combined['education'] = combined['education'].map( {'Below Secondary': 0, 'Bachelor\'s': 1, 'Master\'s & above': 2} ).astype(int)

combined.head()
#Frequency Encoding

# size of each category
encoding = combined.groupby('department').size()

# get frequency of each category
encoding = encoding/len(combined)

combined['department'] = combined.department.map(encoding)
#Frequency Encoding

# size of each category
encoding = combined.groupby('recruitment_channel').size()

# get frequency of each category
encoding = encoding/len(combined)

combined['recruitment_channel'] = combined.recruitment_channel.map(encoding)
combined.head()
combined['previous_year_rating'] = combined['previous_year_rating'].astype(int)
combined.head()
#splitting the data back into train and test as it was already provided

train = combined[combined['data']=='train']
train.drop(['data','employee_id'],axis=1,inplace=True)

test = combined[combined['data']=='test']
submit = test['employee_id']
test.drop(['is_promoted','data','employee_id'],axis=1,inplace=True)

del combined
print(train.shape)
print(test.shape)
#For submission

submission = pd.DataFrame()
submission['employee_id'] = submit
submission['is_promoted'] = np.nan
train["is_promoted"] = train["is_promoted"].astype(int)

y = train["is_promoted"]
X = train.drop(labels = ["is_promoted"],axis = 1)
#Oversampling of the Model

print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y== 0)))

from imblearn.over_sampling import SMOTE

x_sample, y_sample = SMOTE().fit_sample(X, y.values.ravel())

# checking the sizes of the sample data
print("Size of x-sample :", x_sample.shape)
print("Size of y-sample :", y_sample.shape)

print("\nAfter OverSampling, counts of label '1': {}".format(sum(y_sample == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_sample == 0))) 

X = pd.DataFrame(x_sample,columns=X.columns)
y = pd.DataFrame(y_sample)
# splitting x and y into train and validation sets
#train test split for model building
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

print("Shape of x_train: ", X_train.shape)
print("Shape of x_valid: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_valid: ", y_test.shape)
# standard scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
test = sc.transform(test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score, roc_auc_score,roc_curve,classification_report,f1_score
#Logistic Regression

lr = LogisticRegression(solver='liblinear',random_state=3)
lr.fit(X_train,y_train)

y_test_pred = lr.predict(X_test)

print('Accuracy_score:',accuracy_score(y_test,y_test_pred))
f1_score(y_test, y_test_pred, zero_division=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

dt = DecisionTreeClassifier()

params = {'max_depth': sp_randint(2,12),
          'min_samples_split': sp_randint(2,12),
          'min_samples_leaf' : sp_randint(1,12)
          }

rsearch = RandomizedSearchCV(dt, param_distributions= params, n_iter=200 , 
                             cv = 3,scoring='roc_auc',random_state=3,n_jobs=-1,
                            return_train_score=True)

rsearch.fit(X,y)
rsearch.best_params_
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 3)

dt = DecisionTreeClassifier(**rsearch.best_params_)
dt.fit(X_train,y_train)

y_train_pred = dt.predict(X_train)
y_train_prob = dt.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = dt.predict(X_test)
y_test_prob = dt.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
#Feature Importance

imp = pd.DataFrame(dt.feature_importances_,index=X.columns,columns=['importance'])
imp.sort_values(by='importance',ascending=False)
# Feature selection (Dropping insignificant features)

X_new = X[['department','recruitment_channel','no_of_trainings', 'age','previous_year_rating',
           'length_of_service','KPIs_met >80%','avg_training_score',0,1,2,3]]
y_new = y
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

dt = DecisionTreeClassifier()

params = {'max_depth': sp_randint(2,12),
          'min_samples_split': sp_randint(2,12),
          'min_samples_leaf' : sp_randint(1,12)
          }

rsearch = RandomizedSearchCV(dt, param_distributions= params, n_iter=200 , 
                             cv = 10,scoring='roc_auc',random_state=3,n_jobs=-1,
                            return_train_score=True)

rsearch.fit(X_new,y_new)
rsearch.best_params_
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X_new,y_new,test_size = 0.3, random_state = 3)

dt = DecisionTreeClassifier(**rsearch.best_params_)
dt.fit(X_train,y_train)

y_train_pred = dt.predict(X_train)
y_train_prob = dt.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = dt.predict(X_test)
y_test_prob = dt.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
dt = DecisionTreeClassifier(**rsearch.best_params_)

dt.fit(X_new,y_new)
#Feature Importance

imp = pd.DataFrame(dt.feature_importances_,index=X_new.columns,columns=['importance'])
imp.sort_values(by='importance',ascending=False)
# Feature selection (Dropping insignificant features)

X_new2 = X[['department','recruitment_channel', 'age','previous_year_rating','KPIs_met >80%','avg_training_score',0,1,3]]
y_new2 = y
#Hyperparameter Tuning

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

dt = DecisionTreeClassifier()

params = {'max_depth': sp_randint(2,12),
          'min_samples_split': sp_randint(2,12),
          'min_samples_leaf' : sp_randint(1,12)
          }

rsearch = RandomizedSearchCV(dt, param_distributions= params, n_iter=200 , 
                             cv = 10,scoring='roc_auc',random_state=3,n_jobs=-1,
                            return_train_score=True)

rsearch.fit(X_new2,y_new2)
rsearch.best_params_
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X_new2,y_new2,test_size = 0.3, random_state = 3)

dt = DecisionTreeClassifier(**rsearch.best_params_)
dt.fit(X_train,y_train)

y_train_pred = dt.predict(X_train)
y_train_prob = dt.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = dt.predict(X_test)
y_test_prob = dt.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
### RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_train_pred = rf.predict(X_train)
y_train_prob = rf.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = rf.predict(X_test)
y_test_prob = rf.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
#LightGBM

import lightgbm as lgb

lgbc = lgb.LGBMClassifier(random_state=3)

lgbc.fit(X_train,y_train)

y_train_pred = lgbc.predict(X_train)
y_train_prob = lgbc.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = lgbc.predict(X_test)
y_test_prob = lgbc.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
## catboost

from catboost import CatBoostClassifier

cat = CatBoostClassifier()

cat.fit(X_train,y_train)

y_train_pred = cat.predict(X_train)
y_train_prob = cat.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = cat.predict(X_test)
y_test_prob = cat.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
##XGBoost

from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train,y_train)

y_train_pred = xgb.predict(X_train)
y_train_prob = xgb.predict_proba(X_train)[:,1]

print(accuracy_score(y_train,y_train_pred))
print(roc_auc_score(y_train,y_train_prob))

y_test_pred = xgb.predict(X_test)
y_test_prob = xgb.predict_proba(X_test)[:,1]

print('\n')
print(accuracy_score(y_test,y_test_pred))
print(roc_auc_score(y_test,y_test_prob))
imp = pd.DataFrame(xgb.feature_importances_,index=X_new2.columns,columns=['importance'])
imp.sort_values(by='importance',ascending=False)
#best submission score on XGboost Classifier : 0.479793637145314
#The evaluation metric for this competition was F1 Score.