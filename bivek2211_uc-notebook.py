#load packages

import pandas as pd

import matplotlib as mpl 

import numpy as np 

import scipy as sp 

import IPython

from IPython import display 

import sklearn 



from scipy.stats import norm, skew 

from scipy import stats



#misc libraries

import random

import time



#ignore warnings

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings('always')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression



import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD

import graphviz



#Visualization

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix
xlsx = pd.ExcelFile("../input/ucassignmentdataset/uc-assignment-slots.xlsx")



train = xlsx.parse(2) ## train_data

test = xlsx.parse(3) ## test_data
print(train.shape)

print(test.shape)
train["hour"] = [t.hour for t in pd.DatetimeIndex(train.event_time)]

train["day"] = [t.dayofweek for t in pd.DatetimeIndex(train.event_time)]

train["month"] = [t.month for t in pd.DatetimeIndex(train.event_time)]

train['year'] = [t.year for t in pd.DatetimeIndex(train.event_time)]

train['year'] = train['year'].map({2018:0, 2019:1})

train['month'] = train['month'].map({12:0, 1:1})



train.drop('event_time', axis=1, inplace= True)

train.sample(5)
test["hour"] = [t.hour for t in pd.DatetimeIndex(test.event_time)]

test["day"] = [t.dayofweek for t in pd.DatetimeIndex(test.event_time)]

test["month"] = [t.month for t in pd.DatetimeIndex(test.event_time)]

test['year'] = [t.year for t in pd.DatetimeIndex(test.event_time)]

test['year'] = test['year'].map({2018:0, 2019:1})

test['month'] = test['month'].map({12:0, 1:1})



test.drop('event_time', axis=1, inplace= True)

test.sample(5)
train.describe(include= 'all')
test.info()
train.req_id.value_counts().head(15)
train.loc[train['req_id'] == '5c1be07342fb432300cfa4e2']
values = ['5c1be07342fb432300cfa4e2','5c1be4d8a08390260045e159','5c2a638601cd922600ccb72e','5c1be1b744288724007a1f6a','5c1be3064415802500b4effe','5c1be0c16dfd5525002df03a',

       '5c1bdfbfef671024001a29c3','5c2a629eea56562700c42851','5c1be2ffa08390260045e009','5c2a630f8a5ecd240022476b','5c1be0226eaf2c2300ebc830','5c1be065b48f05230000071f']

for val in values:

    train = train[train['req_id'] != val]

test.req_id.value_counts().head(5)
train['has_req_id'] = 0

train.loc[train['req_id'].isnull() == False, 'has_req_id'] = 1

train.drop('req_id', axis=1, inplace = True)



test['has_req_id'] = 0

test.loc[test['req_id'].isnull() == False, 'has_req_id'] = 1

test.drop('req_id', axis=1, inplace = True)
sns.countplot(x="has_req_id", data=train)
cor_mat= train[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(20,10)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
train.city_key.value_counts()
train['city_key'] = train['city_key'].map(lambda x: str(x)[:-3])

train['city_key'] = train['city_key'].map(lambda x: str(x)[5:])



test['city_key'] = test['city_key'].map(lambda x: str(x)[:-3])

test['city_key'] = test['city_key'].map(lambda x: str(x)[5:])
plt.figure(figsize = (12,8))

sns.countplot(train['city_key'])

plt.show()
tab = pd.crosstab(train['city_key'],train['has_req_id'])

print(tab)



dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)

dummy = plt.xlabel("City")

dummy = plt.ylabel("Percentage")
dummy=pd.get_dummies(train['city_key'],prefix='city_')

train=pd.concat([train,dummy],axis=1)

train.drop('city_key', axis=1, inplace= True)



dummy=pd.get_dummies(test['city_key'],prefix='city_')

test=pd.concat([test,dummy],axis=1)

test.drop('city_key', axis=1, inplace= True)
plt.figure(figsize = (8,6))

sns.countplot(train['weekday'])

plt.show()
tab = pd.crosstab(train['weekday'],train['has_req_id'])

print(tab)



dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)

dummy = plt.xlabel("Weekday")

dummy = plt.ylabel("Percentage")
dummy=pd.get_dummies(train['weekday'],prefix='day_')

train=pd.concat([train,dummy],axis=1)

train.drop('weekday', axis=1, inplace= True)



dummy=pd.get_dummies(test['weekday'],prefix='day_')

test=pd.concat([test,dummy],axis=1)

test.drop('weekday', axis=1, inplace= True)
train.rptcatg.value_counts()
tab = pd.crosstab(train['rptcatg'],train['has_req_id'])

dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)

dummy = plt.xlabel("Category")

dummy = plt.ylabel("Percentage")
dummy=pd.get_dummies(train['rptcatg'],prefix='cat_')

train=pd.concat([train,dummy],axis=1)

train.drop('rptcatg', axis=1, inplace= True)



dummy=pd.get_dummies(test['rptcatg'],prefix='cat_')

test=pd.concat([test,dummy],axis=1)

test.drop('rptcatg', axis=1, inplace= True)
train.drop('category_key', axis=1, inplace= True)

test.drop('category_key', axis=1, inplace= True)
train.hour.hist(bins=100)

plt.xlabel('Hour')

plt.ylabel('Number of customers in train')

plt.title('Customers registered in hour')

plt.show()
sns.FacetGrid(train, hue="has_req_id", size=6).map(plt.hist, "hour").add_legend()

plt.title('Relation between customer activity and hour')

plt.show()
sns.FacetGrid(train, hue="has_req_id", size=6).map(sns.kdeplot, "hour").add_legend()
train.day.hist(bins=100)

plt.xlabel('Day')

plt.ylabel('Number of customers in train')

plt.title('Customers registered in day')

plt.show()
sns.FacetGrid(train, hue="has_req_id", size=6).map(sns.kdeplot, "day").add_legend()
train.session_group.value_counts().head(15)
train.session_group = (train.session_group.astype(str).str.zfill(4))

train[['D0_possible','D0_shown','D1_shown','D2_shown']] = train['session_group'].astype(str).apply(lambda x: pd.Series(list(x))).astype(int)



test.session_group = (test.session_group.astype(str).str.zfill(4))

test[['D0_possible','D0_shown','D1_shown','D2_shown']] = test['session_group'].astype(str).apply(lambda x: pd.Series(list(x))).astype(int)
train.drop('session_group',axis=1,inplace= True)

test.drop('session_group',axis=1,inplace= True)
# def slot_method(row):

#     if row['hour']>=0 and row['hour']<=10:

#         return 1

#     elif row['hour']>10 and row['hour']<=12:

#         return 2

#     elif row['hour']>12 and row['hour']<=15:

#         return 3

#     elif row['hour']>15 and row['hour']<=18:

#         return 4

#     elif row['hour']>18 and row['hour']<=20:

#         return 5

#     else:

#         return 6



# train['slot'] = 0    

# train['slot'] = train.apply(lambda row: slot_method(row), axis=1)



# test['slot'] = 0    

# test['slot'] = test.apply(lambda row: slot_method(row), axis=1)
print("Train dataset (rows, cols):",train.shape, "\nTest dataset (rows, cols):",test.shape)
cor_mat= train[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
# only important correlations and not auto-correlations

threshold = 0.75

important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]).unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(

    list(set([(tuple(sorted(key)), important_corrs[key]) \

    for key in important_corrs])), columns=['attribute pair', 'correlation'])

# sorted by absolute value

unique_important_corrs = unique_important_corrs.ix[

    abs(unique_important_corrs['correlation']).argsort()[::-1]]

unique_important_corrs
# train.drop('slot',axis = 1 ,inplace = True)

# test.drop('slot',axis = 1 ,inplace = True)
sample = train.sample(frac=0.05)

var = ['D0_possible', 'hour', 'day', 'day__weekday', 'D0_shown', 'has_req_id']

sample = sample[var]

sns.pairplot(sample,  hue='has_req_id', palette = 'Set1', diag_kind='kde')

plt.show()
train.drop('day__weekday',axis = 1 ,inplace = True)

test.drop('day__weekday',axis = 1 ,inplace = True)
df_train.head()
df_train = train.copy()

df_test = test.copy()
train = df_train.drop(["has_req_id","record_id"],axis=1)

train_ = df_train["has_req_id"]



test = df_test.drop(["has_req_id","record_id"],axis=1)

test_ = df_test["has_req_id"]



X_train = train.values

y_train = train_.values



X_test = test.values

X_test = X_test.astype(np.float64, copy=False)

y_test = test_.values
# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Creating the model

model = Sequential()



# Inputing the first layer with input dimensions

model.add(Dense(28,activation='relu',input_dim=30,kernel_initializer='uniform'))

#The argument being passed to each Dense layer (28) is the number of hidden units of the layer. 

# A hidden unit is a dimension in the representation space of the layer.



# Adding an Dropout layer to previne from overfitting

model.add(Dropout(0.50))



#adding second hidden layer 

model.add(Dense(60,kernel_initializer='uniform', activation='relu'))



# Adding another Dropout layer

model.add(Dropout(0.50))



# adding the output layer that is binary [0,1]

model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))

#With such a scalar sigmoid output on a binary classification problem, the loss

#function you should use is binary_crossentropy



#Visualizing the model

model.summary()
#Creating an Stochastic Gradient Descent

sgd = SGD(lr = 0.01, momentum = 0.9)



# Compiling our model

model.compile(optimizer = sgd,loss = 'binary_crossentropy', metrics = ['accuracy'])

#optimizers list

#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']



# Fitting the ANN to the Training set

model.fit(X_train, y_train,batch_size = 80, epochs = 40, verbose=2)
scores = model.evaluate(X_train, y_train, batch_size=40)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_pred_probability = model.predict(X_test)

y_pred_probability
ann_prediction = (y_pred_probability > 0.5) # convert probabilities to binary output



def display_confusion_matrix(sample_test, prediction, score=None):

    cm = metrics.confusion_matrix(sample_test, prediction)

    plt.figure(figsize=(9,9))

    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    if score:

        all_sample_title = 'Accuracy Score: {0}'.format(score)

        plt.title(all_sample_title, size = 15)

    print(metrics.classification_report(sample_test, prediction))

    

score = metrics.accuracy_score(y_test, ann_prediction)

display_confusion_matrix(y_test, ann_prediction, score=score)
# rearranging the columns

result_df = df_test.copy()

cols = result_df.columns.tolist()



inserted_cols = ['has_req_id']

cols = ( [col for col in result_df if col not in inserted_cols] + [col for col in inserted_cols if col in result_df])

result_df = result_df[cols]
result_df['predicted_request_probability'] = y_pred_probability

result_df['predicted_has_requested'] = ann_prediction

# convert to csv

result_df.to_csv('test_updated.csv', index=False)
result_df = df_test.copy()

cols = result_df.columns.tolist()



inserted_cols = ['has_req_id']

cols = ( [col for col in result_df if col not in inserted_cols] + [col for col in inserted_cols if col in result_df])

result_df = result_df[cols]



result_df['predicted_request_probability'] = y_pred_probability

result_df['predicted_has_requested'] = ann_prediction

# convert to csv

result_df.to_csv('test_updated.csv', index=False)
result_df = pd.read_csv('../input/uc-test-updated/test_updated.csv')

result_df.head()
result_df.sample(5)
len(result_df)
Total = result_df['predicted_request_probability'].sum() / len(result_df)

print (Total)
result_df.has_req_id.value_counts()
test['predicted_request_probability'] = result_df['predicted_request_probability']

test['predicted_has_requested'] = result_df['predicted_has_requested']

test.sample(5)
# grouping on 3 columns and getting the count(sum) 

sessiongroup = test.groupby(['weekday', 'rptcatg','session_group']).size()  # session group is series



# converting series to dataframe 

df1 = sessiongroup.to_frame().reset_index()

df1 = df1.rename(columns= {0: 'total_records'})

df1.index.name = 'index'

df1.sample(5)
# grouping on 3 columns and getting the count(sum) where has_req_id = 1

sessiongroup = test.groupby(['weekday', 'rptcatg','session_group','has_req_id']).size()  # session group is series



# converting series to dataframe 

df2 = sessiongroup.to_frame().reset_index()

df2 = df2.rename(columns= {0: 'total_records'})

df2.index.name = 'index'

df2 = df2[df2.has_req_id != 0]

df2.sample(5)
# left join of dataframes on the basis of 3 columns 

df3 = pd.merge(df1, df2,left_on = ['weekday','rptcatg','session_group'], right_on=['weekday','rptcatg','session_group'], how='left')

df3 = df3.rename(columns={'total_records_x': 'total_records', 'total_records_y': 'total_records_with_request'})

df3['actual_conversion_probability'] = (df3['total_records_with_request']/df3['total_records'])

df3.sample(5)
# new dataframe, groupby on 3 columns with grouped probability

df4 = test.groupby(['weekday', 'rptcatg','session_group'])['predicted_request_probability'].mean().reset_index()

df4.sample(5)
# left join on two dataframes

df = pd.merge(df3, df4,left_on = ['weekday','rptcatg','session_group'], right_on=['weekday','rptcatg','session_group'], how='left')

df = df.dropna()

df.sample(10)
df['diffrence'] = abs(df['actual_conversion_probability']-df['predicted_request_probability'])

df.sample(5)
df_sort_actual_conversion_probability = df.sort_values('actual_conversion_probability')  # , ascending=False)

df_sort_actual_conversion_probability.sample(5)
df_predicted_request_probability = df.sort_values('predicted_request_probability', ascending=False)

df_predicted_request_probability.sample(5)
df_temp = df.sort_values('diffrence').head(5)

df_temp
# id_test = df_test['record_id'].values

# label_test = df_test['has_req_id'].values



# target_train = df_train['has_req_id'].values



# train = df_train.drop(['has_req_id','record_id'], axis = 1)

# test = df_test.drop(['record_id','has_req_id'], axis = 1)
# from sklearn.linear_model import LogisticRegression 

# Model = LogisticRegression() 

# Model.fit(train, target_train) 

# y_pred = Model.predict(test)

# print('accuracy is',accuracy_score(y_pred,label_test))
# from sklearn.tree import DecisionTreeClassifier 

# Model = DecisionTreeClassifier() 

# Model.fit(train, target_train) 

# y_pred = Model.predict(test)

# print('accuracy is',accuracy_score(y_pred,label_test))
# from sklearn.ensemble import AdaBoostClassifier 

# Model=AdaBoostClassifier() 

# Model.fit(train, target_train) 

# y_pred=Model.predict(test)

# print('accuracy is ',accuracy_score(y_pred,label_test))
# from sklearn.ensemble import GradientBoostingClassifier 

# Model=GradientBoostingClassifier() 

# Model.fit(train, target_train) 

# y_pred=Model.predict(test)

# print('accuracy is ',accuracy_score(y_pred,label_test))
# from sklearn.ensemble import RandomForestClassifier 

# Model=RandomForestClassifier(max_depth=2) 

# Model.fit(train, target_train) 

# y_pred=Model.predict(test)

# print('accuracy is ',accuracy_score(y_pred,label_test))



# y_true = label_test

# from sklearn.metrics import confusion_matrix

# cm_dt = confusion_matrix(y_true,y_pred)

# cm_df = pd.DataFrame(cm_dt,

#                      index = ['placed','unplaced'], 

#                      columns = ['placed','unplaced'])



# plt.figure(figsize=(5.5,4))

# sns.heatmap(cm_df, annot=True)

# plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(label_test, y_pred)))

# plt.ylabel('True label')

# plt.xlabel('Predicted label')

# plt.show()
# class Ensemble(object):

#     def __init__(self, n_splits, stacker, base_models):

#         self.n_splits = n_splits

#         self.stacker = stacker

#         self.base_models = base_models



#     def fit_predict(self, X, y, T):

#         X = np.array(X)

#         y = np.array(y)

#         T = np.array(T)



#         folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=314).split(X, y))



#         S_train = np.zeros((X.shape[0], len(self.base_models)))

#         S_test = np.zeros((T.shape[0], len(self.base_models)))

#         for i, clf in enumerate(self.base_models):



#             S_test_i = np.zeros((T.shape[0], self.n_splits))



#             for j, (train_idx, test_idx) in enumerate(folds):

#                 X_train = X[train_idx]

#                 y_train = y[train_idx]

#                 X_holdout = X[test_idx]





#                 print ("Base model %d: fit %s model | fold %d" % (i+1, str(clf).split('(')[0], j+1))

#                 clf.fit(X_train, y_train)

#                 cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')

#                 print("cross_score [roc-auc]: %.5f [gini]: %.5f" % (cross_score.mean(), 2*cross_score.mean()-1))

#                 y_pred = clf.predict_proba(X_holdout)[:,1]                



#                 S_train[test_idx, i] = y_pred

#                 S_test_i[:, j] = clf.predict_proba(T)[:,1]

#             S_test[:, i] = S_test_i.mean(axis=1)



#         results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')

#         # Calculate gini factor as 2 * AUC - 1

#         print("Stacker score [gini]: %.5f" % (2 * results.mean() - 1))



#         self.stacker.fit(S_train, y)

#         res = self.stacker.predict_proba(S_test)[:,1]

#         return res
# # LightGBM params

# # lgb_1

# lgb_params1 = {}

# lgb_params1['learning_rate'] = 0.02

# lgb_params1['n_estimators'] = 650

# lgb_params1['max_bin'] = 10

# lgb_params1['subsample'] = 0.8

# lgb_params1['subsample_freq'] = 10

# lgb_params1['colsample_bytree'] = 0.8   

# lgb_params1['min_child_samples'] = 500

# lgb_params1['seed'] = 314

# lgb_params1['num_threads'] = 4



# # lgb2

# lgb_params2 = {}

# lgb_params2['n_estimators'] = 1090

# lgb_params2['learning_rate'] = 0.02

# lgb_params2['colsample_bytree'] = 0.3   

# lgb_params2['subsample'] = 0.7

# lgb_params2['subsample_freq'] = 2

# lgb_params2['num_leaves'] = 16

# lgb_params2['seed'] = 314

# lgb_params2['num_threads'] = 4



# # lgb3

# lgb_params3 = {}

# lgb_params3['n_estimators'] = 1100

# lgb_params3['max_depth'] = 4

# lgb_params3['learning_rate'] = 0.02

# lgb_params3['seed'] = 314

# lgb_params3['num_threads'] = 4



# # XGBoost params

# xgb_params = {}

# xgb_params['objective'] = 'binary:logistic'

# xgb_params['learning_rate'] = 0.04

# xgb_params['n_estimators'] = 490

# xgb_params['max_depth'] = 4

# xgb_params['subsample'] = 0.9

# xgb_params['colsample_bytree'] = 0.9  

# xgb_params['min_child_weight'] = 10

# xgb_params['num_threads'] = 4
# # Base models

# lgb_model1 = LGBMClassifier(**lgb_params1)

# lgb_model2 = LGBMClassifier(**lgb_params2)

# lgb_model3 = LGBMClassifier(**lgb_params3)

# xgb_model = XGBClassifier(**xgb_params)



# # Stacking model

# log_model = LogisticRegression()
# stack = Ensemble(n_splits=3,stacker = log_model, base_models = (lgb_model1, lgb_model2, lgb_model3, xgb_model))  
# y_prediction = stack.fit_predict(train, target_train, test) 

# y_prediction
# for i in range(len(y_prediction)):

#     if y_prediction[i]<0.5:

#         y_prediction[i]=0

#     else:

#         y_prediction[i]=1

        

# print('accuracy with ensemble is',accuracy_score(y_prediction,label_test))