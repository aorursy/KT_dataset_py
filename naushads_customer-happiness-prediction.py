# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from IPython.display import display

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns



import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle



from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')



train.head()
train.info()
train.describe()
# 0 = Happy Customers , 1 = Unhappy Customers

df = pd.DataFrame(train.TARGET.value_counts())

df['Percentage'] = (df['TARGET']/df['TARGET'].sum())*100

# ~4% unhappy customers in dataset (unbalanced dataset)

df
display(train.shape)

display(train.duplicated(train.columns[1:],keep=False).value_counts()[1])

# 5791 instances which contain duplicates (have every feature incl. target same)

display(train.duplicated(train.columns[1:],keep='first').value_counts()[1])

# out of the 5791 instances, 4807 are duplicate instances, so ulimately we keep 984 instances

# Now lets delete the duplicate instances as they dont add any value

train = train.drop(train[train.duplicated(train.columns[1:],keep='first')].index)

train.reset_index(drop=True,inplace=True)

# dropped 4807 duplicate instances

display(train.shape)
#Let see if the dataset have repeated instances with different class labels, which means noise.

noise_indices = train[train.duplicated(train.columns[1:-1],keep=False)&~train.duplicated(train.columns[1:],keep=False)].index

# print(noise_indices)

non_noise_indices = train.index.difference(noise_indices)

# print(non_noise_indices)

f'We have {len(noise_indices)} instances of noise, {len(non_noise_indices)} non_noise'

# display(train.iloc[noise_indices].head(1))

# display(train.iloc[non_noise_indices].head(1))
# Example of how i got the noise instances: ( Consider Age here as Target)

# df1 = pd.DataFrame({'Id':[1,2,3,4,5,6],'Name':['John','Peter','John','John','John','John'],'City':['Boston','Japan','Boston','Dallas','Dallas','Dallas'],'Age':[23,31,21,21,21,21]})

# display(df1)



# print('Result:')

# print('train without duplicates')

# df1.drop(df1[df1.duplicated(df1.columns[1:],keep='first')].index, axis=0, inplace=True);display(df1)



# noise_sample = df1[df1.duplicated(df1.columns[1:-1],keep=False)&~df1.duplicated(df1.columns[1:],keep=False)].index

# print('instances with multiple ages(noise)')

# display(df1.loc[noise_sample])

# df1.drop(index=[2],axis=0).reset_index(drop=True)
trainNTP = train.loc[non_noise_indices]

trainNTP = shuffle(trainNTP).reset_index(drop=True) #shuffling the training data

print(trainNTP.shape[0])

trainNTPsplit5 = int(trainNTP.shape[0]/5)

trainNTP1 = trainNTP[:trainNTPsplit5]

trainNTP2 = trainNTP[trainNTPsplit5:2*trainNTPsplit5]

trainNTP3 = trainNTP[2*trainNTPsplit5:3*trainNTPsplit5]

trainNTP4 = trainNTP[3*trainNTPsplit5:4*trainNTPsplit5]

trainNTP5 = trainNTP[4*trainNTPsplit5:]

print(trainNTP1.shape[0], trainNTP2.shape[0], trainNTP3.shape[0], trainNTP4.shape[0], trainNTP5.shape[0])
# # train_for_noise_target_pred

# trainNTP = train.loc[non_noise_indices]

# trainNTP = shuffle(trainNTP).reset_index(drop=True) #shuffling the training data

# print("trainNTP",trainNTP.shape[0])



# trainNTP_0 = trainNTP[trainNTP.TARGET == 0]

# trainNTP_1 = trainNTP[trainNTP.TARGET == 1]



# trainNTP_0split5 = int(trainNTP_0.shape[0]/5)



# trainNTP1 = trainNTP_0[:trainNTP_0split5]

# trainNTP1 = pd.concat([trainNTP1,trainNTP_1])



# trainNTP2 = trainNTP_0[trainNTP_0split5:2*trainNTP_0split5]

# trainNTP2 = pd.concat([trainNTP2,trainNTP_1])



# trainNTP3 = trainNTP_0[2*trainNTP_0split5:3*trainNTP_0split5]

# trainNTP3 = pd.concat([trainNTP3,trainNTP_1])



# trainNTP4 = trainNTP_0[3*trainNTP_0split5:4*trainNTP_0split5]

# trainNTP4 = pd.concat([trainNTP4,trainNTP_1])



# trainNTP5 = trainNTP_0[4*trainNTP_0split5:]

# trainNTP5 = pd.concat([trainNTP5,trainNTP_1])



# print(trainNTP1.shape[0], trainNTP2.shape[0], trainNTP3.shape[0], trainNTP4.shape[0], trainNTP5.shape[0])

# print(trainNTP1.shape[0]+trainNTP2.shape[0]+trainNTP3.shape[0]+trainNTP4.shape[0]+trainNTP5.shape[0])
trainNTP1_y = trainNTP1.TARGET

trainNTP1_X = trainNTP1.drop(['ID','TARGET'],axis=1)

trainNTP1_xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, seed=42)

trainNTP1_xgb.fit(trainNTP1_X, trainNTP1_y)



trainNTP2_y = trainNTP2.TARGET

trainNTP2_X = trainNTP2.drop(['ID','TARGET'],axis=1)

trainNTP2_xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, seed=42)

trainNTP2_xgb.fit(trainNTP2_X, trainNTP2_y)



trainNTP3_y = trainNTP3.TARGET

trainNTP3_X = trainNTP3.drop(['ID','TARGET'],axis=1)

trainNTP3_xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, seed=42)

trainNTP3_xgb.fit(trainNTP3_X, trainNTP3_y)



trainNTP4_y = trainNTP4.TARGET

trainNTP4_X = trainNTP4.drop(['ID','TARGET'],axis=1)

trainNTP4_xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, seed=42)

trainNTP4_xgb.fit(trainNTP4_X, trainNTP4_y)



trainNTP5_y = trainNTP5.TARGET

trainNTP5_X = trainNTP5.drop(['ID','TARGET'],axis=1)

trainNTP5_xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, seed=42)

trainNTP5_xgb.fit(trainNTP5_X, trainNTP5_y)
noise = train.iloc[noise_indices]

print('noise:',noise.shape)
noise_y = noise.TARGET

noise_X = noise.drop(['ID','TARGET'],axis=1)

pred1 = trainNTP1_xgb.predict(noise_X)

display(pred1)

pred2 = trainNTP2_xgb.predict(noise_X)

# display(pred2)

pred3 = trainNTP3_xgb.predict(noise_X)

# display(pred3)

pred4 = trainNTP3_xgb.predict(noise_X)

# display(pred4)

pred5 = trainNTP3_xgb.predict(noise_X)

# display(pred5)

pd.DataFrame({'pred1':pred1})
# remove constant columns (std = 0)

remove = []

for col in train.columns:

    if(train[col].std()==0):

        remove.append(col)

print(remove)

print(len(remove))

train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)

train.shape

# we thereby removed 34 columns which dont add any relevancy (trying to minimize the curse of dimentionality)

# 371 features => 337 features
remove = []

cols = train.columns

for i in range(len(cols)-1):

    v = train[cols[i]].values

    for j in range(i+1, len(cols)):

        if np.array_equal(v,train[cols[j]].values):

            remove.append(cols[j])

print(remove)

print(len(remove))

train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)

train.shape

# we thereby removed 29 columns which are duplicates (trying to minimize the curse of dimentionality)

# 337 features => 308 features
test_ids = test.ID

test.drop(['ID'], axis=1, inplace=True)



X = train.drop(['ID','TARGET'], axis=1)

y = train.TARGET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape, X_test.shape, test.shape)
etc = ExtraTreesClassifier(random_state=42)

selector = etc.fit(X_train,y_train)
feat_imp = pd.Series(etc.feature_importances_,index=X_train.columns.values).sort_values(ascending=False)

feat_imp[:40].plot(kind='barh',figsize=(12,8)).invert_yaxis()

plt.ylabel('Feature importance score');

# var38 is most important feature (a little too much)
# feat_imp.shape = (306,)

fs = SelectFromModel(selector, prefit=True) # Meta-transformer for selecting features based on importance weights.

#selector is trained ETC classifier model

#prefit -> Whether a prefit model is expected to be passed into the constructor directly or not.

#If True, transform must be called directly

X_train = fs.transform(X_train) # Reduce X to the selected features

X_test = fs.transform(X_test)

test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)

# Before: (60816, 306) (15204, 306) (75818, 306)
#train model

m_xgb = XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, seed=42)

m_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc')

# eval_set -> A list of (X, y) tuple pairs to use as a validation set

# eval_metrix -> If a str, should be a built-in evaluation metric to use. using Area Under ROC Curve. 
m_xgb.predict_proba(X_test)

#  predict_proba: Predict the probability of each data example being of a given class.

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

print('ROC AUC score', roc_auc_score(y_true=y_test, y_score=m_xgb.predict_proba(X_test)[:,1], average='macro'))

# y_true -> True binary labels

# y_scores -> Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions

# average -> this determines the type of averaging performed on the data:

# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
submission = pd.DataFrame({'ID':test_ids,'TARGET': m_xgb.predict_proba(test)[:,1]})

submission.to_csv('submission.csv',index=False)