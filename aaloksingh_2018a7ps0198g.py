import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline
df = pd.read_csv(("../input/minor-project-2020/train.csv"))
df.head()
df.info()
df.describe()
from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler
# X=df[['col_1','col_2','col_3','col_4','col_5','col_7','col_8','col_11','col_13','col_15','col_16','col_23','col_56']]

# X = df.copy()

# # X.drop(['target'],axis=1,inplace=True)

# y=df[['target']]
# y
df.drop(['id'],axis=1,inplace=True)

X_y = df.copy()
# df.drop(['id'],axis=1,inplace=True)

# X_y=df[['col_1','col_2','col_3','col_4','col_5','col_7','col_8','col_11','col_13','col_15','col_16','col_23','col_56','target']]
X_y
X_y_test = X_y.loc[640000:]
X_y = X_y.loc[:639999]
Z = X_y[X_y['target']==1]
Z.describe()
for i in range(300):

    X_y = X_y.append(Z, ignore_index=True, verify_integrity=False, sort=None)
X_y = X_y.sample(frac=1).reset_index(drop=True)
y_train = X_y[['target']]
X_y.drop(['target'],axis=1,inplace=True)
X_train = X_y
y_test = X_y_test[['target']]
X_y_test.drop(['target'],axis=1,inplace=True)
X_test = X_y_test
print(len(X_train), len(X_test),len(y_train),len(y_test))
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
X_train,X_test,y_train,y_test
# from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)
# from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler()

# scaled_X_train = scalar.fit_transform(X_train)

# scaled_X_test = scalar.transform(X_test)
from sklearn.metrics import roc_auc_score
# model = DecisionTreeClassifier()

# over = SMOTE(sampling_strategy=0.1, k_neighbors=4)

# under = RandomUnderSampler(sampling_strategy=0.5)

# steps = [('over', over), ('under', under), ('model', model)]

# pipeline = Pipeline(steps=steps)

# # evaluate pipeline

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



# scores = cross_val_score(pipeline, scaled_X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# print(scores)
# y_pred=gnb.predict(scaled_X_test)
# from sklearn.calibration import CalibratedClassifierCV

# from sklearn.ensemble import RandomForestClassifier

# calibrated_forest = CalibratedClassifierCV(

#    base_estimator=RandomForestClassifier(n_estimators=20))

# param_grid = {

#    'base_estimator__max_depth': [2, 10]}

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# search = GridSearchCV(calibrated_forest, param_grid, scoring='roc_auc', cv=5, verbose=3 )

# search.fit(scaled_X_train, y_train)
# y_pred = search.predict(scaled_X_test)
# roc_auc_score(y_test,y_pred)
from sklearn.linear_model import LogisticRegression

# from sklearn.cross_validation import KFold, cross_val_score

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
lr = LogisticRegression(C = 3000, penalty = 'l2',max_iter=2000)

lr.fit(scaled_X_train, y_train)

y_pred = lr.predict_proba(scaled_X_test)

# y_pred = lr.predict(scaled_X_test)

print(y_pred)

z=[element[1] for element in y_pred]

print(roc_auc_score(y_test,z))
# from sklearn.calibration import CalibratedClassifierCV

# from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import RandomForestClassifier

# # calibrated_forest = CalibratedClassifierCV(

# #    base_estimator=RandomForestClassifier(n_estimators=20))

# # param_grid = {

# #    'base_estimator__max_depth': [2,6,10]}

# # over = SMOTE(sampling_strategy=0.1, k_neighbors=4)

# # under = RandomUnderSampler(sampling_strategy=0.5)

# # steps = [('over', over), ('under', under), ('model', calibrated_forest)]

# # pipeline = Pipeline(steps=steps)

# # search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=3 )

# # search.fit(scaled_X_train, y_train)

# # model = DecisionTreeClassifier()

# # over = SMOTE(sampling_strategy=0.1, k_neighbors=4)

# # under = RandomUnderSampler(sampling_strategy=0.5)

# # steps = [('over', over), ('under', under), ('model', model)]

# # pipeline = Pipeline(steps=steps)

# # # evaluate pipeline

# # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# parameters = {'criterion': ("gini","entropy"), 'max_depth': [10]}

# search = GridSearchCV(pipeline, param_grid=parameters, scoring='roc_auc', cv=cv, verbose=3 )

# search.fit(scaled_X_train, y_train)

##############################################################################################################################
# df[df['target']==1].describe()
# df.corr()
# pd.set_option('display.max_columns', None)  

# pd.set_option('display.max_rows', None)
# cr=df.corr()
# cr
# cr[abs(cr)>0.1]
# pd.reset_option('display.max_columns')  

# pd.reset_option('display.max_rows')
# mask = np.zeros_like(df.corr(), dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True

# sns.set_style('whitegrid')

# plt.subplots(figsize = (200,200))

# sns.heatmap(df.corr(), 

#             annot=True,

#             mask = mask,

#             cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

#             linewidths=.9, 

#             linecolor='white',

#             fmt='.2g',

#             center = 0,

#             square=True)
# df.drop(['id'],axis=1,inplace=True)
# X=df.copy()

# X.drop(['target'],axis=1,inplace=True)

# y=df[['target']]
# df.iloc[:,0:-1]
# from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)
# print(len(X_train), len(X_test))
# from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler()

# scaled_X_train = scalar.fit_transform(X_train)

# scaled_X_test = scalar.transform(X_test)
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()

# gnb.fit(scaled_X_train, y_train)

# y_pred=gnb.predict(scaled_X_test)
# gnb.score(scaled_X_test,[1 for i in range(len(y_test))])
# gnb.score(scaled_X_test,y_test)
# from sklearn.metrics import roc_auc_score
# roc_auc_score(y_test,y_pred)

# #0.5666675610890487 for gnb

test_df = pd.read_csv("../input/minor-project-2020/test.csv")
test_df.describe()
test_df_X = test_df.copy()

# test_df_X = test_df[['col_1','col_2','col_3','col_4','col_5','col_7','col_8','col_11','col_13','col_15','col_16','col_23','col_56']].copy()
test_df_X.drop('id',axis=1,inplace=True)
test_df_X
X_train
scaled_test_df = scalar.fit_transform(test_df_X)
# test_y_pred = gnb.predict(scaled_test_df)

test_y_pred = lr.predict_proba(scaled_test_df)

# y_pred = lr.predict_proba(scaled_X_test)

# y_pred = lr.predict(scaled_X_test)

print(test_y_pred)

z=[element[1] for element in test_y_pred]

# len(test_y_pred)

len(z)
# test_y_pred

# z
# prediction = pd.DataFrame(test_y_pred, columns=['target'],)

prediction = pd.DataFrame(z, columns=['target'],)
# prediction
id = test_df[['id']]
id['target'] = prediction['target']
# id
# id[id['target']==1]
# pd.set_option('display.max_rows', None)

# print(len(y_test[y_test['target']==1]))

# print(len(y_test))

# pd.reset_option('display.max_rows')
id.to_csv('submission9.csv',index=False)
# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
# dt = DecisionTreeClassifier()
# dt.fit(scaled_X_train, y_train)
# y_pred = dt.predict(scaled_X_test)
# print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))
# roc_auc_score(y_test,y_pred)

# 0.49883233158026546 for decisionTree
# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [2]}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=10)



# clf.fit(scaled_X_train, y_train)
# clf.score(scaled_X_test, y_test)
# y_pred = clf.predict(scaled_X_test)
# roc_auc_score(y_test,y_pred)
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=1)

# rf.fit(scaled_X_train, y_train)

# print(rf.score(scaled_X_test, y_test))
# y_pred = rf.predict(scaled_X_test)
# roc_auc_score(y_test,y_pred)
# from xgboost import XGBClassifier
# xgb = XGBClassifier()

# xgb.fit(scaled_X_train, y_train)
# y_pred = xgb.predict(scaled_X_test)
# roc_auc_score(y_test,y_pred)