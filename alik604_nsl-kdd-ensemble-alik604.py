%config IPCompleter.greedy=True

import pandas as pd

import seaborn as sns

import numpy as np

import re

import sklearn



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import matplotlib as matplot

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



from sklearn.model_selection import train_test_split
train = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv')

test = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv')

train.shape

test.shape 

train.columns = range(train.shape[1])

test.columns = range(test.shape[1])

labels = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',

'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',

'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',

'su_attempted', 'num_root', 'num_file_creations', 'num_shells',

'num_access_files', 'num_outbound_cmds', 'is_host_login',

'is_guest_login', 'count', 'srv_count', 'serror_rate',

'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',

'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',

'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',

'dst_host_srv_serror_rate', 'dst_host_rerror_rate',

'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level']# subclass - > attack_type

combined_data = pd.concat([train, test])

combined_data.shape

combined_data.head(5)
combined_data.columns = labels

combined_data = combined_data.drop('difficulty_level', 1)

combined_data.head(3)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



print(set(list(combined_data['attack_type']))) # use print to make it print on single line 

combined_data['attack_type'] = le.fit_transform(combined_data['attack_type'])

combined_data['protocol_type'] = le.fit_transform(combined_data['protocol_type'])

combined_data['service'] = le.fit_transform(combined_data['service'])

combined_data['flag'] = le.fit_transform(combined_data['flag'])



print('\nDescribing attack_type: ')

combined_data['attack_type'].describe()
# select least correlated

corr_matrix = combined_data.corr().abs().sort_values('attack_type')

# tmp.head(10) # to view CORR matrix 

leastCorrelated = corr_matrix['attack_type'].nsmallest(10)

leastCorrelated = list(leastCorrelated.index)



# select least correlated

leastSTD =  combined_data.std().to_frame().nsmallest(5, columns=0)

leastSTD = list(leastSTD.transpose().columns)  #fuckin pandas.core.indexes.base.Index   -_-

#tmp = tmp.append('num_outbound_cmds')  # might not work...

featureElimination = set(leastCorrelated + leastSTD)

len(featureElimination)

featureElimination
combined_data=combined_data.drop(featureElimination,axis=1)

data_x = combined_data.drop('attack_type', axis=1)

data_y = combined_data.loc[:,['attack_type']]

del combined_data # free mem

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=.5, random_state=42) # TODO
print("Thats how to rid rid of {0} dimentions of data, from the 10 lowest STD and 5 lowest correlation".format(len(featureElimination)))



X_train.shape

X_test.shape
from sklearn import linear_model



from sklearn.ensemble import VotingClassifier



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import IsolationForest



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



import gc 

gc.collect()
x = X_train

y = y_train['attack_type'].ravel()



clf1 = DecisionTreeClassifier() 

clf2 = RandomForestClassifier(n_estimators=25, random_state=1)

clf3 = GradientBoostingClassifier()

ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .76 # without this lil fucker, Acc: 0.75 [Ensemble], 0.78 with 



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='hard') 



for clf, label in zip([clf1, clf2, clf3,ET, eclf], ['DecisionTreeClassifier', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 

    tmp = clf.fit(x,y)

    pred = clf.score(X_test,y_test)

    print("Acc: %0.2f [%s]" % (pred,label))
LR = linear_model.LinearRegression()

LR.fit(X_train, y_train)

lr_score = LR.score(X_test, y_test)

print('Linear regression processing')

print('Linear regression Score: %.2f %%' % lr_score)
AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, learning_rate=1.0)

RF = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto', bootstrap=True)

ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)

GB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, max_features='auto')
# y_train = Y_train['attack_type'].ravel()

# x_train = X_train.values

# x_test = X_test.values
AB.fit(X_train, y_train)

AB_feature = AB.feature_importances_

#AB_feature

ab_score = AB.score(X_test, y_test)



print('AdaBoostClassifier processing ,,,')

print('AdaBoostClassifier Score: %.3f %%' % ab_score)
RF.fit(X_train, y_train)

RF_feature = RF.feature_importances_

#RF_feature



rf_score = RF.score(X_test, y_test)



print('RandomForestClassifier processing ,,,')

print('RandomForestClassifier Score: %.3f %%' % rf_score)
ET.fit(X_train, y_train)

ET_feature = ET.feature_importances_

#ET_feature



et_score = ET.score(X_test, y_test)



print('ExtraTreesClassifier processing ,,,')

print('ExtraTreeClassifier: %.3f %%' % et_score)
GB.fit(X_train, y_train)



GB_feature = GB.feature_importances_

#GB_feature



gb_score = GB.score(X_test, y_test)



print('GradientBoostingClassifier processing ,,,')

print('GradientBoostingClassifier Score: %.3f %%' % gb_score)
feature_df = pd.DataFrame({'features': X_train.columns.values, # names

                           'AdaBoost' : AB_feature,

                           'RandomForest' : RF_feature,

                           'ExtraTree' : ET_feature,

                           'GradientBoost' : GB_feature

                          })

#feature_df.features

feature_df.head(5)
n = 10

a_f = feature_df.nlargest(n, 'AdaBoost')

e_f = feature_df.nlargest(n, 'ExtraTree')

g_f = feature_df.nlargest(n, 'GradientBoost')

r_f = feature_df.nlargest(n, 'RandomForest')



result = pd.concat([a_f, e_f, g_f, r_f]).drop_duplicates() 

result.index

result
# X_train_SF = X_train[result.index]

# X_test_SF = X_test[result.index]







selected_features = result['features'].values.tolist()

X_train_SF = X_train[selected_features]

X_test_SF = X_test[selected_features]





x = X_train_SF#.reshape(-1, 26)  # 31

y = y_train['attack_type'].ravel()





# x=x[:20000]

# y=y[:20000]





x.shape

y.size
clf1 = DecisionTreeClassifier() 

clf2 = RandomForestClassifier(n_estimators=25, random_state=1)# .77

clf3 = GradientBoostingClassifier() # .76

ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .76 # without this lil fucker, Acc: 0.75 [Ensemble], 0.78 with 



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='hard') 

# n =7 with better selection; .79

# n =7 ; .77

# n =10 ; .78

# n =14 ; .77



for clf, label in zip([clf1, clf2, clf3,ET, eclf], ['DecisionTreeClassifier', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 

    # scores = cross_val_score(clf, x, y, cv=2, scoring='accuracy') # cv= 5 

    # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



    tmp = clf.fit(x,y)

    pred = clf.score(X_test_SF,y_test)

    print("Acc: %0.2f [%s]" % (pred,label))