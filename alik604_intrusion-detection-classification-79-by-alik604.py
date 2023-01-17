# [up to date code](https://github.com/alik604/cyber-security/blob/master/Intrusion-Detection/UNSW_NB15.ipynb)



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
import zipfile



zip_ref = zipfile.ZipFile('intrusion-detection.zip', 'r')

zip_ref.extractall()

zip_ref.close()

!ls
train = pd.read_csv('./Train_data.csv')

test = pd.read_csv('./test_data.csv')

test = test.drop('Unnamed: 0', axis=1)
# train=train.sample(frac =.50,random_state=1) # TODO 
from sklearn import preprocessing

le = preprocessing.LabelEncoder()





set(list(train['xAttack']))

train['xAttack'] = le.fit_transform(train['xAttack'])

test['xAttack'] = le.fit_transform(test['xAttack'])



train['xAttack'].describe()



train['protocol_type'] = le.fit_transform(train['protocol_type'])

test['protocol_type'] = le.fit_transform(test['protocol_type'])



print('\n')

tmp = train.corr().abs().sort_values('xAttack')

garbage = tmp['xAttack'].nsmallest(5)



garbage

garbage = list(garbage.index) + ['num_outbound_cmds','is_host_login']



X_train = train.drop('xAttack', axis=1)

Y_train = train.loc[:,['xAttack']]

X_test = test.drop('xAttack', axis=1)

Y_test = test.loc[:,['xAttack']]
# df.ix[2,:]==0 # see where STD== 0, which means all values are the same... which is useless 

# train = train.drop('num_outbound_cmds', axis=1)

## bad sample code I fixed 



con_list = [

    'protocol_type', 'service', 'flag', 'land', 'logged_in', 'su_attempted',

    'is_host_login', 'is_guest_login'

]



df = X_train.drop(con_list, axis=1)



#drop n smallest std features

df = df.std(axis=0).to_frame()

tmp = df.nsmallest(5, columns=0)

tmp = list(

    tmp.transpose().columns)  #fuckin pandas.core.indexes.base.Index   -_-

#tmp = tmp.append('num_outbound_cmds')  # might not work...

tmp = set(tmp + garbage)

len(tmp)

tmp
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
LR = linear_model.LinearRegression()

LR.fit(X_train, Y_train)

lr_score = LR.score(X_test, Y_test)

print('Linear regression processing ,,,')

print('Linear regression Score: %.2f %%' % lr_score)
try:  #TODO

    X_train = X_train.drop(tmp,axis=1)

    X_test = X_test.drop(tmp,axis=1)

except:

    None

    

X_train.shape

X_test.shape
LR = linear_model.LinearRegression()

LR.fit(X_train, Y_train)

lr_score = LR.score(X_test, Y_test)

print('Linear regression processing ,,,')

print('Linear regression Score: %.2f %%' % lr_score)
AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, learning_rate=1.0)

RF = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto', bootstrap=True)

ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)

GB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, max_features='auto')
# y_train = Y_train['xAttack'].ravel()

# x_train = X_train.values

# x_test = X_test.values
AB.fit(X_train, Y_train)

AB_feature = AB.feature_importances_

#AB_feature

ab_score = AB.score(X_test, Y_test)



print('AdaBoostClassifier processing ,,,')

print('AdaBoostClassifier Score: %.3f %%' % ab_score)
RF.fit(X_train, Y_train)

RF_feature = RF.feature_importances_

#RF_feature



rf_score = RF.score(X_test, Y_test)



print('RandomForestClassifier processing ,,,')

print('RandomForestClassifier Score: %.3f %%' % rf_score)
ET.fit(X_train, Y_train)

ET_feature = ET.feature_importances_

#ET_feature



et_score = ET.score(X_test, Y_test)



print('ExtraTreesClassifier processing ,,,')

print('ExtraTreeClassifier: %.3f %%' % et_score)
GB.fit(X_train, Y_train)



GB_feature = GB.feature_importances_

#GB_feature



gb_score = GB.score(X_test, Y_test)



print('GradientBoostingClassifier processing ,,,')

print('GradientBoostingClassifier Score: %.3f %%' % gb_score)
feature_df = pd.DataFrame({#'features': X_train.columns.values, # names

                           'AdaBoost' : AB_feature,

                           'RandomForest' : RF_feature,

                           'ExtraTree' : ET_feature,

                           'GradientBoost' : GB_feature

                          })

#feature_df.features

feature_df.head(2)
n = 7

a_f = feature_df.nlargest(n, 'AdaBoost')

e_f = feature_df.nlargest(n, 'ExtraTree')

g_f = feature_df.nlargest(n, 'GradientBoost')

r_f = feature_df.nlargest(n, 'RandomForest')



result = pd.concat([a_f, e_f, g_f, r_f])

result = result.drop_duplicates() 

result.shape



print('\n')



garbage = np.argsort(result.transpose().mean())

garbage = garbage.sort_index()[-5:] # FML... :'(

    

garbage



result = result.drop(garbage.index)

result.shape







arr = X_train.columns.to_numpy()

result = result.set_index(np.take(arr,result.index))

result
X_train_SF = X_train[result.index]

X_test_SF = X_test[result.index]



x = X_train_SF#.reshape(-1, 26)  # 31

y = Y_train['xAttack'].ravel()

x.shape

y.size
clf1 = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .76 

clf2 = RandomForestClassifier(n_estimators=25, random_state=1)# .77

clf3 = GradientBoostingClassifier() # .76

ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .76 # without this lil fucker, Acc: 0.75 [Ensemble], 0.78 with 



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='hard') 

# n =7 with better selection; .79

# n =7 ; .77

# n =10 ; .78

# n =14 ; .77





# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='soft') #.76



for clf, label in zip([clf1, clf2, clf3,ET, eclf], ['Logistic Regression', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 

    # scores = cross_val_score(clf, x, y, cv=2, scoring='accuracy') # cv= 5 

    # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



    tmp = clf.fit(x,y)

    pred = clf.score(X_test_SF,Y_test)

    print("Acc: %0.2f [%s]" % (pred,label))
# import multiprocessing



# ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .76 

# RF = RandomForestClassifier(n_estimators=25, random_state=1)# .75

# GB = GradientBoostingClassifier() # .74

# ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False) # .77 # without this lil fucker, Acc: 0.75 [Ensemble]

# clfList = [ET,RF,GB,ET]



# def spawn(clf):

#   clf.fit(x,y)

#   print("Done another one!")



# import time

# start = time.time()



# if __name__ == '__main__':

  

#   for i in clfList:

#     print(i)

#     #spawn(i) # 16 secounds 



#     # # 16 secounds

#     # p = threading.Thread(target=spawn, args=(i,))

#     # p.start()

#     # p.join()

    

#   p=multiprocessing.Pool(6) # 15.65

#   results = p.map(spawn,clfList) # clfList has 4 models, first 2 are fast, last 2 are slow

#   results





# end = time.time()

# print(end - start)

# ET.score(X_test_SF,Y_test)

import pandas as pd

test_data = pd.read_csv("../input/intrusion-detection/test_data.csv")

Train_data = pd.read_csv("../input/intrusion-detection/Train_data.csv")