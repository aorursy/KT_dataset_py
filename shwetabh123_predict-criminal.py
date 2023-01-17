# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import lightgbm as lgb
from sklearn.svm import SVC
import matplotlib.pylab as plt
import matplotlib.pyplot as plote
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
%matplotlib inline
dataset =pd.read_csv('../input/criminal_train.csv')
# The first 5 rows of dataset
dataset.head(5)
# feature scaling
dataset.VESTR = preprocessing.scale(dataset.VESTR)
dataset.ANALWT_C = preprocessing.scale(dataset.ANALWT_C)

# separating dependent and independent variable
# dropping IDs
target = dataset.Criminal
features = dataset.drop(['Criminal','PERID'], axis = 1)

#this plot is used to show whether there is null value or not 

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#this is used to show the all the column infornation
dataset.info()

# this command is used to show the statical ananlysis of the dataset
dataset.describe()
ax=plote.figure(figsize=(10,8))
ax=sns.countplot(x="Criminal",palette="inferno",data=dataset)
ax.set_xlabel("Classes")
ax.set_ylabel("Count")
ax.set_title("Criminal Count")
xgb = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.7,
 colsample_bytree=0.6,
 reg_aplha=1.68,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb = xgb.fit(features, target)
importances = xgb.feature_importances_
feature_names = features.columns.values
data = pd.DataFrame({'features': feature_names,'importances':importances})
new_index = (data['importances'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['features'], y=sorted_data['importances'])
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Importances')
plt.title('feature importances')
plt.show()
optimal_features_xgb = ['IFATHER', 'NRCH17_2', 'IRHHSIZ2', 'IRKI17_2', 'IRHH65_2', 'PRXYDATA', 'MEDICARE', 'CAIDCHIP', \
            'GRPHLTIN', 'HLCNOTMO', 'IRMCDCHP', 'OTHINS', 'CELLNOTCL', 'IRFAMSOC', 'IRFSTAMP', 'IRWELMOS',\
            'IRPINC3', 'IRFAMIN3', 'IIFAMIN3', 'GOVTPROG', 'POVERTY3', 'TOOLONG', 'TROUBUND', 'PDEN10',\
            'COUTYP2', 'ANALWT_C', 'VESTR', 'VEREP', 'PRVHLTIN', 'IRPRVHLT', 'IRMEDICR', 'IIWELMOS']

svm = SVC(kernel='linear', C=4)
svm.fit(features, target)
  
importances = svm.coef_
features_names = features.columns.values
data = pd.DataFrame({'features': feature_names,'importances':importances[0]})
new_index = (data['importances'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['features'], y=sorted_data['importances'])
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Importances')
plt.title('Feature importances for Linear SVM')
plt.show()
optimal_features_LSVM = ['IRKI17_2', 'IRFAMPMT', 'IRINSUR4', 'IIPRVHLT', 'IIFAMIN3',
       'IRFAMSVC', 'OTHINS', 'IIFAMSSI','IFATHER', 'IIKI17_2', 'IIHHSIZ2','IIOTHHLT',\
        'IRFAMSOC', 'IRFAMIN3', 'IIMCDCHP', 'IRMEDICR','IRPRVHLT']
train_data=lgb.Dataset(features,label=target)
param = {'num_leaves':120,'max_depth':5,'learning_rate':.1,'max_bin':1200}
num_round=50
lgbm=lgb.train(param,train_data,num_round)
importances = lgbm.feature_importance(importance_type='split')
features_names = features.columns.values
data = pd.DataFrame({'features': feature_names,'importances':importances})
new_index = (data['importances'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['features'], y=sorted_data['importances'])
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Importances')
plt.title('feature importances for LightGBM')
plt.show()
optimal_features_LGBM = sorted_data[0:27].features.values
X_train, X_test, y_train, y_test = train_test_split(features[optimal_features_xgb], target ,test_size=0.4, random_state=7)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test ,test_size=0.5, random_state=7)
xgb = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.7,
 colsample_bytree=0.6,
 reg_alpha=0.5,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb = xgb.fit(X_train, y_train)
res_xgb = xgb.predict(X_test)
score_xgb = metrics.accuracy_score(res_xgb, y_test)
print("Accuracy : %.4g" % score_xgb)
# getting probability for test and cross validation
element_xgb = xgb.predict_proba(X_test)
elementcv_xgb = xgb.predict_proba(X_cv)
clf = SVC(kernel='rbf',C=4,probability=True)
clf.fit(X_train, y_train)
res_ksvm = clf.predict(X_test)
score_ksvm = metrics.accuracy_score(res_ksvm, y_test)
print("Accuracy : %.4g" % score_ksvm)
element_ksvm = clf.predict_proba(X_test)
elementcv_ksvm = clf.predict_proba(X_cv)
X_train, X_test, y_train, y_test = train_test_split(features[optimal_features_LGBM], target ,test_size=0.4, random_state=7)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test ,test_size=0.5, random_state=7)

train_data=lgb.Dataset(X_train,label=y_train)
param = {'num_leaves':120,'max_depth':5,'learning_rate':.1,'max_bin':1200}
num_round=50
lgbm=lgb.train(param,train_data,num_round)
ypred2=lgbm.predict(X_test)
res3 = (ypred2 >= .5).astype(int)
score_lgbm = metrics.accuracy_score(res3, y_test)
print("Accuracy : %.4g" % score_lgbm)
# for test set
temp = pd.DataFrame(np.zeros((len(ypred2), 1)))

for i in range(0,len(ypred2)):
    temp[0][i] = ypred2[i]
    
temp2 = 1-temp
para3 = pd.concat([temp2,temp], axis = 1)
element_lgb = para3.values

# for cv set
pred_cv = lgbm.predict(X_cv)
temp = pd.DataFrame(np.zeros((len(pred_cv), 1)))

for i in range(0,len(pred_cv)):
    temp[0][i] = pred_cv[i]
    
temp2 = 1-temp
para3 = pd.concat([temp2,temp], axis = 1)
elementcv_lgb = para3.values
X_train, X_test, y_train, y_test = train_test_split(features[optimal_features_LSVM], target ,test_size=0.4, random_state=7)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test ,test_size=0.5, random_state=7)
svm = SVC(kernel='linear', C=3)
svm.fit(X_train, y_train)
res_svm = svm.predict(X_test)
score_lsvm = metrics.accuracy_score(res_svm, y_test)
print("Accuracy : %.4g" % score_lsvm)
import keras
from keras.models import Sequential
from keras.layers import Dense

X_train, X_test, y_train, y_test = train_test_split(features, target ,test_size=0.4, random_state=7)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test ,test_size=0.5, random_state=7)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 70))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)
# for test set
pred_nn = classifier.predict(X_test)
res_nn = (pred_nn >= .5).astype(int)
score_nn = metrics.accuracy_score(res_nn, y_test)
print("Accuracy : %.4g" % score_nn)
# for test set
temp = pd.DataFrame(np.zeros((len(pred_nn), 1)))

for i in range(0,len(pred_nn)):
    temp[0][i] = pred_nn[i]
    
temp2 = 1-temp
para4 = pd.concat([temp2,temp], axis = 1)
element_nn = para4.values

# for cross validation set
pred_nn = classifier.predict(X_cv)
temp = pd.DataFrame(np.zeros((len(pred_nn), 1)))

for i in range(0,len(pred_nn)):
    temp[0][i] = pred_nn[i]
    
temp2 = 1-temp
para4 = pd.concat([temp2,temp], axis = 1)
elementcv_nn = para4.values
#para3 = para3.values
decide0 = pd.DataFrame(np.zeros((len(elementcv_nn), 4)))
decide1 = pd.DataFrame(np.zeros((len(elementcv_nn), 4)))
decide0 = decide0.values
decide1 = decide1.values
for i in range(0,len(decide0)):
    """decide0[i][0] = (1 if para1[i][0]>0.5 else 0)
    decide0[i][1] = (1 if para2[i][0]>0.5 else 0)
    decide0[i][2] = (1 if para31[i][0]>0.5 else 0)
    decide1[i][0] = (1 if para1[i][1]>0.5 else 0)
    decide1[i][1] = (1 if para2[i][1]>0.5 else 0)
    decide1[i][2] = (1 if para31[i][1]>0.5 else 0)""" 
    decide0[i][0] = elementcv_xgb[i][0]
    decide0[i][1] = elementcv_lgb[i][0]
    decide0[i][2] = elementcv_ksvm[i][0]
    decide0[i][3] = elementcv_nn[i][0]
    decide1[i][0] = elementcv_xgb[i][1]
    decide1[i][1] = elementcv_lgb[i][1]
    decide1[i][2] = elementcv_ksvm[i][1]
    decide1[i][3] = elementcv_nn[i][1]
k0 = decide0.mean(axis=1)
k1 = decide1.mean(axis=1)
def get_acc(thresh, k0=k0,k1=k1):
    final = pd.DataFrame(np.zeros((len(k0), 1)))
    for i in range(0,len(k0)):
        if (k0[i]-k1[i])>thresh:
            final[0][i]=0
        else:
            final[0][i]=1
    return metrics.accuracy_score(y_cv, final)
#print(get_acc(0.081))
t = np.arange(0.0, 0.5, 0.001)  
s = np.empty([500,1])
i=0
maxi = 0.0
iind = 0
for item in t:
    h = get_acc(item)
    s[i] = h
    if h>maxi:
        maxi = h
        iind = i
    i = i+1
print(iind*0.001)
plote.plot(t,s)
plote.show()
#para3 = para3.values
       
decide0 = pd.DataFrame(np.zeros((len(element_nn), 4)))
decide1 = pd.DataFrame(np.zeros((len(element_nn), 4)))
decide0 = decide0.values
decide1 = decide1.values
for i in range(0,len(decide0)):
    """decide0[i][0] = (1 if para1[i][0]>0.5 else 0)
    decide0[i][1] = (1 if para2[i][0]>0.5 else 0)
    decide0[i][2] = (1 if para31[i][0]>0.5 else 0)
    decide1[i][0] = (1 if para1[i][1]>0.5 else 0)
    decide1[i][1] = (1 if para2[i][1]>0.5 else 0)
    decide1[i][2] = (1 if para31[i][1]>0.5 else 0)""" 
    decide0[i][0] = element_xgb[i][0]
    decide0[i][1] = element_lgb[i][0]
    decide0[i][2] = element_ksvm[i][0]
    decide0[i][3] = element_nn[i][0]
    decide1[i][0] = element_xgb[i][1]
    decide1[i][1] = element_lgb[i][1]
    decide1[i][2] = element_ksvm[i][1]
    decide1[i][3] = element_nn[i][1]
k0 = decide0.mean(axis=1)
k1 = decide1.mean(axis=1)

"""final = pd.DataFrame(np.zeros((len(k0), 1)))
for i in range(0,9144):
    if (k0[i]-k1[i])>0.05:
       final[0][i]=0
    else:  
       final[0][i]=1
       
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, final))"""
def get_acc(thresh, k0=k0,k1=k1):
    final = pd.DataFrame(np.zeros((len(k0), 1)))
    for i in range(0,len(k0)):
        if (k0[i]-k1[i])>thresh:
            final[0][i]=0
        else:
            final[0][i]=1
    return metrics.accuracy_score(y_test, final)
print(get_acc(0.095))
t = np.arange(0.0, 0.5, 0.001)  
s = np.empty([500,1])
i=0
maxi = 0.0
iind = 0
for item in t:
    h = get_acc(item)
    s[i] = h
    if h>maxi:
        maxi = h
        iind = i
    i = i+1
print(maxi)
print(iind)
score_testcv = get_acc(0.059)
plt.figure()
x = np.array([score_lsvm,score_ksvm, score_nn,score_xgb,  score_lgbm, score_testcv, maxi])
y = np.array(['LinearSVM','KernelSVM','Neural Network','XGBoost', 'LightGBM', 'Meta-Stacked (CV parameter)', 'Meta-Stacked (Test Parameter)'])
dfx = pd.DataFrame(x)
colors = sns.color_palette("cool", len(x))
ax= sns.barplot(y, x, data=dfx,palette=colors)
ax = sns.pointplot(y, x, color='cornflowerblue', scale=0.7)
ax.set(ylim=(0.935, 0.96))
plt.title("Accuracy of all algorithms")
plt.xticks(rotation= 90)
plt.ylabel('Accuracy')
plt.xlabel('Algorithms')
#adding the text labels
rects = ax.patches
labels = y

plt.show()

