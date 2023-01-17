# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from sklearn import svm

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,roc_auc_score, precision_score, roc_curve, recall_score,\

                             classification_report, f1_score, precision_recall_fscore_support)

from sklearn.model_selection import StratifiedShuffleSplit

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.combine import SMOTETomek

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import sklearn.ensemble as ensemble

from keras.models import Model

from keras.layers import Input, Dense

from keras import regularizers

import seaborn as sns
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

print(data.shape)

data.head()
data.describe()
data.isna().sum().max()
plt.subplots(figsize = (7,5))

count_classes = pd.value_counts(data['Class'],sort=True).sort_index()

count_classes.plot(kind = 'bar')

plt.title('Fraud class histogram', fontsize = 13)

plt.xlabel('Class', fontsize = 13)

plt.xticks(rotation=0)

plt.ylabel('Frequency', fontsize = 15)

plt.show()
data.hist(figsize=(20,20))

plt.show()
X_train = data.iloc[:,1:-2]
clf = IsolationForest(contamination = 0.03,n_estimators = 100, max_samples = 0.6, max_features = 0.6,random_state = 42)

clf.fit(X_train)
y_pred_1IF = pd.Series(clf.predict(X_train))

y_pred_1IF.replace(1,0,inplace = True)

y_pred_1IF.replace(-1,1,inplace = True)

cross_table = pd.crosstab(data.Class, columns = y_pred_1IF)

cross_table
OCSVM = svm.OneClassSVM(nu = 0.03,kernel = 'rbf')

OCSVM.fit(X_train)
y_pred_2OCSVM = pd.Series(OCSVM.predict(X_train))

y_pred_2OCSVM.replace(1,0,inplace = True)

y_pred_2OCSVM.replace(-1,1,inplace = True)

cross_table = pd.crosstab(data.Class, columns = y_pred_2OCSVM)

cross_table
input_dim = X_train.shape[1]

encoding_dim = 14

input_layer = Input(shape = (input_dim, ))



encoder = Dense(14, activation = 'tanh', activity_regularizer = regularizers.l1(10e-5))(input_layer)

encoder = Dense(input_dim,activation = 'relu')(encoder)

autoencoder = Model(inputs = input_layer, outputs = encoder)



nb_epoch = 10

batch_size = 32

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

autoencoder.summary()
x_train = np.array(X_train)

history = autoencoder.fit(x_train,x_train,epochs = nb_epoch,batch_size = batch_size,shuffle=True,validation_data = (x_train,x_train),verbose = 1).history
y_pred_AT = autoencoder.predict(x_train)

# Restoring Error

mse = np.mean(np.power(x_train - y_pred_AT, 2), axis = 1)

error_df = pd.DataFrame({'reconstruction_error':mse,

                         'true_class':data.Class})
fpr, tpr, thresholds = roc_curve(error_df.true_class,error_df.reconstruction_error)

roc_auc = auc(fpr, tpr)



plt.subplots(figsize = (7,5))

plt.plot(fpr, tpr, label = 'AUC = {:.4f}'.format(roc_auc))

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate',fontsize=13)

plt.xlabel('False Positive Rate',fontsize=13)

plt.title('Receiver Operating Characteristic',fontsize=13)

plt.show()
i = np.arange(len(tpr))

cop = pd.DataFrame(

    {'fpr' : pd.Series(fpr, index = i),

     'tpr' : pd.Series(tpr, index = i),

     '1-fpr' :pd.Series(1-fpr, index = i),

     'tf' : pd.Series(tpr - (1-fpr), index = i),

     'thresholds' : pd.Series(thresholds, index = i)})
fig = plt.figure(figsize = (7,5))

ax1 = fig.add_subplot(111)

ax1.plot(cop['thresholds'], cop['tpr'], 'blue')

ax1.set_ylabel('TPR(blue)',fontsize = 13)

ax1.set_title('Threshold determination curve', fontsize = 13)



ax2 = ax1.twinx()

ax2.plot(cop['thresholds'], cop['1-fpr'], 'red')

ax2.set_xlim([0,4])

ax2.set_ylabel('TNR=1-fpr(red)',fontsize = 13)

ax2.set_xlabel('thresholds',fontsize = 13)

plt.show()
# look up the smallest value

cop.loc[(cop.tf-0).abs().argsort()[:1]]
# Establish confusion matrix

threshold = 3

y_pred_3AR = np.array([1 if e>threshold else 0 for e in error_df.reconstruction_error.values])

cross_table = pd.crosstab(data.Class, columns = y_pred_3AR)

cross_table
three_score = pd.DataFrame([y_pred_1IF,y_pred_2OCSVM,y_pred_3AR]).T

y_infer = three_score.apply(lambda x: x.mode(), axis = 1)

data['Class_infer'] = y_infer

cross_table = pd.crosstab(data.Class, columns = data.Class_infer)

cross_table
label = {

    (0,0):0,

    (1,1):1,

    (0,1):1,

    (1,0):0

}

data['Class_new'] = data[['Class','Class_infer']].apply(lambda x:label[(x[0],x[1])], axis = 1)

count_classes = pd.value_counts(data['Class_new'], sort = True).sort_index()

count_classes
data = data.drop(['Time','Class','Class_infer'], axis = 1)

X = np.array(data.loc[:,:'V28'])

y = np.array(data['Class_new'])

sess = StratifiedShuffleSplit(n_splits = 5,test_size=0.4,random_state=0)

for train_index,test_index in sess.split(X,y):

    X_train,X_test = X[train_index], X[test_index]

    y_train,y_test = y[train_index], y[test_index]

print('train_size: %s' %len(y_train),

     'test_size: %s' %len(y_test))
plt.figure(figsize = (7,5))

count_classes = pd.value_counts(y_train, sort = True)

count_classes.plot(kind = 'bar')

plt.title('The histogram of fraud class in trainingdata', fontsize = 13)

plt.xlabel('Class', fontsize = 13)

plt.ylabel('Frequency', fontsize = 13)

plt.xticks(rotation=0)

plt.show() 
ros = RandomOverSampler(random_state = 0)

sos = SMOTE(random_state=0)

kos = SMOTETomek(random_state=0)



x_ros, y_ros = ros.fit_sample(X_train, y_train)

x_sos, y_sos = sos.fit_sample(X_train, y_train)

x_kos, y_kos = kos.fit_sample(X_train, y_train)

print('ros: {}, sos: {}, kos:{}'.format(len(y_ros),len(y_sos),len(y_kos)))
y_ros.sum(), y_sos.sum(), y_kos.sum()
clf = DecisionTreeClassifier(criterion = 'gini', random_state=1234)

param_grid = {'max_depth':[3, 4, 5, 6], 'max_leaf_nodes':[4, 6, 8, 10, 12]}

cv = GridSearchCV(clf, param_grid  = param_grid, scoring = 'f1')
data = [[X_train, y_train],

        [x_ros, y_ros],

        [x_sos, y_sos],

        [x_kos, y_kos]]



for features, labels in data:

    cv.fit(features, labels)

    predict_test = cv.predict(X_test)

    

    print('auc:{:.3f}'.format(roc_auc_score(y_test, predict_test)),

          'recall:{:.3f}'.format(recall_score(y_test, predict_test)),

          'precision:{:.3f}'.format(precision_score(y_test, predict_test)))
train_data = x_ros

train_target = y_ros

test_target = y_test

test_data = X_test
lr = LogisticRegression(C = 0.1,penalty = 'l1',solver='liblinear')

lr.fit(train_data,train_target)

test_est = lr.predict(test_data)

print('Logistic Regression accuracy:')

print(classification_report(test_target,test_est))

fpr_test, tpr_test, th_test = roc_curve(test_target,test_est)

print('Logistic Regression AUC:{:.4f}'.format(auc(fpr_test,tpr_test)))