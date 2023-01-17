import numpy as np

import pandas as pd
test_data = pd.read_csv('../input/minor-project-2020/test.csv')

test_data.columns
train_data = pd.read_csv('../input/minor-project-2020/train.csv')

train_data.columns
Xtest = test_data[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',

       'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13',

       'col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20',

       'col_21', 'col_22', 'col_23', 'col_24', 'col_25', 'col_26', 'col_27',

       'col_28', 'col_29', 'col_30', 'col_31', 'col_32', 'col_33', 'col_34',

       'col_35', 'col_36', 'col_37', 'col_38', 'col_39', 'col_40', 'col_41',

       'col_42', 'col_43', 'col_44', 'col_45', 'col_46', 'col_47', 'col_48',

       'col_49', 'col_50', 'col_51', 'col_52', 'col_53', 'col_54', 'col_55',

       'col_56', 'col_57', 'col_58', 'col_59', 'col_60', 'col_61', 'col_62',

       'col_63', 'col_64', 'col_65', 'col_66', 'col_67', 'col_68', 'col_69',

       'col_70', 'col_71', 'col_72', 'col_73', 'col_74', 'col_75', 'col_76',

       'col_77', 'col_78', 'col_79', 'col_80', 'col_81', 'col_82', 'col_83',

       'col_84', 'col_85', 'col_86', 'col_87']]
X = train_data[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',

       'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13',

       'col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20',

       'col_21', 'col_22', 'col_23', 'col_24', 'col_25', 'col_26', 'col_27',

       'col_28', 'col_29', 'col_30', 'col_31', 'col_32', 'col_33', 'col_34',

       'col_35', 'col_36', 'col_37', 'col_38', 'col_39', 'col_40', 'col_41',

       'col_42', 'col_43', 'col_44', 'col_45', 'col_46', 'col_47', 'col_48',

       'col_49', 'col_50', 'col_51', 'col_52', 'col_53', 'col_54', 'col_55',

       'col_56', 'col_57', 'col_58', 'col_59', 'col_60', 'col_61', 'col_62',

       'col_63', 'col_64', 'col_65', 'col_66', 'col_67', 'col_68', 'col_69',

       'col_70', 'col_71', 'col_72', 'col_73', 'col_74', 'col_75', 'col_76',

       'col_77', 'col_78', 'col_79', 'col_80', 'col_81', 'col_82', 'col_83',

       'col_84', 'col_85', 'col_86', 'col_87']]
X.head()
y = train_data[['target']]
y.head()
target_count = train_data.target.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])
#trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2,stratify=y)
import imblearn
from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler()

trainX_1, trainy_1 = rus.fit_sample(X, y)



#print('Removed indexes:', id_rus)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
trainX, testX, trainy, testy = train_test_split(trainX_1, trainy_1, test_size=0.3, random_state=2,stratify=trainy_1)
from sklearn.model_selection import GridSearchCV
# model = LogisticRegression()

# grid = {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2']}

# model_cv = GridSearchCV(model,grid,cv=10)

# model_cv.fit(trainX, trainy)
model = LogisticRegression(C=1000.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX,trainy.values.ravel())
y_pred = model.predict(testX)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=testy, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
yhat = model.predict_proba(testX)

pos_probs = yhat[:, 1]

pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

fpr, tpr, _ = roc_curve(testy, pos_probs)

pyplot.plot(fpr, tpr, marker='.', label='Logistic')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.legend('roc-auc curve')

pyplot.show()
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(testy, pos_probs)

print(roc_auc)
model = LogisticRegression(C=1000.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX_1,trainy_1.values.ravel())
c = model.predict(Xtest)
d = model.predict_proba(Xtest)
list1 = []

for i in range(0,len(c)):

  id1 = test_data.loc[[i][:]]

  id1 = id1['id']

  id1 = id1[i]

  #print(id1)

  target1 = d[i][1]

  dict1 = {'id':(int)(id1),'target':(target1)}

  list1.append(dict1)
import csv
csvfile=open('log_reg_oversampling_prob_final.csv','w', newline='')

fields=list(list1[0].keys())

obj=csv.DictWriter(csvfile, fieldnames=fields)

obj.writeheader()

obj.writerows(list1)

csvfile.close()
from imblearn.over_sampling import SMOTE



smote = SMOTE(sampling_strategy='minority')

trainX_1, trainy_1 = smote.fit_sample(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
trainX, testX, trainy, testy = train_test_split(trainX_1, trainy_1, test_size=0.3, random_state=2,stratify=trainy_1)
from sklearn.model_selection import GridSearchCV
# model = LogisticRegression()

# grid = {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2']}

# model_cv = GridSearchCV(model,grid,cv=10)

# model_cv.fit(trainX, trainy)
model = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX,trainy)
y_pred = model.predict(testX)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=testy, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
yhat = model.predict_proba(testX)

pos_probs = yhat[:, 1]

pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

fpr, tpr, _ = roc_curve(testy, pos_probs)

pyplot.plot(fpr, tpr, marker='.', label='Logistic')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.legend('roc-auc curve')

pyplot.show()
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(testy, pos_probs)

print(roc_auc)
model = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX_1,trainy_1)
c = model.predict(Xtest)
d = model.predict_proba(Xtest)
list1 = []

for i in range(0,len(d)):

  id1 = test_data.loc[[i][:]]

  id1 = id1['id']

  id1 = id1[i]

  #print(id1)

  target1 = d[i][1]

  dict1 = {'id':(int)(id1),'target':(target1)}

  list1.append(dict1)
# import csv
# csvfile=open('log_reg_oversampling_SMOTE_prob_final.csv','w', newline='')

# fields=list(list1[0].keys())

# obj=csv.DictWriter(csvfile, fieldnames=fields)

# obj.writeheader()

# obj.writerows(list1)

# csvfile.close()
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler()

trainX_1, trainy_1 = rus.fit_sample(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from matplotlib import pyplot
trainX, testX, trainy, testy = train_test_split(trainX_1, trainy_1, test_size=0.3, random_state=2,stratify=trainy_1)
from sklearn.model_selection import GridSearchCV
# model = LogisticRegression()

# grid = {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2']}

# model_cv = GridSearchCV(model,grid,cv=10)

# model_cv.fit(trainX, trainy)
model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX,trainy)
y_pred = model.predict(testX)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=testy, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
yhat = model.predict_proba(testX)

pos_probs = yhat[:, 1]

pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

fpr, tpr, _ = roc_curve(testy, pos_probs)

pyplot.plot(fpr, tpr, marker='.', label='Logistic')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.legend('roc-auc curve')

pyplot.show()
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(testy, pos_probs)

print(roc_auc)
model = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)
model.fit(trainX_1,trainy_1)
c = model.predict(Xtest)
d = model.predict_proba(Xtest)
list1 = []

for i in range(0,len(d)):

  id1 = test_data.loc[[i][:]]

  id1 = id1['id']

  id1 = id1[i]

  #print(id1)

  target1 = d[i][1]

  dict1 = {'id':(int)(id1),'target':(target1)}

  list1.append(dict1)
# import csv
# csvfile=open('log_reg_undersampling_final.csv','w', newline='')

# fields=list(list1[0].keys())

# obj=csv.DictWriter(csvfile, fieldnames=fields)

# obj.writeheader()

# obj.writerows(list1)

# csvfile.close()