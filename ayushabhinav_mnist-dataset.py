# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../"))

print(os.listdir())



# os.makedirs('work_file')



# Any results you write to the current directory are saved as output.
import os





INPUT_DIR = os.path.join(os.path.split(os.getcwd())[0], 'input'  )



train_file = os.path.join(INPUT_DIR, 'train.csv')

train_data = pd.read_csv(train_file)



print('Train Data Shape :',train_data.shape)

print('Train Data Head :--->')

train_data.head()



    

    
X_train = train_data.drop('label', axis=1)

y_train = train_data['label']

X_train

y_train  
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
a = np.array(X_train[:1])

a.reshape(28, 28)
def display_image(pxl_mtx):

    pxl_mat_reshaped = np.array(pxl_mtx).reshape(28,28)

    plt.imshow(pxl_mat_reshaped)

    plt.show()

    

display_image(X_train[1:2])

    
test_array = np.array(X_train)

display_image(test_array[0])

y_5_train = np.array(y_train == 5 )

pd.value_counts(y_5_train)
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict



sgdc = SGDClassifier(random_state = 42)

sgdc1 = SGDClassifier(random_state = 42)



y_5_predict = cross_val_predict(sgdc, X_train, y_5_train, cv=5, n_jobs=3, verbose=3, method='predict')

y_5_score = cross_val_predict(sgdc1, X_train, y_5_train, cv=5, n_jobs=3, verbose=3, method='decision_function')





print(pd.value_counts(y_5_predict))

print(y_5_score)

from sklearn.metrics import precision_recall_curve

p, r, t = precision_recall_curve(y_5_train, y_5_score)

print(p)

print(r)

print(t)
def plot_precision_recall_curve(p, r, t):

    fig, axis = plt.subplots(nrows=1,ncols=1) 

    axis.plot(t,p[:-1])

    axis.plot(t,r[:-1])

    axis.legend(['precision', 'recall'])



plot_precision_recall_curve(p, r, t)

from sklearn.metrics import precision_score, recall_score

p_score = precision_score(y_5_train, y_5_predict)

r_score = recall_score(y_5_train, y_5_predict)

print('Precision  Score: {}'.format(p_score))

print('Recall  Score: {}'.format(r_score))
def plot_roc_curve(res_list):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for res in res_list:

        ax.plot(res[0], res[1])

    ax.legend(list(range(1,len(res_list) + 1)))
from sklearn.metrics import roc_curve



fpr, tpr, th = roc_curve(y_5_train, y_5_score)

plot_roc_curve([[fpr, tpr, th]])
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'loss' : ('hinge', 'log'),

     'penalty' :('l1', 'l2'),

     'alpha' : (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

    }

]

    



model_search = GridSearchCV(sgdc, param_grid, cv=5, scoring='precision')

model_search.fit(X_train, y_5_train)

    







model_search.best_estimator_
y_5_predict1 = model_search.best_estimator_.predict(X_train)
p_score = precision_score(y_5_train, y_5_predict1)

r_score = recall_score(y_5_train, y_5_predict1)

print('Precision Score : {}'.format(p_score))

print('Recall Score : {}'.format(r_score))
score = cross_val_predict(model_search.best_estimator_, X_train, y_5_train, cv=5, verbose=3, method='decision_function')
p1, r1, t1 = precision_recall_curve(y_5_train, score)

plot_precision_recall_curve(p1, r1, t1)

plot_precision_recall_curve(p, r, t)

print(t1.shape)
from sklearn.metrics import roc_curve



fpr1, tpr1, th1 = roc_curve(y_5_train, score)

from matplotlib import figure



    

   

plot_roc_curve([[fpr, tpr, th], [fpr1, tpr1, th1]])

from sklearn.ensemble import RandomForestClassifier



param_grid = [

    {

        'max_depth' : [10,30],

        'n_estimators': [10,30],

    }

]





rfc = RandomForestClassifier(random_state=42)

model_search = GridSearchCV(rfc, param_grid, cv=3, verbose=3, n_jobs=5, scoring='precision')

model_search.fit(X_train, y_5_train)









model_search.best_estimator_
scores = cross_val_predict(model_search.best_estimator_, X_train, y_5_train, cv=3, method='predict_proba')
scores[:,1]
y_5_predict2 = model_search.best_estimator_.predict(X_train)



p_score = precision_score(y_5_train, y_5_predict2)

r_score = recall_score(y_5_train, y_5_predict2)



print('Precision Score : {}'.format(p_score))

print('Recall Score : {}'.format(r_score))

p3, r3, t3 = precision_recall_curve(y_5_train, scores[:,1])

plot_precision_recall_curve(p3, r3, t3)
fpr2, tpr2 , th2 = roc_curve(y_5_train, y_5_predict2)
plot_roc_curve([

    [fpr, tpr, th ],

    [fpr1, tpr1, th1 ],

    [fpr2, tpr2, th2 ]

])
rfc.fit(X_train, y_5_train)

y_5_predict3 = rfc.predict(X_train)



fpr3, tpr3 , th3 = roc_curve(y_5_train, y_5_predict3)





plot_roc_curve([

    [fpr, tpr, th ],

    [fpr1, tpr1, th1 ],

    [fpr2, tpr2, th2 ],

    [fpr3, tpr3, th3 ],

])





scores = cross_val_predict(rfc, X_train, y_5_train, cv = 3, n_jobs=3, verbose= 3, method='predict_proba')
p,r,t =precision_recall_curve(y_5_train, scores[:,1])

plot_precision_recall_curve(p,r, t)
print('Precison Score : {}'.format(precision_score(y_5_train, y_5_predict3)))

print('Recall Score : {}'.format(recall_score(y_5_train, y_5_predict3)))
from sklearn.metrics import confusion_matrix



confusion_matrix(y_5_train, y_5_predict3)
from sklearn.metrics import f1_score

print('F1 Score for 1: {}'.format(f1_score(y_5_train, y_5_predict)))

print('F1 Score for 2: {}'.format(f1_score(y_5_train, y_5_predict1)))

print('F1 Score for 3: {}'.format(f1_score(y_5_train, y_5_predict2)))

print('F1 Score for 4: {}'.format(f1_score(y_5_train, y_5_predict3)))
test_file = os.path.join(INPUT_DIR, 'test.csv')

test_data = pd.read_csv(test_file)
y_5_test_predict = rfc.predict(test_data)
pd.value_counts(y_5_test_predict)
X_test = np.array(test_data)

display_image(X_test[0])

y_5_test_predict[0]
# def get_true_index(y):

#     idx = list()

#     for i, data in enumerate(y):

#         if data:

#             idx.append(i)

#     return idx



      

# true_index = get_true_index(y_5_test_predict)

# true_index
# for i in true_index:

#     display_image(X_test[i])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_curve, roc_curve

from sklearn.model_selection import cross_val_predict
X_my_train = X_train[:40000]

y_my_train = y_train[:40000]



X_my_test = X_train[40000:]

y_my_test = y_train[40000:]
rfc_multi_num = RandomForestClassifier(random_state=42,n_estimators=100)

rfc_multi_num.fit(X_my_train, y_my_train)
scores_multi_num = rfc_multi_num.predict_proba(X_my_train)
scores_multi_num[:10]

x_my_train_array = np.array(X_my_train)
display_image(x_my_train_array[8])
y_predict_multi_num = rfc_multi_num.predict(X_my_train)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_my_train, y_predict_multi_num)
y_pre_multi_test = rfc_multi_num.predict(X_my_test)
conf_mat=confusion_matrix(y_my_test, y_pre_multi_test)

conf_mat
plt.matshow(conf_mat, cmap=plt.cm.gray)
individual_num_count = conf_mat.sum(axis=1, keepdims=True)

individual_num_count

error_ratio_mat = conf_mat/ individual_num_count

error_ratio_mat
np.fill_diagonal(error_ratio_mat, 0)

plt.matshow(error_ratio_mat, cmap=plt.cm.gray)