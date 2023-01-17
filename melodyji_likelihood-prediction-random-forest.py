import numpy as np

import pandas as pd

import itertools

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import confusion_matrix,recall_score,classification_report

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

import sklearn.metrics as sm

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import BorderlineSMOTE

from sklearn.model_selection import train_test_split

from collections import Counter

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)



# this method plots the confusion matrix

def plot_cm(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True/False')

    plt.xlabel('Predict')





def predict_and_plot(model, X, Y, name = ''):

    y_pred = model.predict(X.values)

    cm = confusion_matrix(Y,y_pred)

    np.set_printoptions(precision=2)



    # Recall = TP/(TP+FN)

    print("Recall by using " + name + " : ", cm[1,1]/(cm[1,0]+cm[1,1]))

    # Precision = TP/(TP+FP)

    print("Precision by using " + name + " : ", cm[1,1]/(cm[0,1]+cm[1,1]))



    class_names = [0,1]

    plt.figure()

    plot_cm(cm, classes=class_names,title='Confusion Matrix')

    plt.show()

    

def plot_roc(model, probs, X_test, Y_test, name):

    roc_auc = roc_auc_score(Y_test, model.predict(X_test))

    fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])

    plt.figure()

    plt.title(name)

    plt.plot(fpr, tpr, label= name + ' (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(name + ' ROC')

    plt.legend(loc="lower right")

    plt.show()

    

def plot_precision_recall(model, X_test, Y_test, name):

    probs = model.predict_proba(X_test)

    # keep probabilities for the positive outcome only

    probs = probs[:, 1]

    # predict class values

    yhat = model.predict(X_test)

    _precision, _recall, _ = precision_recall_curve(Y_test, probs)

    _f1, _auc = f1_score(Y_test, yhat), auc(_recall, _precision)

    # summarize scores

    print(name + ': f1=%.3f auc=%.3f' % (_f1, _auc))

    # plot the precision-recall curves

    no_skill = len(Y_test[Y_test==1]) / len(Y_test)

    pyplot.plot(_recall, _precision, marker='.', label=name)

    # axis labels

    pyplot.xlabel('Recall')

    pyplot.ylabel('Precision')

    pyplot.title(name + ": Precision Recall Curve")

    # show the legend

    pyplot.legend()

    # show the plot

    pyplot.show()

    

def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

        

def print_precision_recall_score(model, X_test, Y_test, name):

    probs = model.predict_proba(X_test)

    # keep probabilities for the positive outcome only

    probs = probs[:, 1]

    # predict class values

    yhat = model.predict(X_test)

    _precision, _recall, _ = precision_recall_curve(Y_test, probs)

    _f1, _auc = f1_score(Y_test, yhat), auc(_recall, _precision)

    # summarize scores

    print(name + ': f1=%.3f auc=%.3f' % (_f1, _auc))

    

def get_precision_recall_score(model, X_test, Y_test):

    probs = model.predict_proba(X_test)

    # keep probabilities for the positive outcome only

    probs = probs[:, 1]

    # predict class values

    yhat = model.predict(X_test)

    _precision, _recall, _ = precision_recall_curve(Y_test, probs)

    _f1, _auc = f1_score(Y_test, yhat), auc(_recall, _precision)

    # summarize scores

    return _auc

    
df = pd.read_csv("/kaggle/input/sgms-data/ForLikelihood.csv")

df = df.fillna(df.mean())

columns_tokeep = [

    'tic',

    'cshtr_c', 'chech_sd','dp_sd','sale_sd','xad_sd','xsga_sd',

#      'splticrm',

     'emp_sd','mrcta_sd','ebit_sd',

#     'spce_sd','intan_sd',

#                  'xad','optosey','mkvalt_sd','xacc',

#                   'xsga', 'ch_sd','aco_sum','intano_sd','capx_sd',

# #                   'capx','ch','at_sd','dvpsp_f_sd','prcc_c_sd',

#                   'state',

                  'isSued']

df1 = df.loc[:, columns_tokeep]
dfSGMS = df1[df1.tic == 'SGMS']

dfSGMS = dfSGMS.drop(['tic','isSued'], axis=1)
df1 = df1[df1.tic != 'SGMS']

df1 = df1.drop(['tic'], axis=1)

df1
import matplotlib.pyplot as plt

count=pd.value_counts(df1['isSued'],sort=True).sort_index()

count
ax=count.plot(kind='bar')

plt.title('Sued Distribution')

plt.ylabel('Count')

plt.show()
for c in df1.columns.tolist():

    if c != 'isSued' and c!= 'state':

        cols_to_norm = [c]

        cols_new = [c]

        df1[cols_new] = StandardScaler().fit_transform(df1[cols_to_norm])
X = df1.iloc[:, df1.columns != 'isSued']

Y = df1.iloc[:, df1.columns == 'isSued']



# train/test for the original dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



oversampler=SMOTE()

X_O_train,Y_O_train=oversampler.fit_sample(X_train,Y_train['isSued'])





print(Counter(Y_O_train))
parameters = {

    'n_estimators': [5, 10, 50, 250, 1000,2000],

    'max_depth': [2, 4, 8, 16, 32, None]

}

for n in parameters['n_estimators']:

    for m in parameters['max_depth']:

        

        rf = RandomForestClassifier(n_estimators=n, max_depth=m)

        rf.fit(X_O_train, Y_O_train.values.ravel())

        rf_pred = rf.predict(X_test)

        predict_prob = rf.predict_proba(X_test)[:, 1]

        _auc = get_precision_recall_score(rf, X_test, Y_test)

        print('-----------n_estimators: '+ str(n) +'---------max_depth: '+ str(m)+'-----auc '+ str(_auc) +'-----------------')
import time

start = time.time()

rf = RandomForestClassifier(n_estimators=1000, max_depth=None)

# log how long for training

rf.fit(X_O_train, Y_O_train.values.ravel())

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))





rf_pred = rf.predict(X_test)

predict_and_plot(rf, X_test, Y_test, "Random Forest")

predict_prob = rf.predict_proba(X_test)[:, 1]

# print_precision_recall_score(rf, X_test, Y_test, 'Random Forest')

plot_precision_recall(rf, X_test, Y_test, 'Random Forest')

plot_roc(rf,predict_prob, X_test, Y_test, 'Random Forest')

sgms_rf_prob = rf.predict_proba(dfSGMS)[:, 1]

sgms_rf_prob
total = 0

loop = 100

for i in range(loop): 

    

    rf = RandomForestClassifier(n_estimators=250, max_depth=None)

    rf.fit(X_O_train, Y_O_train.values.ravel())

    rf_pred = rf.predict(X_test)

    predict_prob = rf.predict_proba(X_test)[:, 1]

    sgms_rf_prob = rf.predict_proba(dfSGMS)[:, 1]

    total +=sgms_rf_prob[0]

    _auc = get_precision_recall_score(rf, X_test, Y_test)

    print('----------------auc '+ str(_auc) +'-------likelihood: '+str(sgms_rf_prob[0])+'----------')

    

print ('avg likelihood:' + str(total/loop))