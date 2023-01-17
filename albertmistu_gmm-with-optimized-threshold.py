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
df = pd.read_csv('../input/creditcard.csv')

from sklearn.model_selection import train_test_split

from matplotlib.colors import LogNorm

from sklearn import mixture



df_0=df[df.Class==0]    #Dataset with non-fraudulent only

df_1=df[df.Class==1]    #Dataset with fraudulent only



#Split non-fraudulent data in 90% for training GMM and 10% for cross-validation and testing

X_N_train, X_N_cv_test, y_N_train, y_N_cv_test = train_test_split(df_0.drop(['Class'],axis=1), df_0['Class'] , test_size=0.1, random_state=1)

#Split the fraudulent data in 50% for cross-validation and 50% for testing

X_F_cv, X_F_test, y_F_cv, y_F_test = train_test_split(df_1.drop(['Class'],axis=1), df_1['Class'] , test_size=0.5, random_state=1)

#Split the remaining 10% non-fraudulent in 50% for cross-validation and 50% for testing

X_N_cv, X_N_test, y_N_cv, y_N_test = train_test_split(X_N_cv_test, y_N_cv_test , test_size=0.5, random_state=1)



#Generate the 3 new datasets (Train + CV + test)

X_CV = np.vstack([X_N_cv, X_F_cv])

y_CV = np.hstack([y_N_cv, y_F_cv])

X_test = np.vstack([X_N_test, X_F_test])

y_test = np.hstack([y_N_test, y_F_test])



# Fit a Gaussian Mixture Model with the data from the NORMAL cases. *Note we are using ALL available features now.

clf = mixture.GaussianMixture()

clf.fit(X_N_train)
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score





kfold = StratifiedKFold(n_splits=5, random_state=1)                                   #Create 5-CV split object

T_vec=-np.arange(0,1000,2)          #Trying thresholds in steps of 2, from 0 to -1000. Note we are evaluating the negative log-likelihood.

aucpr_vs_t=[]

precision_vs_t=[]

recall_vs_t=[]

    

for t in T_vec:

    

    aucpr=[]

    precision=[]

    recall=[]

    k=0

    for train_index, test_index in kfold.split(X_CV,y_CV):

            

            y_cv_proba=clf.score_samples(X_CV[test_index])                                 #Predict the probabilities of fold "k" using the fitted GMM 

            y_cv_pred=y_cv_proba.copy()

            y_cv_pred[y_cv_pred>=t]=0

            y_cv_pred[y_cv_pred<t]=1   

            #print('Classification report')

            #print(classification_report(y_CV[test_index], y_cv_pred))

            precision.append(precision_score( y_CV[test_index], y_cv_pred))

            recall.append(recall_score( y_CV[test_index], y_cv_pred))

            aucpr.append(average_precision_score( y_CV[test_index], y_cv_pred))

            #print("Threshold T = %i --> Fold %i - aucpr=%.3f - Precision=%.3f - Recall=%.3f" %(t, k+1, aucpr[k], precision[k], recall[k]))

            k=k+1  

            

            

    aucpr_vs_t.append(np.mean(aucpr))

    precision_vs_t.append(np.mean(precision))

    recall_vs_t.append(np.mean(recall))

    #print('CV average AUCPR: %.3f +/- %.3f' % ( np.mean(aucpr), np.std(aucpr)))    

    #print('CV average precision: %.3f +/- %.3f' % ( np.mean(precision), np.std(precision)))    

    #print('CV average recall: %.3f +/- %.3f' % ( np.mean(recall), np.std(recall)))
import matplotlib.pyplot as plt

plt.plot(T_vec, aucpr_vs_t)

plt.plot(T_vec, precision_vs_t)

plt.plot(T_vec,recall_vs_t)

ax = plt.gca()

ax.set(title='Evolution of performance scores vs. threshold for GMM probability', xlabel='Threshold T [neg loglikelihood]')

ax.legend(['AUCPR (5 fold CV average)', 'Precision (5 fold CV average)', 'Recall (5 fold CV average)'])

ax.invert_xaxis()
print('Maximum cross validation AUCPR='+str(max(aucpr_vs_t)))

T_opt=T_vec[np.argmax(aucpr_vs_t)]

print('Optimal threshold T = '+str(T_opt))





y_test_proba = clf.score_samples(X_test)

y_test_pred=y_test_proba.copy()

y_test_pred[y_test_pred>=T_opt]=0

y_test_pred[y_test_pred<T_opt]=1   



test_precision = (precision_score( y_test, y_test_pred))

test_recall = (recall_score( y_test, y_test_pred))

test_aucpr = (average_precision_score( y_test, y_test_pred))

print("TEST results --> aucpr=%.3f - Precision=%.3f - Recall=%.3f" %(test_aucpr, test_precision, test_recall) )