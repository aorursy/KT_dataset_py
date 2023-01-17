# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



#AUTHOR RONALD TURYATEMBA

#USING GRADIENT BOOSTING CLASSIFIER TO PREDICT WHETHER FRAUD OR NOT 

#USING THE DATA UPON WHICH PRINCIPAL COMPONENT ANALYSIS HAS BEEN DONE 

#SO WE SHALL USE THESE LINEAR UNCORRELATED DATA

import pandas as pd

import numpy as np

df_credit = pd.read_csv('../input/creditcard.csv',header=0)

df_credit

x =df_credit.drop(['Class'], axis=1)

y = df_credit['Class']







from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn.cross_validation import KFold,cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingClassifier



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)

Kfold =KFold(len(df_credit),n_folds=100, shuffle=False)



#using the gradient boosting algorithm

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,random_state=0)

GBC = GBC.fit(x,y)

#GBC = GBC.predict([....])

print('calculating...')

print('The GBC accuracy score is %s' %cross_val_score(GBC,x,y,cv=10).mean())

print('done')
#AUTHOR RONALD TURYATEMBA

#USING ROC_AUC TO MEASURE PERFORMANCE OF GRADIENT BOOSTING CLASSIFIER 

#USING THE DATA UPON WHICH PRINCIPAL COMPONENT ANALYSIS HAS BEEN DONE 



import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn.cross_validation import KFold,cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

print('started')

df_credit = pd.read_csv('../input/creditcard.csv',header=0)

df_credit

x =df_credit.drop(['Class'], axis=1)

y = df_credit['Class']



#binarize the class data that is y

y_array = np.array(y)

y = label_binarize(y_array, classes=[0,1])

#this will count the number of elements in row

#n_classses = y.shape[1]

n_classes = 2



#shuffle and slipt the the data into training and test sets

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)





#using the gradient boosting algorithm

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=2.0, max_depth=1,random_state=0, presort='TRUE')



#train the classifier with the train data

y_score = GBC.fit(X_train,y_train).predict_proba(X_test)



fpr = dict()

tpr = dict()

roc_auc = dict()

#a dictionary is initialised for the false positive rate and true positive rate

#plus the roc_ area under the curve 

#fpr is the false positve rate and tpr is the true positive rate



fpr[0],tpr[0],_ = roc_curve(y_test[:,0], y_score[:,0])

#we have an underscore there because this functions returns 3 values 

#but we dont need the third value therefore place a an underscore there



roc_auc[0] = auc(fpr[0],tpr[0])

#plot the figure

plt.figure()

lw=2

plt.plot(fpr[0],tpr[0], color='darkorange',lw=lw,label='ROC curve (area=%0.2f)' %roc_auc[0])

plt.plot([0,1],[0,1], color='grey', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC All_state')

plt.legend(loc='lower right')

plt.show()



print('done')
