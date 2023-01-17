# import libraries



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sb

from matplotlib import rcParams
%matplotlib inline

rcParams['figure.figsize'] = 5, 4

sb.set_style('whitegrid')
#import the data file



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
# take a look at 10 head of data file

data = pd.read_csv("../input/creditcard.csv")

data.head(10)
# get insight what's inside of the data file

data.info()
data['Hour'] = data['Time'].apply(lambda x: np.round(float(x)/3600))

del data["Time"]

data.pivot_table(values='Amount',index='Hour',columns='Class',aggfunc='count')
plt.figure(figsize=(12,2*2))

Fraud = data[data["Class"]==1]

Not_Fraud= data[data["Class"]==0]

color = ['salmon']

plt.subplot(121)

Fraud.Amount.plot.hist(title="Fraud Transacation", color=color)

plt.xlabel('Amount'); plt.ylabel('Number of Frauds');

plt.subplot(122)

Not_Fraud.Amount.plot.hist(title="Not_Fraud Transactions")

plt.xlabel('Amount'); plt.ylabel('Number of Norn_Fraud Transactions');
TotalFraud=data['Class'].value_counts()[1]

TotalNonFraud=data['Class'].value_counts()[0]

Num= TotalFraud/(TotalFraud+TotalNonFraud)

Percentage=Num*100

print(Percentage)
import matplotlib.gridspec as gridspec

from scipy import stats



plt.figure(figsize=(12,28*5))

Tabfeatures = data.ix[:,1:29].columns

f= gridspec.GridSpec(28, 1)

for i, cn in enumerate(data[Tabfeatures]):

    ax = plt.subplot(f[i])

    sb.distplot(data[cn][data.Class == 1], kde=False, fit=stats.norm)

    sb.distplot(data[cn][data.Class == 0], kde=False, fit=stats.norm)

    ax.set_xlabel('')

    ax.set_title('feature hist: ' + str(cn))

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle





#features = pd.concat([data.loc[:,'V1':'Amount'],data.loc[:,'Time']],axis=1)

#target = data['Class']



    

#X_train, X_test, y_train, y_test = train_test_split(features,target, stratify=target,test_size = 0.3, random_state = 0)



X=data.iloc[:,:-1]

Y=data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)



#randomizing the data

X_train = shuffle(X_train)

X_test = shuffle(X_test)



print("# of train dataset in class 1 and 0 : ", len(X_train))

print("# of test dataset in class 1 and 0: ", len(X_test))

print("Total # of transactionsin class 1 and 0: ", len(X_train)+len(X_test))

print('-------------------------------------------------------')

print('y_train values')

print(y_train.value_counts())

print('')

print('y_test values')

print(y_test.value_counts())

print('-------------------------------------------------------')

print ("(X train shape %s, X test shape %s, \ny train shape %s, y test shape %s"% (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

from sklearn.linear_model import LogisticRegression



RegModel = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)



RegModel.fit(X_train,y_train)

predicted = RegModel.predict(X_test)



print (predicted)

from itertools import cycle

from sklearn.metrics import precision_recall_curve



thresholds=np.linspace(1000, 8000, 1000)

#thresholds = [0.1,0.2,0.3,0.4,0.5]

colors = cycle(['magenta', 'blue', 'darkorange', 'cyan', 'green', ' black'])



plt.figure(figsize=(10,5))



for i,color in zip(thresholds,colors):

    predictedD = predicted[:,1] > i

    

    precision, recall, thresholds = precision_recall_curve(y_test,predictedD)

    

    # Plot Precision-Recall curve

    plt.plot(recall, precision, color=color,label='Threshold: %s'%i)

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall example')

    plt.legend(loc="lower left")
from sklearn.metrics import roc_curve,auc, roc_auc_score, classification_report

from sklearn.metrics import confusion_matrix

    

FPR, TPR, thresholds = roc_curve(y_test, predicted)

roc_auc = auc(FPR, TPR)

plt.plot(FPR, TPR, color='blue')

plt.xlim([0, 1])

plt.ylim([0, 1])



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc='best')

plt.show()



print('roc_auc_score: %0.3f'% roc_auc_score(y_test, predicted))

print("----------------------------------------------------------------------")

print("Logistic Regression report \n",classification_report(y_test, predicted))

print("----------------------------------------------------------------------")

print("Logistic Regression confusion matrix \n",confusion_matrix(y_test, predicted))