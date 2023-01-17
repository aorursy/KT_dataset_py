import os
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
########## Import data and preprocess
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
# drop bad data
df.dropna()
df.isnull().sum()

# clean unwanted data
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# Convert Malignant diagnosis into 1 and Benign into 0
df['diagnosis'].astype(str)
df['diagnosis'] = [1 if val == 'M' else 0 for val in df['diagnosis'].values ]
########## Select data and labels
x = df.iloc[:,2:len(df.columns)]
y = df['diagnosis'].values
########## Data train test split
(x_train,x_test,y_train,y_test) = train_test_split(x, y, test_size=0.3)
########## Data scaling and normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) #fit is done for train set after split to prevent leak
x_test = scaler.transform(x_test) # Using the same fit as in train 
########## Initialize containers
SVM1AccTrain = []
SVM1AccTest = []
SVM1Precision = []
SVM1Recall = []

########## SVM1 with linear kernel
for i in range (0, 20):
    clf = SVC(kernel='linear', C=10**(10))
    clf.fit(x_train, y_train)

    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)
    
    SVM1AccTrain.append(accuracy_score(y_train, y_train_pred))
    SVM1AccTest.append(accuracy_score(y_test, y_test_pred))
    SVM1Precision.append(precision_score(y_test, y_test_pred))
    SVM1Recall.append(recall_score(y_test, y_test_pred))
########## Print average stats of SVM1
print ('''
        Average Statistics for SVM1
        \n SVM1AccTrainAvg = {}
        \n SVM1AccTestAvg = {}
        \n SVM1PrecisionAvg = {}
        \n SVM1RecallAvg = {}
        '''.format(
        np.mean(SVM1AccTrain),
        np.mean(SVM1AccTest),
        np.mean(SVM1Precision),
        np.mean(SVM1Recall)
        ))
########## Initialize containers
SVM2AccTrain = []
SVM2AccTest = []
SVM2Precision = []
SVM2Recall = []

########## SVM2 with RBF kernel
for i in range (0, 20):
    clf = SVC(kernel='rbf', C=10**(10))
    clf.fit(x_train, y_train)

    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)
    
    SVM2AccTrain.append(accuracy_score(y_train, y_train_pred))
    SVM2AccTest.append(accuracy_score(y_test, y_test_pred))
    SVM2Precision.append(precision_score(y_test, y_test_pred))
    SVM2Recall.append(recall_score(y_test, y_test_pred))
########## Print average stats of SVM2
print ('''
        Average Statistics for SVM2
        \n SVM2AccTrainAvg = {}
        \n SVM2AccTestAvg = {}
        \n SVM2PrecisionAvg = {}
        \n SVM2RecallAvg = {}
        '''.format(
        np.mean(SVM2AccTrain),
        np.mean(SVM2AccTest),
        np.mean(SVM2Precision),
        np.mean(SVM2Recall)
        ))
CList = [10**(n) for n in range (-30,30)]
########## Initialize containers
SVM3MeanAccTrainList = []
SVM3MeanAccTestList = []
SVM3MeanPrecisionList = []
SVM3MeanRecallList = []

########## SVM3 RBF kernel with regularization (soft margin)
for CVal_i, CVal in enumerate(CList):
    SVM3AccTrain = []
    SVM3AccTest = []
    SVM3Precision = []
    SVM3Recall = []
    for i in range(0, 20):
        clf = SVC(kernel='linear', C=CVal)
        clf.fit(x_train, y_train)

        y_test_pred = clf.predict(x_test)
        y_train_pred = clf.predict(x_train)

        SVM3AccTrain.append(accuracy_score(y_train, y_train_pred))
        SVM3AccTest.append(accuracy_score(y_test, y_test_pred))
        SVM3Precision.append(precision_score(y_test, y_test_pred))
        SVM3Recall.append(recall_score(y_test, y_test_pred))
        
    SVM3MeanAccTrainList.append(np.mean(SVM3AccTrain))
    SVM3MeanAccTestList.append(np.mean(SVM3AccTest))
    SVM3MeanPrecisionList.append(np.mean(SVM3Precision))
    SVM3MeanRecallList.append(np.mean(SVM3Recall))
bestAccIndex = SVM3MeanAccTestList.index(max(SVM3MeanAccTestList))
print ('''
             Statistics for SVM3
        \n bestSVM3AccTrainAvg = {}
        \n bestSVM3AccTestAvg = {}
        \n bestSVM3PrecisionAvg = {}
        \n bestSVM3RecallAvg = {}
        \n
        \n bestCParameter = {}
        '''.format(
        SVM3MeanAccTrainList[bestAccIndex],
        SVM3MeanAccTestList[bestAccIndex],
        SVM3MeanPrecisionList[bestAccIndex],
        SVM3MeanRecallList[bestAccIndex],
        CList[bestAccIndex] #index of CList and SVM3MeanAccTestList corresponds
        ))
########## Plot grapth of average test accuracy over tree depth 
fig = plt.figure()
plt.title("Average test accuracy with different regularization C")
plt.xlabel("value of regularization C")
plt.ylabel("average test accuracy")
plt.xscale('log')
plt.plot(CList, SVM3MeanAccTestList)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
plt.savefig('avgTestAccSVM3Norm_{}'.format(dt_string), format='png')
plt.show()