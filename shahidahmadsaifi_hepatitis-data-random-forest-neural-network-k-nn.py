# Import basic libraries 

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt        

%matplotlib inline



import sklearn.preprocessing as skp

import sklearn.model_selection as skm

import os

#import classification modules

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

# first neural network with keras tutorial

from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_auc_score,roc_curve, auc, f1_score 
df = pd.read_csv("../input/hepatitis/hepatitis.csv")
print("Shape is :", df.shape)

display(df.dtypes)
df.isnull().sum()
df.head(5)
df.columns
df.describe()
df['age']  = np.where((df['age'] <18) ,'Teenager',

                               np.where((df['age'] >=18) & (df['age'] <=25),'Young',

                                np.where((df['age'] >=25) & (df['age'] <=40),'Adult',

                               'Old')))
df['age'].value_counts()
df['sex'].value_counts()
df.head(5)
import seaborn as sns

sns.boxplot(x ='alk_phosphate', data=df)
from scipy import stats 

df['alk_phosphate']  = (df.alk_phosphate - df.alk_phosphate.min()) / (df.alk_phosphate.max() - df.alk_phosphate.min())
df.alk_phosphate=df.alk_phosphate.round(2)
df['age'] = pd.Categorical(df['age'])
dfDummies = pd.get_dummies(df['age'], prefix = 'age')
df = pd.concat([df, dfDummies], axis=1)
df.head(5)
del df['age']
df['sgot']  = (df.sgot - df.sgot.min()) / (df.sgot.max() - df.sgot.min())
df.sgot=df.sgot.round(2)
df['bilirubin']  = (df.bilirubin - df.bilirubin.min()) / (df.bilirubin.max() - df.bilirubin.min())
df.bilirubin=df.bilirubin.round(2)
df['protime']  = (df.protime - df.protime.min()) / (df.protime.max() - df.protime.min())

df.protime=df.protime.round(2)
df['albumin']  = (df.albumin - df.albumin.min()) / (df.albumin.max() - df.albumin.min())

df.albumin=df.albumin.round(2)
dfDummies = pd.get_dummies(df['sex'], prefix = 'sex')
df = pd.concat([df, dfDummies], axis=1)
del df['sex'];
dfDummies = pd.get_dummies(df['steroid'], prefix = 'steroid')

df = pd.concat([df, dfDummies], axis=1)

del df['steroid'];
dfDummies = pd.get_dummies(df['antivirals'], prefix = 'antivirals')

df = pd.concat([df, dfDummies], axis=1)

del df['antivirals'];
dfDummies = pd.get_dummies(df['fatigue'], prefix = 'fatigue')

df = pd.concat([df, dfDummies], axis=1)

del df['fatigue'];
dfDummies = pd.get_dummies(df['malaise'], prefix = 'malaise')

df = pd.concat([df, dfDummies], axis=1)

del df['malaise'];
dfDummies = pd.get_dummies(df['anorexia'], prefix = 'anorexia')

df = pd.concat([df, dfDummies], axis=1)

del df['anorexia'];
dfDummies = pd.get_dummies(df['spleen_palable'], prefix = 'spleen_palable')

df = pd.concat([df, dfDummies], axis=1)

del df['spleen_palable'];
dfDummies = pd.get_dummies(df['liver_big'], prefix = 'liver_big')

df = pd.concat([df, dfDummies], axis=1)

del df['liver_big'];
dfDummies = pd.get_dummies(df['liver_firm'], prefix = 'liver_firm')

df = pd.concat([df, dfDummies], axis=1)

del df['liver_firm'];
dfDummies = pd.get_dummies(df['spiders'], prefix = 'spiders')

df = pd.concat([df, dfDummies], axis=1)

del df['spiders'];
dfDummies = pd.get_dummies(df['ascites'], prefix = 'ascites')

df = pd.concat([df, dfDummies], axis=1)

del df['ascites'];
dfDummies = pd.get_dummies(df['varices'], prefix = 'varices')

df = pd.concat([df, dfDummies], axis=1)

del df['varices'];
df["class"].replace((1,2),(0,1),inplace=True)
df["class"]=df["class"].astype("bool")
df.describe()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV



# Load classification dataset

y = df['class'].copy()

X = df.drop('class',axis=1)

estimator = RandomForestClassifier()

cv = StratifiedKFold(5)

visualizer = RFECV(estimator, cv=cv, scoring='f1_weighted')

visualizer.fit(X, y)        # Fit the data to the visualizer

# visualizer.show()           # Finalize and render the figure
# print summaries for the selection of attributes

print(visualizer.support_)

print(visualizer.ranking_)

#storing index of the false columns

cols=[index for index,value in enumerate(visualizer.support_) if value == False] 

print(cols)

features= X.columns

print(features)
sorted_features=pd.DataFrame(list(zip(features,visualizer.support_))).sort_values(by=1,ascending=False)

print(sorted_features)
#dropping faetures that are not that informative column index starts from zero index

X_after_dropping = X.drop(X.columns[cols],axis=1)

print("Shape of X: ", X.shape)

print("Shape of y: ", y.shape)

features_orig= X.columns

features_select=X_after_dropping.columns

print(features_select)
#create train-test split parts for manual split

trainX, testX, trainy, testy= skm.train_test_split(X_after_dropping, y, test_size=0.25, random_state=99)

print("\n Shape of train split: ")

print(trainX.shape, trainy.shape)

print("\n Shape of test split: ")

print(testX.shape, testy.shape)
# Random Forest

clf = RandomForestClassifier(n_estimators=50)

clf.fit(trainX,trainy)



predictions = clf.predict(testX)



acc_RF = accuracy_score(testy, predictions)*100

print('Accuracy of Random Forest (%): \n',acc_RF)

      



fpr1 , tpr1, _ = roc_curve(testy, predictions)

auc_RF = auc(fpr1, tpr1)*100

print("AUC of Random Forest (%): \n", auc_RF)



pre_RF = precision_score(testy,predictions)

print('Precision Score (%): \n',pre_RF )

rec_RF = recall_score(testy,predictions)

print('Recall Score (%): \n', rec_RF)
#MLP/Neural Network on Manual Split

#Default hyperparametres activation function is relu, optimiser is adams, default batch size=200, default max_iter/epochs=200

#clf = MLPClassifier()

#Try changing the number of layers and other parametres in the neural network and observe the effect on accuracy

clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=10,batch_size=200,max_iter=200, random_state=1)

clf.fit(trainX,trainy)

predictions = clf.predict(testX)



acc_NN = accuracy_score(testy, predictions)*100

print('Accuracy of MLP/Neural Network (%): \n',acc_NN)

      



fpr2 , tpr2, _ = roc_curve(testy, predictions)

auc_NN = auc(fpr2, tpr2)*100

print("AUC of MLP/Neural Network (%): \n", auc_NN)



pre_NN = precision_score(testy,predictions)

print('Precision Score (%): \n',pre_NN )

rec_NN = recall_score(testy,predictions)

print('Recall Score (%): \n', rec_NN)
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib

import seaborn as sns

matplotlib.rcParams['savefig.dpi'] = 144

import sklearn.preprocessing as skp

import sklearn.model_selection as skm



# import classificaiton modules



from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_auc_score, roc_curve, auc, f1_score
# K- NN on nanual Split



clf = KNeighborsClassifier(n_neighbors=20)

clf.fit(trainX,trainy)

predictions = clf.predict(testX)



acc_K_NN = accuracy_score(testy, predictions)*100

print('Accuracy of KNN (%): \n',acc_K_NN)

      



fpr3 , tpr3, _ = roc_curve(testy, predictions)

auc_KNN = auc(fpr3, tpr3)*100

print("AUC of KNN (%): \n", auc_KNN)



pre_KNN = precision_score(testy,predictions)

print('Precision Score (%): \n',pre_KNN )

rec_KNN = recall_score(testy,predictions)

print('Recall Score (%): \n', rec_KNN)

algos=["Random Forest","MLP/Neural Network","K Nearest Neighbor"]

acc=[acc_RF,acc_NN,acc_K_NN]

auc=[auc_RF,auc_NN,auc_KNN]

recall=[pre_RF,pre_NN,pre_KNN]

prec=[rec_RF,rec_NN,rec_KNN]

comp={"Algorithms":algos,"Accuracies":acc,"Area Under the Curve":auc,"Recall":recall,"Precision":prec}

compdf=pd.DataFrame(comp)

display(compdf.sort_values(by=["Accuracies","Area Under the Curve","Recall","Precision"], ascending=False))
import sklearn.metrics as metrics

roc_auc1 = metrics.auc(fpr1, tpr1)

roc_auc2 = metrics.auc(fpr2, tpr2)

roc_auc3 = metrics.auc(fpr3, tpr3)

# method I: plt

import matplotlib.pyplot as plt

plt.figure()

plt.title('Receiver Operating Characteristic')

plt.plot(fpr1, tpr1, 'b', label = 'ROC DecisionTree AUC = %0.2f' % roc_auc1)

plt.plot(fpr2, tpr2, 'r', label = 'ROC of MLP AUC = %0.2f' % roc_auc2)

plt.plot(fpr3, tpr3, 'g', label = 'ROC of KNN AUC = %0.2f' % roc_auc3)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()