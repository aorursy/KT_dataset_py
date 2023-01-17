import pandas as pd

import numpy as np

from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor

import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score,f1_score,classification_report

import seaborn as sns

from imblearn.over_sampling import SMOTE
dataset=pd.read_csv('../input/creditcardfraud/creditcard.csv')   
dataset.head()
dataset['Class'].value_counts().plot(kind='bar',figsize=[10,5])

dataset['Class'].value_counts()

dataset = dataset.drop(['Time', 'Amount'], axis=1)

dataset.head()
label=dataset['Class']

Data=dataset.drop(["Class"],axis=1)
print("Data Types\n{}".format(Data.dtypes))
null_counts = Data.isnull().sum()

print("Number of null values in each feature:\n{}".format(null_counts))
x_train,x_test,y_train,y_test = train_test_split(Data,label,test_size=0.33,random_state=1236)

print("Label 1, Before using SMOTE: {} ".format(sum(y_train==1)))

print("Label 0, Before using SMOTE: {} ".format(sum(y_train==0)))
OS = SMOTE(random_state=12)

x_train_OS, y_train_OS = OS.fit_sample(x_train, y_train)
print("Label 1, After using SMOTE: {}".format(sum(y_train_OS==1)))

print("Label 0, After using SMOTE: {}".format(sum(y_train_OS==0)))
model = CatBoostClassifier(iterations=100,

                             depth=12,

                             eval_metric='AUC',

                             random_seed = 2018,

                             od_type='Iter',

                             metric_period = 1,

                             od_wait=100)
model.fit(x_train_OS,y_train_OS)
predict = model.predict(x_test)
cm = pd.crosstab(y_test, predict, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))



sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues" ,fmt='g',

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],)

ax1.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1,fontsize=14)

plt.show()
acc=accuracy_score(y_test,predict)

print('Accuracy =' ,acc)
precision = precision_score(y_test, predict)

print('Precision =' ,precision)
auc_score=roc_auc_score(y_test, predict)

print('AUC =' ,auc_score)
fpr, tpr, threshold = metrics.roc_curve(y_test, predict)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

recall = recall_score(y_test, predict)

print("Recall : ",recall )
f1score = f1_score(y_test,predict, average='macro')

print("F1 Score : ",f1score )