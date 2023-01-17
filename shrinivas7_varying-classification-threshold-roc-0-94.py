## Importing necessary libraries

import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import numpy as np



%matplotlib inline
df=pd.read_csv('../input/creditcard.csv')
df.shape
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

Class_split = df.groupby(['Class']).size()

print(Class_split)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 50



ax1.hist(df.Time[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Time[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Number of Transactions')

plt.show()
df.isnull().values.any() # We see that there are no missing values in the data set
columns=df.columns

# The labels are in the last column ('Class'). 

features_columns=columns.delete(len(columns)-1)



features=df[features_columns]

labels=df['Class']
features['Amount'] = (features['Amount'] - features['Amount'].min()) /  (features['Amount'].max() - features['Amount'].min())

features['Time'] = (features['Time'] - features['Time'].min()) /  (features['Time'].max() - features['Time'].min())
from sklearn import ensemble

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(1, oob_score = True, random_state =99)

model.fit(features,labels)
feature_importance = pd.Series(model.feature_importances_, index = features.columns)

feature_importance.plot( kind = 'barh', figsize = (7,6));
# Dropping the least important Features

df = df.drop(['V2','V8','V9','V5','V3','V23','V18','V6','V25','V24','V28'], axis =1)
features_train, features_test, labels_train, labels_test = train_test_split(features, 

                                                                            labels, 

                                                                            test_size=0.3, 

                                                                            random_state=1)
oversampler=SMOTE(random_state=1)

os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
from sklearn.cross_validation import KFold, cross_val_score

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
clf=RandomForestClassifier(n_estimators = 100, max_depth = 4 ,max_features = 'auto',random_state=99)

clf.fit(os_features,os_labels)
actual=labels_test

predictions=clf.predict(features_test)
confusion_matrix(actual,predictions)
print(classification_report(actual,predictions))
from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print ('AUC:', roc_auc)
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 10,  penalty = 'l1', random_state=99)
lr.fit(os_features,os_labels)
LR_predictions=lr.predict(features_test)
confusion_matrix(labels_test,LR_predictions)
print(classification_report(labels_test,LR_predictions))
from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(labels_test, LR_predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print ('AUC:',roc_auc)
predprob = lr.predict_proba(features_test) # Getting the probabilty of the classes
predprob
prob_dataframe = pd.DataFrame(predprob)
prob_dataframe['class'] = np.where(prob_dataframe[1] > .40, 1, 0)
prob_dataframe.head(10)
predicted40 = prob_dataframe['class']
confusion_matrix(labels_test,predicted40)
print(classification_report(labels_test,predicted40))
from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(labels_test, predicted40)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')