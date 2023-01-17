## Importing necessary libraries

import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec

import numpy as np







%matplotlib inline



df=pd.read_csv('creditcard.csv')



columns=df.columns

# The labels are in the last column ('Class'). Simply remove it to obtain features columns

features_columns=columns.delete(len(columns)-1)



features=df[features_columns]

labels=df['Class']
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 50



ax1.hist(df.Time[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Time[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Number of Transactions')

plt.show()
_features = df.ix[:,1:29].columns

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show()
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df.head(5)
features_train, features_test, labels_train, labels_test = train_test_split(features, 

                                                                            labels, 

                                                                            test_size=0.1, 

                                                                            random_state=1)
oversampler=SMOTE(random_state=1)

os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
len(os_labels[os_labels==1])
clf=RandomForestClassifier(random_state=1)

clf.fit(os_features,os_labels)
actual=labels_test

predictions=clf.predict(features_test)
confusion_matrix(actual,predictions)

from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

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