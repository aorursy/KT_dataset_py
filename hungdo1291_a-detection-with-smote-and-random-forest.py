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
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
# load dataset

train = pd.read_csv("../input/sample.csv",header=None)



#drop outliner

train.drop(labels=[29115,52648],axis=0,inplace=True)



#feature engineers

#there are three int cols that are not binary, 

# I feature engineer them to catergorical cloumns

train['col4a'] = train[4].map(lambda s: 1 if s == 0 else 0)

train['col4b'] = train[4].map(lambda s: 1 if 1<=s<=2 else 0)

train['col4c'] = train[4].map(lambda s: 1 if 3<=s<=4 else 0)

train['col4d'] = train[4].map(lambda s: 1 if 5<=s else 0)



train['col23a'] = train[23].map(lambda s: 1 if s == -1 else 0)

train['col23b'] = train[23].map(lambda s: 1 if 1<=s<=4 else 0)

train['col23c'] = train[23].map(lambda s: 1 if 5<=s else 0)



train['col36a'] = train[36].map(lambda s: 1 if s == 1 else 0)

train['col36b'] = train[36].map(lambda s: 1 if (s == 0 or s==3) else 0)

train['col36c'] = train[36].map(lambda s: 1 if s==2 else 0)

train['col36d'] = train[36].map(lambda s: 1 if (4<=s<=7) else 0)

train['col36e'] = train[36].map(lambda s: 1 if (8<=s or s<0) else 0)



train.drop(labels=[4,23,36], axis =1, inplace=True)



#there are 4 float64 columns, all are possitive

float_cols = [col for col in train.columns

              if(train[col].dtype == np.float64)]

train.loc[:,float_cols]=train.loc[:,float_cols]/train[float_cols].max()



#split the label from the data

Y = train[295]



# Drop 'label' column

features = train.drop(labels = [295],axis = 1)



labels=Y.map({'A':1,'B':0,'C':0,'D':0,'E':0})
features_train, features_test, labels_train, labels_test = train_test_split(features, 

                                                                            labels, 

                                                                            test_size=0.2, 

                                                                            random_state=0)
features_train.shape
#oversampling with SMOTE

oversampler=SMOTE(random_state=0)

os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
# verify new data set is balanced

len(os_labels[os_labels==1])
len(os_labels[os_labels==0])
clf=RandomForestClassifier(random_state=0)

clf.fit(os_features,os_labels)
# perform predictions on test set

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