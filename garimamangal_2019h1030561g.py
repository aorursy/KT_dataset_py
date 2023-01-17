# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline
df = pd.read_csv("../input/minor-project-2020/train.csv")

df.head()
from sklearn.model_selection import train_test_split

from sklearn.utils import resample





negative = df[df.target==0]

positive = df[df.target==1]



# downsample majority

neg_downsampled = resample(negative,

 replace=True, 

 n_samples=len(positive), 

 random_state=27) 



# combine minority and downsampled majority

downsampled = pd.concat([positive, neg_downsampled])



# check new class counts

downsampled.target.value_counts()


Y=downsampled[['target']]

downsampled.drop(['target'],axis=1,inplace=True)

X=downsampled
X_train, X_test, y_train, y_test = train_test_split( X,Y, test_size=0.3, random_state=40)
from sklearn.preprocessing import StandardScaler

X_train.info()

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)

rf.fit(scaled_X_train, y_train)

print(rf.score(scaled_X_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

plot_confusion_matrix(rf, scaled_X_test, y_test, cmap = plt.cm.Blues)
y_pred = rf.predict_proba(scaled_X_test)[:,1]

from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for binary classification', fontsize= 18)

plt.show()
test=pd.read_csv('../input/minor-project-2020/test.csv')

test_pred = rf.predict_proba(test)[:,1]

print(test_pred)
idd = test['id']

submission_garima = pd.DataFrame({

                  "id": idd, 

                  "target": test_pred})
submission_garima.to_csv('submission_garima_mangal.csv', index=False)