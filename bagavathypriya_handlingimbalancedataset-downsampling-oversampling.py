# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
df=pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
x=df.drop('Class',axis=1)
y=df['Class']
state=np.random.RandomState(42)
xoutlier=state.uniform(low=0,high=1,size=(x.shape[0],x.shape[1]))
sns.countplot(y)
normal=df[df['Class']==0]
fraud=df[df['Class']==1]
print('No of fraud records of data: {}'.format(fraud.shape[0]))
print('No of normal records of data: {}'.format(normal.shape[0]))
from imblearn.under_sampling import NearMiss
nm=NearMiss()
xsam,ysam=nm.fit_sample(x,y)
fraud=ysam[ysam==1]
normal=ysam[ysam==0]
print('No of fraud records of data: {}'.format(fraud.shape[0]))
print('No of normal records of data: {}'.format(normal.shape[0]))
from collections import Counter
print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(ysam)))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()
xosam,yosam=ros.fit_sample(x,y)
print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(yosam)))
from imblearn.combine import SMOTETomek
com=SMOTETomek(random_state=42)
xcom,ycom=com.fit_sample(x,y)
print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(ycom)))
plt.figure(figsize=(7,7))
plt.tight_layout()
plt.subplot(2,2,1)
plt.title('Original Data')
sns.countplot(y)


plt.tight_layout()
plt.subplot(2,2,2)
plt.title('Downsampled data')
sns.countplot(ysam)

plt.tight_layout()
plt.subplot(2,2,3)
plt.title('Oversampled data')
sns.countplot(yosam)

plt.tight_layout()
plt.subplot(2,2,4)
plt.title('Smotet data')
sns.countplot(ycom)

plt.show()
