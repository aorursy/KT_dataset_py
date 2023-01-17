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
import pandas as pd 

import numpy as np

import  matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

%matplotlib inline
data = pd.read_csv('../input/voicegender/voice.csv')
data.shape
data.info()
data.describe()
data.head()
sns.countplot(x='label',data=data)

plt.show()

print(data['label'].value_counts(normalize=True)*100)
sns.distplot(data['meanfreq'])
sns.distplot(data['sd'])
sns.distplot(data['median'])
X = data.drop(['label'],axis=1)

Y = data['label']
from scipy.stats import shapiro
stat, p = shapiro(X)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# the data is normally distributed
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

Y = labelencoder_y.fit_transform(Y)
Y
from sklearn.preprocessing import StandardScaler

stnd_scaler = StandardScaler()
stnd_scaler.fit(X)

X = stnd_scaler.transform(X)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

print(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

conf_matrix = pd.DataFrame(data=cm,columns=['Predicate:Male','Predicate:female'],index=['Actual:Male','Actual:female'])

sns.heatmap(data=conf_matrix,fmt='d',annot=True)