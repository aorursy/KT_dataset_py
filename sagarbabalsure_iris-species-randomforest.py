# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

df.head()
df.describe()
df.info()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df['target'] = enc.fit_transform(df['species'])
df.drop(['species'],axis='columns')
import matplotlib.pyplot as plt

plt.legend('Flower species')

plt.xlabel('sepal length(cm)')

plt.ylabel('output')

plt.scatter(df['sepal_length'],df['target'])

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:4].values,df['target'],test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(x_train,y_train)
y_predict = RFC.predict(x_test)

y_predict
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,classification_report,precision_score

cm = confusion_matrix(y_test,y_predict)

cm
print(accuracy_score(y_test,y_predict))