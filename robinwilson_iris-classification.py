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
import matplotlib.pyplot as plt
import seaborn as sns
iris=pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
iris.head()
print(iris.info())
print(iris.shape)
iris.describe()
iris.corr()
sns.heatmap(iris.corr(),annot=True)
iris['species'].unique()
plt.figure(figsize = (6,6))
plot_shipmode = iris['species'].value_counts()
colors = ('Gold' ,'Cyan' , 'LightGreen')
label = iris['species'].unique()

plt.pie(plot_shipmode, 
       autopct = '%1.1f%%',
       explode = (0.08,0.08,0.08),
       shadow = True,
       colors = colors,
       labels = label);
iris.isnull().sum()
# handling categorical values
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
iris['species']=label.fit_transform(iris['species'])
iris['species'].head()
#splitting data
train=iris.drop(columns=['species'],axis=1)
test=iris['species']
print(train.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=30)
x_train.shape,y_train.shape
from sklearn.naive_bayes import GaussianNB
classi=GaussianNB()
classi.fit(x_train,y_train)
predict=classi.predict(x_test)
print(predict)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy is:",accuracy_score(y_test,predict))
print("F1 Score is:",classification_report(y_test,predict))
print("confusion matrix:",confusion_matrix(y_test,predict))