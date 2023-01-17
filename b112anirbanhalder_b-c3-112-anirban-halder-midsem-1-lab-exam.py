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
from matplotlib import pyplot as plt
SUBJECTS=["English","Maths","Science","History","Geography"]
MARKS=[86,83,86,90,88] 
tick_label=["English","Maths","Science","History","Geography"]
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.8,color=['blue','yellow','blue','blue','blue'])
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')
plt.title("STUDENT's MARKS DATASET")
plt.show()
import numpy as np
a=np.array([9,5,4,1,8,12])
b=np.array([2,8,6,16,2,1])
print("CHECK IF B HAS SAME VIEWS TO MEMORY IN A")
print(b.base is a)
print("CHECK IF A HAS SAME VIEWS TO MEMORY IN B")
print(a.base is b)
div_by_3=a%3==0
div1_by_3=b%3==0
print("Divisible By 3")
print(a[div_by_3])
print(b[div1_by_3])
b[::-1].sort()
print("SECOND ARRAY SORTED")
print(b)
print("SUM OF ELEMENTS OF FIRST ARRAY")
print(np.sum(a))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)
df = df.drop(['Id'], axis=1)
df['Species'] = pd.factorize(df["Species"])[0] 
Target = 'Species'
Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

model = LogisticRegression(solver='lbfgs', multi_class='auto')
Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]
x2 = x[Target]
y1 = y[Features]
y2 = y[Target]

nb_model = GaussianNB() 
nb_model.fit(X=x1, y=x2)
result= nb_model.predict(y[Features]) 

f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)
import pandas as pd
titanic_data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("TITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

titanicdata.dropna(axis=1, how='all')
print("__\nTITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

print("__\nMean value of first 50 samples: \n",titanicdata[:50].mean())

print("__\nMean of the number of male passengers( Sex=1) on the ship :\n",titanic_data[titanicdata['Sex']==1].mean())

print("__\nHighest fare paid by any passenger: ",titanic_data['Fare'].max())