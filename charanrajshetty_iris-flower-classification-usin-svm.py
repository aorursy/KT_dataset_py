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
df_Iris = pd.read_csv('../input/iris-svm-dataset/Iris.csv')
df_Iris
Species = df_Iris.Species.unique()

Species =Species.tolist()
count = [0]*len(Species)
for i in Species:

    for j in df_Iris['Species']:

        if ( i == j ):

            count[Species.index(i)] += 1
count
print(Species)


from matplotlib import pyplot as plt

plt.figure(figsize=(20,20))

plt.title("Pie Chart of Species")

plt.pie(count,labels=Species,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.show()
import seaborn as sns

sns.pairplot(df_Iris,hue='Species')
plt.figure(figsize=(10,10))

sns.heatmap(df_Iris.iloc[:,1:5].corr(),annot=True,fmt='.0%')
from sklearn.model_selection import train_test_split

X = df_Iris.iloc[:,1:5]

Y = df_Iris.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.2,random_state=42)
from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)
preds = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))