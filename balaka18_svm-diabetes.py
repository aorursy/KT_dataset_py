# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as ml
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ml.style.use('fivethirtyeight')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.info()
data.describe()
X = np.array(data.iloc[:,:-1].values)
Y = np.array(data.iloc[:,-1].values)
trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.2,random_state=0)
# Scaling
sc = StandardScaler()
trainx,testx = sc.fit_transform(trainx),sc.fit_transform(testx)

print("For X : Training : {} ; Testing : {}".format(trainx.shape,testx.shape))
print("\nFor Y : Training : {} ; Testing : {}".format(trainy.shape,testy.shape))

sns.pairplot(data,hue='Outcome')
plt.show()
# RBF Kernel
m=50
accuracy,recall,f1 = [],[],[]
x_axis = [i for i in range(1,m)]
for i in range(1,m):
    svm_c = SVC(kernel='rbf',C=i,gamma='auto')
    svm_c.fit(trainx,trainy)
    y_pred_i = svm_c.predict(testx)
    accuracy.append(accuracy_score(testy,y_pred_i))
    recall.append(recall_score(testy,y_pred_i))
    f1.append(f1_score(testy,y_pred_i))
    print("For C = {},".format(i),"confusion matrix : \n",confusion_matrix(testy,y_pred_i),"\n")
plt.figure(figsize=(20,10))
plt.plot(x_axis,accuracy,color='red',marker='o',label='Accuracy')
plt.plot(x_axis,recall,color='blue',marker='x',label='Recall score')
plt.plot(x_axis,f1,color='green',marker='x',label='F1 score')
plt.legend()
plt.xlabel("Value of C")
plt.ylabel("Metric score")
plt.title("Comparison plot")
plt.show()
print("METRIC : ACCURACY\nHighest accuracy is obtained for C = {}, and the accuracy is = {}".format(np.argmax(accuracy)+1,accuracy[np.argmax(accuracy)]))
print("\nMETRIC : RECALL SCORE\nHighest accuracy is obtained for C = {}, and the accuracy is = {}".format(np.argmax(recall)+1,accuracy[np.argmax(recall)]))
print("\nMETRIC : F1 SCORE\nHighest accuracy is obtained for C = {}, and the accuracy is = {}".format(np.argmax(f1)+1,accuracy[np.argmax(f1)]))
