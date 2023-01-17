# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as ml
import matplotlib.pyplot as plt
import seaborn as sns
ml.style.use('ggplot')

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.tail(20)
data.info()
data.describe()
data.columns
data['diagnosis'] = data['diagnosis'].replace('M',1)
data['diagnosis'] = data['diagnosis'].replace('B',0)
data.drop(columns=['id','Unnamed: 32'],inplace=True)
data.head()
data.columns
sns.pairplot(data.iloc[:,:21],'diagnosis')
plt.show()
X,Y = np.array(data.iloc[:,1:].values),np.array(data.iloc[:,0].values)
trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.2,random_state=0)
ss = StandardScaler()
trainx2,testx2 = ss.fit_transform(trainx),ss.fit_transform(testx)
param_grid = [
    {'C':[0.1,1,10,100,1000],'kernel':['rbf','sigmoid'],'gamma':['scale','auto']},
    {'C':[0.1,1,10,100,1000],'kernel':['poly'],'degree':[2,3]}
]
# Initial run(without tuning)
svm1,svm2 = SVC(),SVC()
svm1.fit(trainx,trainy)
print("BEFORE SCALING THE FEATURES -->\n")
print("Initial accuracy score = {:.3f}".format(accuracy_score(testy,svm1.predict(testx))))
print("\nInitial recall score = {:.3f}".format(recall_score(testy,svm1.predict(testx))))
print("Initial confusion matrix :\n",confusion_matrix(testy,svm1.predict(testx)))
svm2.fit(trainx2,trainy)
print("\n\nAFTER SCALING THE FEATURES -->\n")
print("Initial accuracy score = {:.3f}".format(accuracy_score(testy,svm2.predict(testx2))))
print("\nInitial recall score = {:.3f}".format(recall_score(testy,svm2.predict(testx2))))
print("Initial confusion matrix :\n",confusion_matrix(testy,svm2.predict(testx2)))
# Tuning the hyperparameters
svm_cv1,svm_cv2 = GridSearchCV(svm1,param_grid=param_grid,cv=10,n_jobs=-1),GridSearchCV(svm2,param_grid=param_grid,cv=10,n_jobs=-1)
svm_cv1.fit(trainx,trainy)
print("AFTER TUNING THE HYPERPARAMETERS : \n")
print("-"*100)
print("BEFORE SCALING THE FEATURES -->\n")
print("Accuracy score = {:.3f}".format(accuracy_score(testy,svm_cv1.predict(testx))))
print("\nRecall score = {:.3f}".format(recall_score(testy,svm_cv1.predict(testx))))
print("Confusion matrix :\n",confusion_matrix(testy,svm_cv1.predict(testx)))
print("\nThe best hyperparameters and accuracy score are : {} and {:.3f} respectively".format(svm_cv1.best_params_,svm_cv1.best_score_))
svm_cv2.fit(trainx2,trainy)
print("AFTER SCALING THE FEATURES -->\n")
print("Accuracy score = {:.3f}".format(accuracy_score(testy,svm_cv2.predict(testx2))))
print("\nRecall score = {:.3f}".format(recall_score(testy,svm_cv2.predict(testx2))))
print("Confusion matrix :\n",confusion_matrix(testy,svm_cv2.predict(testx2)))
print("\nThe best hyperparameters and accuracy score are : {} and {:.3f} respectively".format(svm_cv2.best_params_,svm_cv2.best_score_))
