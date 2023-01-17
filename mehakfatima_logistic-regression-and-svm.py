# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Social_Network_Ads.csv')
print(data.shape)
data.keys()
data = data.drop(['User ID'] , axis = 1)
print(data)
data= pd.get_dummies(data)
import matplotlib.pyplot as plt 
import seaborn as sns
corr = data.corr()
corr_mat=data.corr(method='pearson')
plt.figure(figsize=(5,5))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
from sklearn.model_selection import train_test_split
train , test = train_test_split(data , test_size = 0.2, random_state=0)
predictions = ['Age' , 'EstimatedSalary', 'Gender_Male' , 'Gender_Female']
x_train = train[predictions]
y_train = train['Purchased']
x_test = test[predictions]
y_test = test['Purchased']
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
pred=logisticRegr.predict(x_test)
score1 = logisticRegr.score(x_train, y_train)
print (score1)
score = logisticRegr.score(x_test, y_test)
print (score)

from sklearn.linear_model import Ridge
ridge = Ridge().fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge.coef_ != 0)))
ridge10 = Ridge(alpha=10).fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge.coef_ != 0)))
ridge01 = Ridge(alpha=0.1).fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge.coef_ != 0)))
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(logisticRegr.coef_, 'o', label="LogisticRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(logisticRegr.coef_))
plt.ylim(-25, 25)
plt.legend()
from sklearn.linear_model import Lasso
lasso = Lasso().fit(x_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(x_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(x_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(x_train,y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ !=0)))
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,pred)
print (cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True,cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xx_train = sc.fit_transform(x_train)
xx_test = sc.transform(x_test)
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(xx_train, y_train)
classifier.score(xx_train,y_train)
y_pred = classifier.predict(xx_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
print (cm1)
