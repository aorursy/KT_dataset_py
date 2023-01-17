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
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
data.head()
data.describe()
import seaborn as sns
#data.corr()
sns.heatmap(data.corr())
data.isnull().sum()
import matplotlib.pyplot as plt

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.diagnosis.value_counts()
sns.countplot(x = 'diagnosis', data = data, palette = 'hls')
plt.show()
y = data.diagnosis.values
x = data.drop(['diagnosis'], axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
confusion
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = 2*(precision*recall)/(precision+recall)

print("F1 score:",f1_score)
y_pred = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b

def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs


def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred
n_features = x_train.shape[1]
print('Number of Features', n_features)
w, b = weightInitialization(n_features)

coeff, gradient, costs = model_predict(w, b, x_train, y_train, learning_rate=0.0001,no_iterations=4500)

w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept',b)

final_train_pred = sigmoid_activation(np.dot(w,x_train.T)+b)
final_test_pred = sigmoid_activation(np.dot(w,x_test.T)+b)

m_tr =  x_train.shape[0]
m_ts =  x_test.shape[0]

y_tr_pred = predict(final_train_pred, m_tr)
print('Training Accuracy',metrics.accuracy_score(y_tr_pred.T, y_train))

y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy',metrics.accuracy_score(y_ts_pred.T, y_test))
print("Accuracy:",metrics.accuracy_score(y_ts_pred.T, y_test))
print("Precision:",metrics.precision_score(y_ts_pred.T, y_test))
print("Recall:",metrics.recall_score(y_ts_pred.T, y_test))
precision = metrics.precision_score(y_ts_pred.T, y_test)
recall = metrics.recall_score(y_ts_pred.T, y_test)
f1_score = 2*(precision*recall)/(precision+recall)
print("F1 score:",f1_score)
y_pred = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_ts_pred.T, y_test)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
