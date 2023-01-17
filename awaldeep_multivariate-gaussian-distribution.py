import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Load data as dataframe
data=pd.read_csv("../input/creditcard.csv")
data.head()
# min max scaling except on time and class columns
X=data.drop(["Time","Class"],axis=1)
X= (X- X.min()) / (X.max() - X.min())
X.head()
for j in list(data):
    for i in range(2):
        sns.kdeplot(data[data.Class==i][j])
    plt.show()
# using only some of the features that look distinct for anomaly and normal class. 
keep=['V1','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19']
X=X[keep]
X['Class']=data['Class']
anomalies=X[X['Class']==1]
normal=X[X['Class']==0]
# data_train, data_cv, data_test

data_train=normal.iloc[:int(len(normal)*0.6)]
data_cv=normal.iloc[int(len(normal)*0.6):int(len(normal)*0.8)]
data_test=normal.iloc[int(len(normal)*0.8):]

data_cv=data_cv.append(anomalies.iloc[:int(len(anomalies)*0.5)])
data_test=data_test.append(anomalies.iloc[int(len(anomalies)*0.5):])

x_train=data_train.drop(labels='Class',axis=1)
y_train=data_train['Class']
x_cv=data_cv.drop(labels='Class',axis=1)
y_cv=data_cv['Class']
x_test=data_test.drop(labels='Class',axis=1)
y_test=data_test['Class']
from scipy.stats import multivariate_normal
# function to calculate parameters, mu and sigma 
def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma

# function to calculate probability density
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
# mu,sigma of normal transactions to be used
mu,sigma=estimateGaussian(x_train)

p_train=multivariateGaussian(x_train,mu,sigma)
p_cv=multivariateGaussian(x_cv,mu,sigma)
p_test=multivariateGaussian(x_test,mu,sigma)
test_cases=np.sort(np.unique(p_train))[:1000]
#epsilon=1.0527717316e-70 
d={}
d2={}
for epsilon in test_cases:
    y_pred_cv=p_cv<epsilon
    d[epsilon]=f1_score(y_cv,y_pred_cv.astype(int))
epsilon=max(d.items(),key=lambda x: x[1])[0]

#refining epsilon value
for i in range(100):
    epsilon=epsilon*0.9
    y_pred_cv=p_cv<epsilon
    d2[epsilon]=f1_score(y_cv,y_pred_cv.astype(int))
epsilon=max(d2.items(),key=lambda x: x[1])[0]
print(f"Optimum epsilon value : {epsilon}")
print("f1 score,recall,precision on test set :")
f1_score(y_cv,y_pred_cv.astype(int)),recall_score(y_cv,y_pred_cv.astype(int)),precision_score(y_cv,y_pred_cv.astype(int))
confusion_matrix(y_cv,y_pred_cv.astype(int))
y_pred_test=p_test<epsilon
print("f1 score,recall,precision on test set :")
f1_score(y_test,y_pred_test.astype(int)),recall_score(y_test,y_pred_test.astype(int)),precision_score(y_test,y_pred_test.astype(int))
confusion_matrix(y_test,y_pred_test.astype(int))

