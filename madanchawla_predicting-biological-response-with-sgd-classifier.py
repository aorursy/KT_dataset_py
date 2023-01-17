%matplotlib inline
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
import pandas as pd
## Read in the data set.
train_data = pd.read_csv('../input/train.csv', delimiter=',')
test_data = pd.read_csv('../input/test.csv', delimiter=',')

trainx = train_data.iloc[:,1:]
trainy = train_data.iloc[:,0]
## Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
trainy = 2*trainy - 1

testx = test_data.iloc[:,:]
testy = test_data.iloc[:,0]
## Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
testy = 2*testy - 1



print()
print("Train data:")
print(trainx.shape)
print(trainy.shape)
print()
print("Test data:")
print(testx.shape)
print(testy.shape)
n1 = np.array(np.where(trainy==1)).shape[1]
n2 = np.array(np.where(trainy==-1)).shape[1]
n1, n2
from sklearn.linear_model import SGDClassifier

def Classifier(trainx, trainy, testx, c):
    ## Fit logistic classifier on training data
    clf = SGDClassifier(loss="log", penalty="none", alpha=c)
    clf.fit(trainx, trainy)

    ## Pull out the parameters (w,b) of the logistic regression model
    w = clf.coef_[0,:]
    b = clf.intercept_

    ## Get predictions on training and test data
    trainy_predict = clf.predict(trainx)
    testy_predict = clf.predict(testx)
    pred_prob = clf.predict_proba(testx)[:,1]
    
    return trainy_predict, testy_predict, pred_prob

def calculate_error(y, y_predict):
    ## Compute errors
    err = float(np.sum((y_predict > 0.0) != (y > 0.0))/len(trainx))
    return err

C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#s=array of errors correspnding to C
s = np.zeros(len(C))
N = len(trainx)
k=11   
I = np.random.permutation(N)
for j in range(len(C)):
    s[j] = 0
    for i in range(k):
        test_ind = I[int(i*(N/k)):int((i+1)*(N/k)-1)]
        train_ind = np.setdiff1d(I, test_ind)
        trainy_predict, testy_predict, pred_prob = Classifier(trainx.iloc[train_ind], trainy.iloc[train_ind], trainx.iloc[test_ind], C[j])
        #calculating error
        s[j] += calculate_error(trainy[test_ind], testy_predict)
    s[j] = s[j]/k
    
C_final = C[np.argmin(s)]
trainy_predict, testy_predict, pred_prob = Classifier(trainx, trainy, testx, C_final)

train_error = calculate_error(trainy, trainy_predict)
print("Training error = ", train_error)
print("Prediction: ", testy_predict)
plt.plot(C, s)
print(pred_prob[:25])
testy_predict[:25]
submission = pd.DataFrame({
    "MoleculeId": testx.index+1,
    "PredictedProbability": pred_prob
})
submission.to_csv("bioresponse_submission.csv", index=False)