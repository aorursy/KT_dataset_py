import math
import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv("../input/train.csv")
X_train = train_data.iloc[:,1:]
Y_train = train_data.Activity
print('读取训练数据完毕\n...\n')
test_data = pd.read_csv("../input/test.csv")
X_test = test_data.iloc[:,:]
print('读取待预测数据完毕\n...\n')
# code for logistic regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
print('Logistic Regression训练完毕!\n...\n')
predicted_probs = lr.predict_proba(X_test)
PredictedProbability = predicted_probs[:,1]
MoleculeId = np.array(range(1,len(PredictedProbability)+1))
result=pd.DataFrame()
result['MoleculeId'] = MoleculeId
result['PredictedProbability'] = PredictedProbability
print('Logistic Regression预测完毕!\n文件保存...\n')
result.to_csv('lr_solution.csv',index=None)



