import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
diab_data = pd.read_csv("....\\PimaIndians.csv")
diab_data.info()

#diab_data.info()
#fig  = plt.figure()
#plt.plot(diab_data['age'], diab_data['test'])
#plt.show()
Y = diab_data['test']
Output  = Y.copy()
Output = Output.replace(['negatif','positif'], [0,1])
col_del = ['test']
diab_data_modified = diab_data.copy()
diab_data_modified = diab_data_modified.drop(col_del, 1)
Input2 = diab_data_modified.copy()   # size = m+1,1
Input1 = (Input2 - np.min(Input2))/(np.max(Input2) - np.min(Input2))
one_col = np.ones((diab_data_modified.shape[0],1), int)
Input1['const'] =  one_col

Input = Input1.copy()
Input

parameters = np.random.rand(Input.shape[1], )
n = Input.shape[0]
alpha = 0.1
#z = np.dot(Input, parameters)
#y_cap = 1/(1+ np.exp(-z))
#Y.shape
#c = y_cap - Output
#grad = (1/n)*(np.transpose(Input))*(y_cap - Output)
#grad.shape
for i in range(1000):
    
    z = np.dot(Input, parameters)
    y_cap = 1/(1+ np.exp(-z))
    
    for j in range(y_cap.shape[0]):
        if y_cap[j] >= 0.5:
            y_cap[j] = 1
        else:
            y_cap[j] = 0
                
    grad = (1/n)*np.dot((np.transpose(Input)),(y_cap - Output))
    
    parameters = parameters - alpha*grad

y_predict = np.dot(Input, parameters)
y_predict = 1/(1+ np.exp(-y_predict))

for k in range(y_predict.shape[0]):
    if y_predict[k] >= 0.5:
        y_predict[k] = 1
    else:
        y_predict[k] = 0
y_predict   
score = 0
for l in range(y_predict.shape[0]):
    if (y_predict[l] - Output[l]) == 0:
        score = score+1
    else:
        score = score
        
score = score/Output.shape[0]
score
