import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
print(os.listdir("../input"))
class class_def:    
    def __init__(self,class_def,number): # constructor
        self.class_def = class_def
        self.mean = class_def.mean(axis = 0)
        self.N = class_def.shape[0]
        self.total_digitnumber = number

    def std(self):
        return self.class_def.std(axis = 0)

    def scatter(self) :
        return np.cov(self.class_def.T)
    
    def cov(self) :
        return np.cov(self.class_def.T)
    
    def scatter_b(self,mean_vektör):
        return self.N*(((self.mean-mean_vektör).values.reshape(62,1)).dot((self.mean-mean_vektör).values.reshape(1,62)))
              
train_all = pd.read_csv("../input/opt_digits_train.csv")
train_all = train_all.reset_index(drop=True)
labels = train_all.iloc[:,64]
train_all = train_all.drop(labels = ["64"],axis = 1) 
g = sns.countplot(labels)
labels.value_counts()
train_all.columns = range(train_all.shape[1])
var = train_all.std(axis = 0 )
var_zero_index =[] 
m = 0
#variance zero is deleted
for i in range(var.shape[0]):
    if var[i] == 0:   
        train_all = train_all.drop(i,axis = 1) 
train_all.shape
var = train_all.std(axis = 0 )
train_all.shape
digit0 = class_def(train_all[labels==0],train_all.shape[0])
digit1 = class_def(train_all[labels==1],train_all.shape[0])
digit2 = class_def(train_all[labels==2],train_all.shape[0])
digit3 = class_def(train_all[labels==3],train_all.shape[0])
digit4 = class_def(train_all[labels==4],train_all.shape[0])
digit5 = class_def(train_all[labels==5],train_all.shape[0])
digit6 = class_def(train_all[labels==6],train_all.shape[0])
digit7 = class_def(train_all[labels==7],train_all.shape[0])
digit8 = class_def(train_all[labels==8],train_all.shape[0])
digit9 = class_def(train_all[labels==9],train_all.shape[0])
train_mean = train_all.mean(axis = 0)

train_all, test_all, labels, test_labels = train_test_split(train_all, labels, test_size = 0.1,random_state=0)

train_all = train_all.reset_index(drop=True)
labels = labels.reset_index(drop=True)
test_all = test_all.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)
mean_vectör = train_all.mean(axis=0)
sw = digit0.cov()+digit1.cov()+digit2.cov()+digit3.cov()+digit4.cov()+digit5.cov()+digit6.cov()+digit7.cov()+digit8.cov()+digit9.cov()
sb = digit0.scatter_b(mean_vectör)+digit1.scatter_b(mean_vectör)+digit2.scatter_b(mean_vectör)+digit3.scatter_b(mean_vectör)+digit4.scatter_b(mean_vectör)+digit5.scatter_b(mean_vectör)+digit6.scatter_b(mean_vectör)+digit7.scatter_b(mean_vectör)+digit8.scatter_b(mean_vectör)+digit9.scatter_b(mean_vectör)

ssb = np.linalg.inv(sw).dot(sb) 
eigval, vectors = np.linalg.eig(ssb)

z1 = np.dot(train_all, vectors[:,0])
z2 = np.dot(train_all, vectors[:,1])

for i in range(10):
    z1_digits = z1[labels == i]
    z2_digits = z2[labels == i]
    plt.scatter(z1_digits,z2_digits,label =""+str(i))

plt.title('Traning Data after LDA')
plt.legend()
plt.show()

z1_test = np.dot(test_all, vectors[:,0])
z2_test = np.dot(test_all, vectors[:,1])

plt.figure()
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for i in range(10):
        z1_digits_test = z1_test[test_labels==i]
        z2_digits_test = z2_test[test_labels==i]
        plt.scatter(z1_digits_test,z2_digits_test,label =""+str(i))
    plt.title('Test Data after LDA')
    plt.legend()

success_cnt = 0
for k in range(test_labels.shape[0]):
    distance =[]
    distance_1 = z1_test[k]-z1
    distance_2 = z2_test[k]-z2
    distance = distance_1*distance_1+distance_2*distance_2
    if(labels[np.argmin(distance)] == test_labels[k]):
        success_cnt = success_cnt+1 
        
test_error_LDA = 1-(success_cnt/test_all.shape[0])

print('Q2-LDA Test Error')
print(test_error_LDA)