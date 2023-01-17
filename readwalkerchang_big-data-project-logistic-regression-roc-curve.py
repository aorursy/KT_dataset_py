# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import math as math
from math import exp
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Sum = pd.read_csv('../input/Sum_of_coeffcients.csv')
data2 = pd.read_csv('../input/data2.csv')
Sum.head(5)
logistic_regression = pd.Series({}, name='logistic_regression')

cal = pd.DataFrame({'Sum_of_coeffcients':Sum['Sum'],'logistic_regression':logistic_regression,'ServiceStatus':data2['ServiceStatus']})
f = lambda x:float(np.exp(x-2.44268)/(1+np.exp(x-2.44268)))
cal.logistic_regression = cal.Sum_of_coeffcients.apply(f)

plt.figure(figsize=(5,5))
plt.scatter(x=cal.logistic_regression, y=cal.logistic_regression, alpha=0.5)
plt.show()
pov_pre = cal.logistic_regression[cal.ServiceStatus == 'Complete']
pov_pre.describe() 
neg_pre = cal.logistic_regression[cal.ServiceStatus == 'Quit']
neg_pre.describe() 
def calculate_sum(name):
    name = (ServiceStatus == name)
    return name.sum()

ServiceStatus = data2['ServiceStatus']
Complete_of_all = calculate_sum(name = 'Complete')
Quit_of_all = calculate_sum(name = 'Quit')
print("Acual Number of complete is "+str(Complete_of_all))
print("Acual Number of quiz is "+str(Quit_of_all))
pov =(cal.logistic_regression  > 0.18).sum()
neg =(cal.logistic_regression  < 0.18).sum()


print("Estimate Number of complete is "+str(pov))
print("Estimate Number of quiz is "+str(neg))

TP = ( (cal.logistic_regression  > 0.18)&(data2.ServiceStatus =='Complete')).sum()
FP = ( (cal.logistic_regression  < 0.18)&(data2.ServiceStatus !='Quit')).sum()
TPR = TP/pov
FPR = FP/neg

print("TP is "+str(TP))
print("FP is "+str(FP))
print("TPR is "+str(TPR))
print("FPR is "+str(FPR))
def pdf(x, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((x-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist

x = np.linspace(0, 10, num=100)
good_pdf = pdf(x,pov_pre.std(),pov_pre.mean())
bad_pdf = pdf(x,neg_pre.std(),neg_pre.mean())
TPR_list=[]
FPR_list=[]


for i in range(100):
    pov1 =(cal.logistic_regression > x[i]).sum()
    neg1 =(cal.logistic_regression  < x[i]).sum()
    TP = ( (cal.logistic_regression  > x[i])&(data2.ServiceStatus =='Complete')).sum()
    FP = ( (cal.logistic_regression  < x[i])&(data2.ServiceStatus !='Quit')).sum()
    TPR = TP/pov1
    FPR = FP/neg1
    TPR_list.append(TPR)
    FPR_list.append(FPR)


auc=np.sum(TPR_list)/100
auc
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(FPR_list, TPR_list)
ax.plot(x,x, "--")
ax.set_xlim([0,0.5])
ax.set_ylim([0,0.5])
ax.set_title("ROC Curve", fontsize=14)
ax.set_ylabel('TPR', fontsize=12)
ax.set_xlabel('FPR', fontsize=12)
ax.grid()
ax.legend(["AUC=%.3f"%auc])
def pdf(x, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((x-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist

x = np.linspace(0, 1, num=100)
good_pdf = pdf(x,pov_pre.std(),pov_pre.mean())
bad_pdf = pdf(x,neg_pre.std(),neg_pre.mean())
def plot_roc(good_pdf, bad_pdf, ax):
    #Total
    total_bad = np.sum(bad_pdf)
    total_good = np.sum(good_pdf)
    #Cumulative sum
    cum_TP = 0
    cum_FP = 0
    #TPR and FPR list initialization
    TPR_list=[]
    FPR_list=[]
    #Iteratre through all values of x
    for i in range(len(x)):
        #We are only interested in non-zero values of bad
        if bad_pdf[i]>0:
            cum_TP+=bad_pdf[len(x)-1-i]
            cum_FP+=good_pdf[len(x)-1-i]
            FPR=cum_FP/total_good
            TPR=cum_TP/total_bad
            TPR_list.append(TPR)
            FPR_list.append(FPR)
    
    #Calculating AUC, taking the 100 timesteps into account
    auc=np.sum(TPR_list)/100
    #Plotting final ROC curve
    ax.plot(FPR_list, TPR_list)
    ax.plot(x,x, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f"%auc])
fig, ax = plt.subplots(1,1, figsize=(5,5))
plot_roc(good_pdf, bad_pdf, ax)
def plot_pdf(good_pdf, bad_pdf, ax):
    ax.fill(x, good_pdf, "g", alpha=0.5)
    ax.fill(x, bad_pdf,"r", alpha=0.5)
    ax.set_xlim([0,1])
    ax.set_ylim([0,5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_xlabel('P(X="bad")', fontsize=12)
    ax.legend(["good","bad"])
fig, ax = plt.subplots(1,1, figsize=(5,5))
plot_pdf(good_pdf, bad_pdf, ax)