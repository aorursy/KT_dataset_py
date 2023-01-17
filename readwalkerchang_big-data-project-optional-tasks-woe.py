# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
WOE_IV = pd.read_csv('../input/WOE_IV.csv')
Coefficient = pd.read_csv('../input/Coefficient_modified.csv')
data2 = pd.read_csv('../input/data2.csv')
WOE_IV
Types = ['ServiceTypeB','ServiceTypeC','ServiceTypeE','ServiceTypeH','ServiceTypeI','ServiceTypeM','ServiceTypeN','ServiceTypeS']
ServiceType= pd.DataFrame(index=['0'], columns = [Types])

def logistic_regression(Types):
    i = 0
    for x in Types:
        b = Coefficient[x].sum()
        c =WOE_IV.WOE.iloc[[i]].sum()
        Sum = c*b
        regression = np.exp(Sum-2.44268)/(1+np.exp(Sum-2.44268))
        ServiceType[x].iloc[[0]] = regression
        #regression function
        i = i+1

logistic_regression(Types)

ServiceType
data2.ServiceType.replace('C',0.0944917,inplace=True,regex = True)
data2.ServiceType.replace('B',0.0799755,inplace=True,regex = True)
data2.ServiceType.replace('E',0.820374,inplace=True,regex = True)
data2.ServiceType.replace('H',0.156565,inplace=True,regex = True)
data2.ServiceType.replace('I',0.0932235,inplace=True,regex = True)
data2.ServiceType.replace('M',0.0959576,inplace=True,regex = True)
data2.ServiceType.replace('N',0.123247,inplace=True,regex = True)
data2.ServiceType.replace('S',0.0812917,inplace=True,regex = True)
data2
data2.ServiceType.describe() 
pre_pov = data2.ServiceType[(data2.ServiceType>0.1)]
pre_pov.describe() 
pre_neg = data2.ServiceType[(data2.ServiceType<0.1)].describe() 
pre_neg.describe() 
def calculate_sum(name):
    name = (ServiceStatus == name)
    return name.sum()

ServiceStatus = data2['ServiceStatus']
Complete_of_all = calculate_sum(name = 'Complete')
Quit_of_all = calculate_sum(name = 'Quit')
print("Acual Number of complete is "+str(Complete_of_all))
print("Acual Number of quiz is "+str(Quit_of_all))
pov = ((data2.ServiceType>0.1)).sum()
neg = ((data2.ServiceType<0.1)).sum()
print("Estimate Number of complete is "+str(pov))
print("Estimate Number of quit is "+str(neg))
TP = ( (data2.ServiceType  > 0.1)&(data2.ServiceStatus =='Complete')).sum()
FP = ( (data2.ServiceType  < 0.1)&(data2.ServiceStatus !='Quit')).sum()
TPR = TP/pov
FPR = FP/neg
print("TP is "+str(TP))
print("FP is "+str(FP))
print("TPR is "+str(TPR))
print("FPR is "+str(FPR))
def pdf(threshold, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((threshold-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist

threshold = np.linspace(0.07, 0.82, num=100)
good_pdf = pdf(threshold,pre_pov.std(),pre_pov.mean())
bad_pdf = pdf(threshold,pre_neg.std(),pre_neg.mean())
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
    for i in range(len(threshold)):
        #We are only interested in non-zero values of bad
        if bad_pdf[i]>0:
            cum_TP+=bad_pdf[len(threshold)-1-i]
            cum_FP+=good_pdf[len(threshold)-1-i]
            FPR=cum_FP/total_good
            TPR=cum_TP/total_bad
            TPR_list.append(TPR)
            FPR_list.append(FPR)
    
    #Calculating AUC, taking the 100 timesteps into account
    auc=np.sum(TPR_list)/100
    #Plotting final ROC curve
    ax.plot(FPR_list, TPR_list)
    ax.plot(threshold,threshold, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f"%auc])
fig, ax = plt.subplots(1,1, figsize=(5,5))
plot_roc(good_pdf, bad_pdf, ax)