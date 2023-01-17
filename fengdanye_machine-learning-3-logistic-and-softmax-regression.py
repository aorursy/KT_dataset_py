import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import os
print(os.listdir("../input"))
# set pyplot parameters to make things pretty
plt.rc('axes', linewidth = 1.5, labelsize = 14)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 3, width = 1.5)
plt.rc('ytick.major', size = 3, width = 1.5)
wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()
wineData.quality.unique() # 3-8
plt.hist(wineData.quality.values,bins=[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],edgecolor = 'black')
plt.xlabel('wine quality', fontsize = 18)
plt.ylabel('# of occurrences', fontsize = 18)
plt.show()
wineData['category'] = wineData['quality'] >= 7
wineData.head()
from sklearn.linear_model import LogisticRegression
X = wineData[['fixed acidity','volatile acidity']].values
y = wineData['category'].values.astype(np.int)

scaler = StandardScaler()
Xstan = scaler.fit_transform(X)

# save the standardized data for plotting
dataStan=pd.DataFrame()
dataStan['fixed acidity(stan)']=Xstan[:,0]
dataStan['volatile acidity(stan)']=Xstan[:,1]
dataStan['category']=y
dataStan.head()
logReg = LogisticRegression()
logReg.fit(Xstan,y)
logReg.coef_, logReg.intercept_
logReg.predict_proba(Xstan)
yhat = logReg.predict(Xstan)
dataStan['predict']=yhat
dataStan.head()
# This is only availbale in seaborn 0.9.0, but Kaggle only has seaborn 0.8.1 :(
# sns.scatterplot('fixed acidity(stan)','volatile acidity(stan)',hue='category',data=dataStan) 
sns.lmplot(x='fixed acidity(stan)',y='volatile acidity(stan)',hue='category',data=dataStan, fit_reg=False, legend=False)
x1stan = np.linspace(-2,5,num=50)
theta1 = logReg.coef_[0][0]
theta2 = logReg.coef_[0][1]
theta0 = logReg.intercept_[0]
x2stan = -(theta1*x1stan + theta0)/theta2
#ax.set_xlabel('fixed acidity(stan)', fontsize = 18)
#ax.set_ylabel('volatile acidity(stan)', fontsize = 18)
plt.plot(x1stan,x2stan,marker='',linestyle='--', color='r',lw=2, label='decision boundary')
plt.legend()
plt.show()
wineData.head()
X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)

scaler = StandardScaler()
Xstan = scaler.fit_transform(X)
# save the standardized data for plotting
dataStan=pd.DataFrame(data = Xstan, columns = wineData.columns[0:11])
dataStan['category']=y
dataStan.head()
# fit model
logReg = LogisticRegression()
logReg.fit(Xstan,y)
logReg.intercept_, logReg.coef_
yhat = logReg.predict(Xstan)
dataStan['predict'] = yhat
dataStan.head()
TP = np.sum([(c==1 and p==1) for c,p in zip(dataStan['category'].values,dataStan['predict'].values)])
TN = np.sum([(c==0 and p==0) for c,p in zip(dataStan['category'].values,dataStan['predict'].values)])
FP = np.sum([(c==0 and p==1) for c,p in zip(dataStan['category'].values,dataStan['predict'].values)])
FN = np.sum([(c==1 and p==0) for c,p in zip(dataStan['category'].values,dataStan['predict'].values)])
P = np.sum(dataStan['category'].values)
N = len(dataStan['category'].values) - P
print('Precision is ',TP/(TP + FP))
print('True positive rate is ',TP/P)
print('Accuracy is ', (TP+TN)/(P+N))
print('False positive rate is ', FP/N)
phat = logReg.predict_proba(Xstan)[:,1]
print(phat)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(dataStan['category'].values, phat)
plt.plot(fpr, tpr)
plt.plot(FP/N, TP/P, marker='o', ms = 8, color = 'red', label = 'thresh. = 0.5')
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
from sklearn.metrics import auc
print('AUC is: ', auc(fpr,tpr))
c = []
for q in wineData['quality'].values:
    if q < 6:
        c.append(0)
    elif q > 6:
        c.append(2)
    else:
        c.append(1)

wineData['category'] = c
wineData.head()
X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values

scaler = StandardScaler()
Xstan = scaler.fit_transform(X)
dataStan=pd.DataFrame(data = Xstan, columns = wineData.columns[0:11])
dataStan['category']=y
dataStan.head()
softReg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
softReg.fit(Xstan,y)
softReg.intercept_,softReg.coef_
yhat = softReg.predict(Xstan)
dataStan['predict'] = yhat
dataStan.head()
from sklearn.metrics import confusion_matrix
C = confusion_matrix(dataStan['category'].values,yhat)
confusionMatrix = pd.DataFrame(data = C, index=['poor(0), true','good(1), true','great(2), true'], columns = ['poor(0), predicted','good(1), predicted','great(2), predicted'])
confusionMatrix.loc['sum'] = confusionMatrix.sum()
confusionMatrix['sum'] = confusionMatrix.sum(axis=1)
confusionMatrix
confMx = confusionMatrix.values[0:3,0:3]
plt.matshow(confMx, cmap=plt.cm.gray)
plt.show()
rowSums = confMx.sum(axis=1, keepdims=True) # contains number of samples for each true class
confMxNorm = confMx/rowSums
np.fill_diagonal(confMxNorm, 0)
plt.matshow(confMxNorm, cmap=plt.cm.gray)
plt.show()
