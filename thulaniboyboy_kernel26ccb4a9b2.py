# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
testData  = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv" , index_col = 'id')
trainData  = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv", index_col = 'id')
combinedData = pd.concat([ testData, trainData] , sort = False, ignore_index = True )
#creating new index
ts = list(range(0, 1000000))
df = pd.DataFrame( ts , columns = ["id"])
df.head()
comData = pd.concat([df, combinedData], axis = 1) 
comData.set_index('id', inplace=True)
#Slicing Data into different Dataframes
comBinary1Data = comData.iloc[ : , 0 : 3]
comBinary2Data = comData.iloc[ : , 3 : 5]
comNominalData = comData.iloc[ : , 5 : 15]
comOrdinalData = comData.iloc[ : , 15 : 21]
comCyclicData = comData.iloc[ : , 21 : 23]
targetData = comData.iloc[ : ,  23]

#handling missing data 
comBinary1Data.fillna(0 , inplace =True)
comBinary2Data.replace({ 'bin_3' : 'F' , 'bin_4' : 'N'} , 0, inplace = True)
comBinary2Data.replace({ 'bin_3' : 'T' , 'bin_4' : 'Y'} , 1, inplace = True)
comBinary2Data.fillna(method = 'ffill' , inplace =True)

comBinary2Data.head(50)
comNominalData.nom_0.fillna(method = "ffill", inplace= True)
comNominalData.nom_2.fillna("Lion", inplace= True)
comNominalData.nom_3.fillna("India", inplace= True)
comNominalData.nom_4.fillna("Piano", inplace= True)
comNominalData.iloc[ : , 5 : 10].fillna( '0', inplace= True)
comNominalData.nom_1.fillna( "Polygon" , inplace =True)
comOrdinalData.ord_0.fillna(method = "ffill", inplace= True)
comOrdinalData.ord_1.fillna(method = "ffill", inplace= True)
comOrdinalData.ord_2.fillna(method = "ffill", inplace= True)
comOrdinalData.ord_3.fillna(method = "ffill", inplace= True)
comOrdinalData.ord_4.fillna(method = "ffill", inplace= True)
comOrdinalData.ord_5.fillna(method = "ffill", inplace= True)
comCyclicData.day.fillna(method = "ffill", inplace= True)
comCyclicData.month.fillna(method = "ffill", inplace= True)
finalComData = pd.concat([comBinary1Data,
comBinary2Data,
comNominalData,
comOrdinalData,
comCyclicData,
targetData] , axis = 1)
finalComData.head(50)
finalTrainData = finalComData[400000 : ]
finalTrainData.tail()
finalTestData = finalComData[0:400000].drop('target' , axis = 1)
finalTestData.head()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X=finalTrainData.drop(['target'],axis=1)
y=finalTrainData['target']

x=y.value_counts()
plt.bar(x.index,x)
plt.gca().set_xticks([0,1])
plt.title('distribution of target variable')
plt.show()
x
from sklearn.preprocessing import LabelEncoder
le=pd.DataFrame()
label=LabelEncoder()
for c in  X.columns:
    if(X[c].dtype=='object'):
        le[c]=label.fit_transform(X[c])
    else:
        le[c]=X[c]
        
le.head(3)
le_test = pd.DataFrame()
for c in  finalTestData.columns:
    if(finalTestData[c].dtype=='object'):
        le_test[c]=label.fit_transform(finalTestData[c])
    else:
        le_test[c]=finalTestData[c]
        
le_test.head(3)

lr=LogisticRegression()
lr.fit(le,y)
target_pre=lr.predict(le_test)
len(target_pre)
y.head(20)
print('Accuracy : ',lr.score(le_test,target_pre))
le_test[100:200]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y[: 400000],target_pre )

import seaborn as sn
plt.figure(figsize = (10 , 7))
sn.heatmap(cm, annot =True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

