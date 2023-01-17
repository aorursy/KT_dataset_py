import pandas as pd

from sklearn import preprocessing

from matplotlib import pyplot as plt

from sklearn import covariance

import seaborn as sns

import numpy as np

from numpy import linalg as LA
Input_Data = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")
Input_Data.describe()
Input_Data.head()
Input_Data = Input_Data[Input_Data.Customer_Segment.isin([1,2])]
DataColumnNames = Input_Data.columns

Input_Data_List_Temp = []

for DataColumnName in DataColumnNames:

    Input_Data_List_Temp.append(preprocessing.normalize([Input_Data[DataColumnName].to_list()],norm='max')[0])

Input_Data_List = np.transpose(Input_Data_List_Temp)
plt.figure(figsize=(10,10))

sns.set(font_scale=1.5)

hm = sns.heatmap(pd.DataFrame(Input_Data_List).corr(),annot=True,annot_kws={"size":8},xticklabels=DataColumnNames, yticklabels=DataColumnNames)

plt.title('Covariance matrix showing correlation coefficients')

plt.tight_layout()

plt.show()
Customer_Segment = Input_Data['Customer_Segment'].to_list()

Ash_Alcanity = Input_Data['Ash_Alcanity'].to_list()

Proline = Input_Data['Proline'].to_list()

Ash_Alcanity_Normalized = preprocessing.normalize([Ash_Alcanity],norm='max')[0]

Proline_Normalized = preprocessing.normalize([Proline],norm='max')[0]



Ash_Alcanity_Class1 = []

Proline_Class1 = []

Ash_Alcanity_Class2 = []

Proline_Class2 = []



for i in range(0,len(Ash_Alcanity_Normalized) - 1):

    if Customer_Segment[i] == 1:

        Ash_Alcanity_Class1.append(Ash_Alcanity_Normalized[i])

        

for i in range(0,len(Proline_Normalized) - 1):

    if Customer_Segment[i] == 1:

        Proline_Class1.append(Proline_Normalized[i])



for i in range(0,len(Ash_Alcanity_Normalized) - 1):

    if Customer_Segment[i] == 2:

        Ash_Alcanity_Class2.append(Ash_Alcanity_Normalized[i])

        

for i in range(0,len(Proline_Normalized) - 1):

    if Customer_Segment[i] == 2:

        Proline_Class2.append(Proline_Normalized[i])



Ash_AlcanityMean = np.array(Ash_Alcanity_Normalized).mean()

ProlineMean = np.array(Proline_Normalized).mean()

Ash_AlcanityMean1 = np.array(Ash_Alcanity_Class1).mean()

Ash_AlcanityMean2 = np.array(Ash_Alcanity_Class2).mean()

ProlineMean1 = np.array(Proline_Class1).mean()

ProlineMean2 = np.array(Proline_Class2).mean()
plt.scatter(Ash_Alcanity_Class1,Proline_Class1,marker='x')

plt.scatter(Ash_Alcanity_Class2,Proline_Class2,marker='o')
Ash_Alcanity_Class1_Subtracted = [x - Ash_AlcanityMean1 for x in Ash_Alcanity_Class1]

Proline_Class1_Subtracted = [x - ProlineMean1 for x in Proline_Class1]

S_w1 = np.array([Ash_Alcanity_Class1_Subtracted,Proline_Class1_Subtracted]).dot(np.array([Ash_Alcanity_Class1_Subtracted,Proline_Class1_Subtracted]).T)
Ash_Alcanity_Class2_Subtracted = [x - Ash_AlcanityMean2 for x in Ash_Alcanity_Class2]

Proline_Class2_Subtracted = [x - ProlineMean2 for x in Proline_Class2]

S_w2 = np.array([Ash_Alcanity_Class2_Subtracted,Proline_Class2_Subtracted]).dot(np.array([Ash_Alcanity_Class2_Subtracted,Proline_Class2_Subtracted]).T)
S_w = S_w1 + S_w2
Class1_Mean_Subtracted = [Ash_AlcanityMean1 - Ash_AlcanityMean, ProlineMean1 - ProlineMean]

Class2_Mean_Subtracted = [Ash_AlcanityMean2 - Ash_AlcanityMean, ProlineMean2 - ProlineMean]
S_b = np.array([Class1_Mean_Subtracted, Class2_Mean_Subtracted]).T.dot(np.array([Class1_Mean_Subtracted, Class2_Mean_Subtracted]))
S = np.linalg.inv(S_w).dot(S_b)
EigenValue,EigenVector = LA.eig(S)

print(EigenVector)

print(EigenValue)
plt.scatter([x - Ash_AlcanityMean for x in Ash_Alcanity_Class1],[x - ProlineMean for x in Proline_Class1],marker='x')

plt.scatter([x - Ash_AlcanityMean for x in Ash_Alcanity_Class2],[x - ProlineMean for x in Proline_Class2],marker='o')

plt.quiver([0, 0], [0, 0], EigenVector[1], EigenVector[0], scale=4, color=['#330033','g'], label=['Vector1','Vector2'])

plt.text(0.0,-0.2,'Vector1')

plt.text(-0.2,0.2,'Vector2')
LDAProjectionClass1 = EigenVector.T.dot(np.array([Ash_Alcanity_Class1, Proline_Class1]))

LDAProjectionClass2 = EigenVector.T.dot(np.array([Ash_Alcanity_Class2, Proline_Class2]))
plt.scatter(LDAProjectionClass1[1],LDAProjectionClass1[0],marker='x')

plt.scatter(LDAProjectionClass2[1],LDAProjectionClass2[0],marker='o')