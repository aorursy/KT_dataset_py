import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from mlxtend.preprocessing import TransactionEncoder

print(os.listdir('../input/'))
data = pd.read_csv("../input/athlete_events.csv")

#data.info()

print(data.shape)

data.head(1)
print(data.shape)

print(data.columns)

print(data.isna().sum())
AgeMode = int(data['Age'].mode()[0])

print('Age Mode', AgeMode)

data['Age'] = data['Age'].fillna(AgeMode)

data['Age'].isna().sum()
HeightMode = data['Height'].mode()[0] ## 0 -> the column wise mode 

print('Height Mode',HeightMode)

data['Height'] = data['Height'].fillna(HeightMode)

data['Height'].isna().sum()
WeightMode = data['Weight'].mode()[0]

print('Weight Mode',WeightMode)

data['Weight'] = data['Weight'].fillna(WeightMode)

data['Weight'].isna().sum()
data['Medal'] = data['Medal'].fillna('None')

data['Medal'].isna().sum()
data.isna().sum()


# TO group a range of values together

def convertDataToRange(data,col,span):

    column = []

    for index,rows in data.iterrows():

        i = data.loc[index,col]

        temp = int(i/span)*span

        val = '(',col,':',str(temp),'-',str(temp+span),')'

        

        column.append("".join(val))

        #print(i,temp,val)

    return column       

def makeSelectiveSportDataset(fromData,sport,colList):

    toData = fromData[fromData['Sport'] == sport]

    toData =toData[colList]

    #to.head(5)

    med = ['Gold','Silver','Bronze']

    toData = toData[toData.Medal.isin(med)]

    return(toData)

def convertDataForApriori(data):

    data['Height']  = convertDataToRange(data,'Height',10)



    data['Weight']  = convertDataToRange(data,'Weight',10)



    data['Age']  = convertDataToRange(data,'Age',10)

    return data
def applyApriori(dataset,support):

    rec = dataset.values.tolist()

    # Finding Frequent Item Sets



    te = TransactionEncoder()

    te_ary = te.fit(rec).transform(rec)

    df = pd.DataFrame(te_ary, columns=te.columns_)

    

    # Applying Aprioi Algo

    from mlxtend.frequent_patterns import apriori

    

    freq_Itemsets =  apriori(df,min_support = support, use_colnames = True)

    freq_Itemsets['length'] = freq_Itemsets['itemsets'].apply(lambda x:len(x))

    return(freq_Itemsets[freq_Itemsets['length'] > 1])
Athletics = makeSelectiveSportDataset(data,'Athletics',['Sex','Age', 'Height','Weight','Team','Medal'])



Athletics= convertDataForApriori(Athletics)



freq_Itemsets_Athletics = applyApriori(Athletics,0.4)

print("Athletics Data")

print(freq_Itemsets_Athletics)
Archery = makeSelectiveSportDataset(data,'Archery',['Sex','Age', 'Height','Weight','Team','Medal'])

Archery= convertDataForApriori(Archery)

freq_Itemsets_Archery = applyApriori(Archery,0.4)

print("Archery Data")



print(freq_Itemsets_Archery)
BasketBall = makeSelectiveSportDataset(data,'Basketball',['Sex','Age', 'Height','Weight','Team','Medal'])



BasketBall= convertDataForApriori(BasketBall)



freq_Itemsets_BasketBall =  applyApriori(BasketBall,0.4)

print("Basket Ball Data")



freq_Itemsets_BasketBall
Boxing = makeSelectiveSportDataset(data,'Boxing',['Sex','Age', 'Height','Weight','Team','Medal'])

Boxing= convertDataForApriori(Boxing)



freq_Itemsets_Boxing = applyApriori(Boxing,0.4)

print("Boxing Data")

freq_Itemsets_Boxing
Cycling = makeSelectiveSportDataset(data,'Cycling',['Sex','Age', 'Height','Weight','Team','Medal'])



Cycling= convertDataForApriori(Cycling)

Cycling.head()



freq_Itemsets_Cycling = applyApriori(Cycling,0.4)

print("Cycling Data")



print(freq_Itemsets_Cycling)
Diving = makeSelectiveSportDataset(data,'Diving',['Sex','Age', 'Height','Weight','Team','Medal'])

Diving= convertDataForApriori(Diving)

Diving.head()



freq_Itemsets_Diving = applyApriori(Diving,0.4)

print("Diving Data")



freq_Itemsets_Diving
Football = makeSelectiveSportDataset(data,'Football',['Sex','Age', 'Height','Weight','Team','Medal'])

Football= convertDataForApriori(Football)

Football.head()



freq_Itemsets_Football = applyApriori(Football,0.4)

print("Footbal Data")

print(freq_Itemsets_Football)
Gymnastics = makeSelectiveSportDataset(data,'Gymnastics',['Sex','Age', 'Height','Weight','Team','Medal'])



Gymnastics= convertDataForApriori(Gymnastics)



freq_Itemsets_Gymnastics = applyApriori(Gymnastics,0.4)

print("Gymnastics Data")



print(freq_Itemsets_Gymnastics)
Handball = makeSelectiveSportDataset(data,'Handball',['Sex','Age', 'Height','Weight','Team','Medal'])

Handball= convertDataForApriori(Handball)



freq_Itemsets_Handball = applyApriori(Handball,0.4)

print("HandBall Data")



print(freq_Itemsets_Handball)
Judo = makeSelectiveSportDataset(data,'Judo',['Sex','Age', 'Height','Weight','Team','Medal'])

Judo= convertDataForApriori(Judo)



freq_Itemsets_Judo = applyApriori(Judo,0.4)

print("Judo Data")

print(freq_Itemsets_Judo)
SpeedSkt = makeSelectiveSportDataset(data,'Speed Skating',['Sex','Age', 'Height','Weight','Team','Medal'])

SpeedSkt= convertDataForApriori(SpeedSkt)

SpeedSkt.head()



freq_Itemsets_SpeedSkt = applyApriori(SpeedSkt,0.4)

print("Speed Skating Data")



print(freq_Itemsets_SpeedSkt)
'''

med = ['Gold','Silver','Bronze']

medalC1 = data[data.Medal.isin(med)]

medalC = medalC1.groupby(['Team']).count()

medalC.reset_index(drop = False,inplace = True)

medalC = medalC[['Team','Medal']]



#medalC = medalC.reset_index(drop=False)

medalCount = medalC[['Team','Medal']]

medalCount.sort_values('Team',inplace = True)

medalCount.head()

'''