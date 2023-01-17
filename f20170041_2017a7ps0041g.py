import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm
import os



df =pd.read_csv("/kaggle/input/dmassign1/data.csv")

data = pd.read_csv("/kaggle/input/dmassign1/data.csv")
df
df = df.filter(["Col42","Col45",'Col69', "Col84","Col85","Col86",'Col150'],axis=1)
df = df.replace(to_replace='?',value=np.nan)

data = data.replace(to_replace='?',value=np.nan)
for column in df.columns:

    df[column].fillna(df[column].mode()[0], inplace=True)



for column in data.columns:

    data[column].fillna(data[column].mode()[0], inplace=True)

    
for col in df.columns:

    for x in range(len(df[col])):

        if type(df[col][x]) == str:

            df[col].replace({df[col][x]: float(df[col][x])},inplace=True)
from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

scaled_data=scaler.fit(df).transform(df)

df=pd.DataFrame(scaled_data,columns=df.columns)

df.head()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)

df = pd.DataFrame(np_scaled)

df.head()
from sklearn.cluster import KMeans



wcss = []

for i in range(10, 20):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(df)

    wcss.append(kmean.inertia_)

    

plt.plot(range(10, 20),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
from sklearn.cluster import KMeans

#colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','brown','darksalmon','orange','crimson','seagreen','greenyellow','beige','khaki']



plt.figure(figsize=(33, 8))



kmean = KMeans(n_clusters = 33, random_state = 50)

kmean.fit(df)

pred = kmean.predict(df)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()







pred
len(pred)
dict0={1:0,2:0,3:0,4:0,5:0}

dict1={1:0,2:0,3:0,4:0,5:0}

dict2={1:0,2:0,3:0,4:0,5:0}

dict3={1:0,2:0,3:0,4:0,5:0}

dict4={1:0,2:0,3:0,4:0,5:0}

dict5={1:0,2:0,3:0,4:0,5:0}

dict6={1:0,2:0,3:0,4:0,5:0}

dict7={1:0,2:0,3:0,4:0,5:0}

dict8={1:0,2:0,3:0,4:0,5:0}

dict9={1:0,2:0,3:0,4:0,5:0}

dict10={1:0,2:0,3:0,4:0,5:0}

dict11={1:0,2:0,3:0,4:0,5:0}

dict12={1:0,2:0,3:0,4:0,5:0}

dict13={1:0,2:0,3:0,4:0,5:0}

dict14={1:0,2:0,3:0,4:0,5:0}

dict15={1:0,2:0,3:0,4:0,5:0}

dict16={1:0,2:0,3:0,4:0,5:0}

dict17={1:0,2:0,3:0,4:0,5:0}

dict18={1:0,2:0,3:0,4:0,5:0}

dict19={1:0,2:0,3:0,4:0,5:0}

dict20={1:0,2:0,3:0,4:0,5:0}

dict21={1:0,2:0,3:0,4:0,5:0}

dict22={1:0,2:0,3:0,4:0,5:0}

dict23={1:0,2:0,3:0,4:0,5:0}

dict24={1:0,2:0,3:0,4:0,5:0}

dict25={1:0,2:0,3:0,4:0,5:0}

dict26={1:0,2:0,3:0,4:0,5:0}

dict27={1:0,2:0,3:0,4:0,5:0}

dict28={1:0,2:0,3:0,4:0,5:0}

dict29={1:0,2:0,3:0,4:0,5:0}

dict30={1:0,2:0,3:0,4:0,5:0}

dict31={1:0,2:0,3:0,4:0,5:0}

dict32={1:0,2:0,3:0,4:0,5:0}





for i in range(1300):

    if pred[i] == 0:

           dict0[data["Class"][i]] += 1

    if pred[i] == 1:

           dict1[data["Class"][i]] += 1

    if pred[i] == 2:

           dict2[data["Class"][i]] += 1

    if pred[i] == 3:

           dict3[data["Class"][i]] += 1

    if pred[i] == 4:

           dict4[data["Class"][i]] += 1

    if pred[i] == 5:

           dict5[data["Class"][i]] += 1

    if pred[i] == 6:

           dict6[data["Class"][i]] += 1

    if pred[i] == 7:

           dict7[data["Class"][i]] += 1

    if pred[i] == 8:

           dict8[data["Class"][i]] += 1

    if pred[i] == 9:

           dict9[data["Class"][i]] += 1

    if pred[i] == 10:

           dict10[data["Class"][i]] += 1

    if pred[i] == 11:

           dict11[data["Class"][i]] += 1

    if pred[i] == 13:

           dict13[data["Class"][i]] += 1

    if pred[i] == 14:

           dict14[data["Class"][i]] += 1

    if pred[i] == 15:

           dict15[data["Class"][i]] += 1

    if pred[i] == 12:

           dict12[data["Class"][i]] += 1

    if pred[i] == 16:

           dict16[data["Class"][i]] += 1

    if pred[i] == 17:

           dict17[data["Class"][i]] += 1

    if pred[i] == 18:

           dict18[data["Class"][i]] += 1

    if pred[i] == 19:

           dict19[data["Class"][i]] += 1

    if pred[i] == 20:

           dict20[data["Class"][i]] += 1

    if pred[i] == 21:

           dict21[data["Class"][i]] += 1

    if pred[i] == 22:

           dict22[data["Class"][i]] += 1

    if pred[i] == 23:

           dict23[data["Class"][i]] += 1

    if pred[i] == 24:

           dict24[data["Class"][i]] += 1

    if pred[i] == 25:

           dict25[data["Class"][i]] += 1

    if pred[i] == 26:

           dict26[data["Class"][i]] += 1

    if pred[i] == 27:

           dict27[data["Class"][i]] += 1

    if pred[i] == 28:

           dict28[data["Class"][i]] += 1

    if pred[i] == 29:

           dict29[data["Class"][i]] += 1

    if pred[i] == 30:

           dict30[data["Class"][i]] += 1

    if pred[i] == 31:

           dict31[data["Class"][i]] += 1

    if pred[i] == 32:

           dict32[data["Class"][i]] += 1

            







            

        

    
print(0,dict0)

print(1,dict1)

print(2,dict2)

print(3,dict3)

print(4,dict4)

print(5,dict5)

print(6,dict6)

print(7,dict7)

print(8,dict8)

print(9,dict9)

print(10,dict10)

print(11,dict11)

print(12,dict12)

print(13,dict13)

print(14,dict14)

print(15,dict15)

print(16,dict16)

print(17,dict17)

print(18,dict18)

print(19,dict19)

print(20,dict20)

print(21,dict21)

print(22,dict22)

print(23,dict23)

print(24,dict24)

print(25,dict25)

print(26,dict26)

print(27,dict27)

print(28,dict28)

print(29,dict29)

print(30,dict30)

print(31,dict31)

print(32,dict32)



data0 =[]

data1 =[]

data7 =[]

data9 =[]

data13=[]

data17=[]

data31=[]
for i in range(13000):

    if pred[i]==0:

        data0.append(df.iloc[i])

    if pred[i]==1:

        data1.append(df.iloc[i])

    if pred[i]==7:

        data7.append(df.iloc[i])

    if pred[i]==9:

        data9.append(df.iloc[i])

    if pred[i]==13:

        data13.append(df.iloc[i])

    if pred[i]==17:

        data17.append(df.iloc[i])

    if pred[i]==31:

        data31.append(df.iloc[i])
kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data0)

pred0 = kmean.predict(data0)

pred_pd = pd.DataFrame(pred0)







kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data1)

pred1 = kmean.predict(data1)

pred_pd = pd.DataFrame(pred1)







kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data7)

pred7 = kmean.predict(data7)

pred_pd = pd.DataFrame(pred7)







kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data9)

pred9 = kmean.predict(data9)

pred_pd = pd.DataFrame(pred9)







kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data13)

pred13 = kmean.predict(data13)

pred_pd = pd.DataFrame(pred13)





kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data17)

pred17 = kmean.predict(data17)

pred_pd = pd.DataFrame(pred17)





kmean = KMeans(n_clusters = 8, random_state = 50)

kmean.fit(data31)

pred31 = kmean.predict(data31)

pred_pd = pd.DataFrame(pred31)

dict00={1:0,2:0,3:0,4:0,5:0}

dict01={1:0,2:0,3:0,4:0,5:0}

dict02={1:0,2:0,3:0,4:0,5:0}

dict03={1:0,2:0,3:0,4:0,5:0}

dict04={1:0,2:0,3:0,4:0,5:0}

dict05={1:0,2:0,3:0,4:0,5:0}

dict06={1:0,2:0,3:0,4:0,5:0}

dict07={1:0,2:0,3:0,4:0,5:0}



dict10={1:0,2:0,3:0,4:0,5:0}

dict11={1:0,2:0,3:0,4:0,5:0}

dict12={1:0,2:0,3:0,4:0,5:0}

dict13={1:0,2:0,3:0,4:0,5:0}

dict14={1:0,2:0,3:0,4:0,5:0}

dict15={1:0,2:0,3:0,4:0,5:0}

dict16={1:0,2:0,3:0,4:0,5:0}

dict17={1:0,2:0,3:0,4:0,5:0}









dict70={1:0,2:0,3:0,4:0,5:0}

dict71={1:0,2:0,3:0,4:0,5:0}

dict72={1:0,2:0,3:0,4:0,5:0}

dict73={1:0,2:0,3:0,4:0,5:0}

dict74={1:0,2:0,3:0,4:0,5:0}

dict75={1:0,2:0,3:0,4:0,5:0}

dict76={1:0,2:0,3:0,4:0,5:0}

dict77={1:0,2:0,3:0,4:0,5:0}



dict90={1:0,2:0,3:0,4:0,5:0}

dict91={1:0,2:0,3:0,4:0,5:0}

dict92={1:0,2:0,3:0,4:0,5:0}

dict93={1:0,2:0,3:0,4:0,5:0}

dict94={1:0,2:0,3:0,4:0,5:0}

dict95={1:0,2:0,3:0,4:0,5:0}

dict96={1:0,2:0,3:0,4:0,5:0}

dict97={1:0,2:0,3:0,4:0,5:0}











dict130={1:0,2:0,3:0,4:0,5:0}

dict131={1:0,2:0,3:0,4:0,5:0}

dict132={1:0,2:0,3:0,4:0,5:0}

dict133={1:0,2:0,3:0,4:0,5:0}

dict134={1:0,2:0,3:0,4:0,5:0}

dict135={1:0,2:0,3:0,4:0,5:0}

dict136={1:0,2:0,3:0,4:0,5:0}

dict137={1:0,2:0,3:0,4:0,5:0}











dict170={1:0,2:0,3:0,4:0,5:0}

dict171={1:0,2:0,3:0,4:0,5:0}

dict172={1:0,2:0,3:0,4:0,5:0}

dict173={1:0,2:0,3:0,4:0,5:0}

dict174={1:0,2:0,3:0,4:0,5:0}

dict175={1:0,2:0,3:0,4:0,5:0}

dict176={1:0,2:0,3:0,4:0,5:0}

dict177={1:0,2:0,3:0,4:0,5:0}













dict310={1:0,2:0,3:0,4:0,5:0}

dict311={1:0,2:0,3:0,4:0,5:0}

dict312={1:0,2:0,3:0,4:0,5:0}

dict313={1:0,2:0,3:0,4:0,5:0}

dict314={1:0,2:0,3:0,4:0,5:0}

dict315={1:0,2:0,3:0,4:0,5:0}

dict316={1:0,2:0,3:0,4:0,5:0}

dict317={1:0,2:0,3:0,4:0,5:0}











c0 =0

c1=0

c7=0

c9 =0

c13=0

c17=0

c31=0

for i in range(1300):

    if pred[i]==0:

        if pred0[c0]==0:

            dict00[data["Class"][i]] += 1

        if pred0[c0]==1:

            dict01[data["Class"][i]] += 1

        if pred0[c0]==2:

            dict02[data["Class"][i]] += 1

        if pred0[c0]==3:

            dict03[data["Class"][i]] += 1

        if pred0[c0]==4:

            dict04[data["Class"][i]] += 1

        if pred0[c0]==5:

            dict05[data["Class"][i]] += 1

        if pred0[c0]==6:

            dict06[data["Class"][i]] += 1

        if pred0[c0]==7:

            dict07[data["Class"][i]] += 1

        c0+=1

    if pred[i]==1:

        if pred1[c1]==0:

            dict10[data["Class"][i]] += 1

        if pred1[c1]==1:

            dict11[data["Class"][i]] += 1

        if pred1[c1]==2:

            dict12[data["Class"][i]] += 1

        if pred1[c1]==3:

            dict13[data["Class"][i]] += 1

        if pred1[c1]==4:

            dict14[data["Class"][i]] += 1

        if pred1[c1]==5:

            dict15[data["Class"][i]] += 1

        if pred1[c1]==6:

            dict16[data["Class"][i]] += 1

        if pred1[c1]==7:

            dict17[data["Class"][i]] += 1

        c1+=1

    if pred[i]==7:

        if pred7[c7]==0:

            dict70[data["Class"][i]] += 1

        if pred7[c7]==1:

            dict71[data["Class"][i]] += 1

        if pred7[c7]==2:

            dict72[data["Class"][i]] += 1

        if pred7[c7]==3:

            dict73[data["Class"][i]] += 1

        if pred7[c7]==4:

            dict74[data["Class"][i]] += 1

        if pred7[c7]==5:

            dict75[data["Class"][i]] += 1

        if pred7[c7]==6:

            dict76[data["Class"][i]] += 1

        if pred7[c7]==7:

            dict77[data["Class"][i]] += 1

        c7+=1

    if pred[i]==9:

        if pred9[c9]==0:

            dict90[data["Class"][i]] += 1

        if pred9[c9]==1:

            dict91[data["Class"][i]] += 1

        if pred9[c9]==2:

            dict92[data["Class"][i]] += 1

        if pred9[c9]==3:

            dict93[data["Class"][i]] += 1

        if pred9[c9]==4:

            dict94[data["Class"][i]] += 1

        if pred9[c9]==5:

            dict95[data["Class"][i]] += 1

        if pred9[c9]==6:

            dict96[data["Class"][i]] += 1

        if pred9[c9]==7:

            dict97[data["Class"][i]] += 1

        c9+=1

    if pred[i]==13:

        if pred13[c13]==0:

            dict130[data["Class"][i]] += 1

        if pred13[c13]==1:

            dict131[data["Class"][i]] += 1

        if pred13[c13]==2:

            dict132[data["Class"][i]] += 1

        if pred13[c13]==3:

            dict133[data["Class"][i]] += 1

        if pred13[c13]==4:

            dict134[data["Class"][i]] += 1

        if pred13[c13]==5:

            dict135[data["Class"][i]] += 1

        if pred13[c13]==6:

            dict136[data["Class"][i]] += 1

        if pred13[c13]==7:

            dict137[data["Class"][i]] += 1

        c13+=1

    if pred[i]==17:

        if pred17[c17]==0:

            dict170[data["Class"][i]] += 1

        if pred17[c17]==1:

            dict171[data["Class"][i]] += 1

        if pred17[c17]==2:

            dict172[data["Class"][i]] += 1

        if pred17[c17]==3:

            dict173[data["Class"][i]] += 1

        if pred17[c17]==4:

            dict174[data["Class"][i]] += 1

        if pred17[c17]==5:

            dict175[data["Class"][i]] += 1

        if pred17[c17]==6:

            dict176[data["Class"][i]] += 1

        if pred17[c17]==7:

            dict177[data["Class"][i]] += 1

        c17+=1

        

    if pred[i]==31:

        if pred31[c31]==0:

            dict310[data["Class"][i]] += 1

        if pred31[c31]==1:

            dict311[data["Class"][i]] += 1

        if pred31[c31]==2:

            dict312[data["Class"][i]] += 1

        if pred31[c31]==3:

            dict313[data["Class"][i]] += 1

        if pred31[c31]==4:

            dict314[data["Class"][i]] += 1

        if pred31[c31]==5:

            dict315[data["Class"][i]] += 1

        if pred31[c31]==6:

            dict316[data["Class"][i]] += 1

        if pred31[c31]==7:

            dict317[data["Class"][i]] += 1

        c31+=1











        

        
print("00",dict00)

print("01",dict01)

print("02",dict02)

print("03",dict03)

print("04",dict04)

print("05",dict05)

print("06",dict06)

print("07",dict07)



print("***********************")

print("10",dict10)

print("11",dict11)

print("12",dict12)

print("13",dict13)

print("14",dict14)

print("15",dict15)

print("16",dict16)

print("17",dict17)



print("***********************")







print("70",dict70)

print("71",dict71)

print("72",dict72)

print("73",dict73)

print("74",dict74)

print("75",dict75)

print("76",dict76)

print("77",dict77)



print("***********************")





print("90",dict90)

print("91",dict91)

print("92",dict92)

print("93",dict93)

print("94",dict94)

print("95",dict95)

print("96",dict96)

print("97",dict97)



print("***********************")



print("130",dict130)

print("131",dict131)

print("132",dict132)

print("133",dict133)

print("134",dict134)

print("135",dict135)

print("136",dict136)

print("137",dict137)



print("***********************")





print("170",dict170)

print("171",dict171)

print("172",dict172)

print("173",dict173)

print("174",dict174)

print("175",dict175)

print("176",dict176)

print("177",dict177)









print("***********************")

print("310",dict310)

print("311",dict311)

print("312",dict312)

print("313",dict313)

print("314",dict314)

print("315",dict315)

print("316",dict316)

print("317",dict317)



res=[]
c0 =0

c1=0

c7=0

c9 =0

c13=0

c17=0

c31=0

for i in range(len(pred)):

    if pred[i] == 0:

        if pred0[c0]==0:

            res.append(4)

        if pred0[c0]==1:

            res.append(2)

        if pred0[c0]==2:

            res.append(5)

        if pred0[c0]==3:

            res.append(2)

        if pred0[c0]==4:

            res.append(5)

        if pred0[c0]==5:

            res.append(5)

        if pred0[c0]==6:

            res.append(5)

        if pred0[c0]==7:

            res.append(2)

        c0+=1

        

    if pred[i] == 1:

        if pred1[c1]==0:

            res.append(2)

        if pred1[c1]==1:

            res.append(4)

        if pred1[c1]==2:

            res.append(3)

        if pred1[c1]==3:

            res.append(1)

        if pred1[c1]==4:

            res.append(4)

        if pred1[c1]==5:

            res.append(4)

        if pred1[c1]==6:

            res.append(4)

        if pred1[c1]==7:

            res.append(4)

        c1+=1

    if pred[i] == 2:

        res.append(1)

    if pred[i] == 3:

        res.append(1)

    if pred[i] == 4:

        res.append(1);

    if pred[i] == 5:

        res.append(1)

    if pred[i] == 6:

        res.append(1)

    if pred[i] == 7:

        if pred7[c7]==0:

            res.append(3)

        if pred7[c7]==1:

            res.append(2)

        if pred7[c7]==2:

            res.append(3)

        if pred7[c7]==3:

            res.append(1)

        if pred7[c7]==4:

            res.append(4)

        if pred7[c7]==5:

            res.append(4)

        if pred7[c7]==6:

            res.append(1)

        if pred7[c7]==7:

            res.append(3)

        c7+=1

    if pred[i] == 8:

        res.append(1)

    if pred[i] == 9:

        if pred9[c9]==0:

            res.append(1)

        if pred9[c9]==1:

            res.append(4)

        if pred9[c9]==2:

            res.append(4)

        if pred9[c9]==3:

            res.append(4)

        if pred9[c9]==4:

            res.append(1)

        if pred9[c9]==5:

            res.append(1)

        if pred9[c9]==6:

            res.append(4)

        if pred9[c9]==7:

            res.append(1)

        c9+=1

    if pred[i] == 10:

        res.append(1)

    if pred[i] == 11:

        res.append(1)

    if pred[i] == 12:

        res.append(1)

    if pred[i] == 13:

        if pred13[c13]==0:

            res.append(3)

        if pred13[c13]==1:

            res.append(5)

        if pred13[c13]==2:

            res.append(4)

        if pred13[c13]==3:

            res.append(4)

        if pred13[c13]==4:

            res.append(4)

        if pred13[c13]==5:

            res.append(2)

        if pred13[c13]==6:

            res.append(2)

        if pred13[c13]==7:

            res.append(2)

        c13+=1

    if pred[i] == 14:

        res.append(1)

    if pred[i] == 15:

        res.append(1)

    if pred[i] == 16:

        res.append(1)

    if pred[i] == 17:

        if pred17[c17]==0:

            res.append(1)

        if pred17[c17]==1:

            res.append(1)

        if pred17[c17]==2:

            res.append(1)

        if pred17[c17]==3:

            res.append(1)

        if pred17[c17]==4:

            res.append(1)

        if pred17[c17]==5:

            res.append(1)

        if pred17[c17]==6:

            res.append(1)

        if pred17[c17]==7:

            res.append(1)

        c17+=1

    if pred[i] == 18:

        res.append(1)

    if pred[i] == 19:

        res.append(1)

    if pred[i] == 20:

        res.append(1)

    if pred[i] == 21:

        res.append(1)

    if pred[i] == 22:

        res.append(1)

    if pred[i] == 23:

        res.append(1)

    if pred[i] == 24:

        res.append(1)

    if pred[i] == 25:

        res.append(1)

    if pred[i] == 26:

        res.append(1)

    if pred[i] == 27:

        res.append(1)

    if pred[i] == 28:

        res.append(1)

    if pred[i] == 29:

        res.append(1)

    if pred[i] == 30:

        res.append(1)

    if pred[i] == 31:

        if pred31[c31]==0:

            res.append(3)

        if pred31[c31]==1:

            res.append(2)

        if pred31[c31]==2:

            res.append(3)

        if pred31[c31]==3:

            res.append(4)

        if pred31[c31]==4:

            res.append(5)

        if pred31[c31]==5:

            res.append(1)

        if pred31[c31]==6:

            res.append(4)

        if pred31[c31]==7:

            res.append(3)

        c31+=1

    if pred[i] == 32:

        res.append(1)

    if pred[i] == 33:

        res.append(1)

   

len(res)
ans = res
ans1 = pd.DataFrame(ans)

final = pd.concat([data['ID'], ans1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final = final[1300:]
final.head()
final.to_csv('submissionAFINAL3Portable.csv', index = False)
from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(final)