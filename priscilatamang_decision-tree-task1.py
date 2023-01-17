# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import math
data=pd.read_csv('/kaggle/input/play-tennis/play_tennis.csv')
data
data.head(5)
#Finding the no of people playing i.e Pyes and not playing i.e Pno
Pyes=data['play'].value_counts()[0]
Pno=data['play'].value_counts()[1]

#Finding the total number of people who played and didnt play
total=data.shape[0]

#Finding the probability of people playing i.e P_Pyes and of people not playing i.e P_Pno
P_Pyes=Pyes/total
P_Pno=Pno/total
print("Probability of playing:", P_Pyes)
print("Probability of not playing:", P_Pno)
E_P = -(P_Pyes * math.log(P_Pyes)) - (P_Pno * math.log(P_Pno))
print("Entropy:", E_P)
outlook=data['outlook'].value_counts()
print(outlook)
pd.crosstab(data['outlook'],data['play'])
#Calculating number of people played in Sunny i.e OS , in Rain OR and in Overcast OO
OS=data['outlook'].value_counts()[0]
OR=data['outlook'].value_counts()[1]
OO=data['outlook'].value_counts()[2]

#Finding their respective probabilities
P_OSyes=2/5
P_OSno=3/5
P_ORyes=3/5
P_ORno=2/5
P_OOyes=4/4
P_OOno=0/4

#Calculating Entropy of each child dataset i.e for Sunny, Rain and Overcast
E_OS=-(P_OSyes * math.log(P_OSyes)) - (P_OSno * math.log(P_OSno))
E_OR=-(P_ORyes * math.log(P_ORyes)) - (P_ORno * math.log(P_ORno))
E_OO=-(P_OOyes * math.log(P_OOyes)) - 0
print("Entropy for Sunny:", E_OS)
print("Entropy for Rain:", E_OR)
print("Entropy for Overcast:", E_OO)

#Weighted Entropy for column Outlook
WE_outlook= OS/total * E_OS + OR/total * E_OR + OO/total * E_OO
print("Weighted Entropy of Outlook:", WE_outlook)

#Information Gain for column Outlook
IG_outlook=E_P-WE_outlook
print("Information Gain of Outlook:",IG_outlook)
temp=data['temp'].value_counts()
print(temp)
pd.crosstab(data['temp'],data['play'])
#Calculating number of people played in Mild i.e TM, in Hot i.e TH and in Cool i.e TC
TM=data['temp'].value_counts()[0]
TH=data['temp'].value_counts()[1]
TC=data['temp'].value_counts()[2]

#Finding their respective probabilities
P_TMyes=4/6
P_TMno=2/6
P_THyes=2/4
P_THno=2/4
P_TCyes=3/4
P_TCno=1/4


#Calculating Entropy of each child dataset i.e for Mild, Hot and Cool
E_TM=-(P_TMyes * math.log(P_TMyes)) - (P_TMno * math.log(P_TMno))
E_TH=-(P_THyes * math.log(P_THyes)) - (P_THno * math.log(P_THno))
E_TC=-(P_TCyes * math.log(P_TCyes)) - (P_TCno * math.log(P_TCno))

print("Entropy for Mild:", E_TM)
print("Entropy for Hot:", E_TH)
print("Entropy for Cool:", E_TC)

#Weighted Entropy for column Temp
WE_temp= TM/total * E_TM + TH/total * E_TH + TC/total * E_TC
print("Weighted Entropy of Temp:", WE_temp)

#Information Gain for column Temp
IG_temp=E_P-WE_temp
print("Information Gain of Temp:",IG_temp)
humidity=data['humidity'].value_counts()
print(humidity)
pd.crosstab(data['humidity'],data['play'])
#Calculating number of people played in High Humidity i.e HH and in Normal Humidity i.e HN
HH=data['humidity'].value_counts()[0]
HN=data['humidity'].value_counts()[1]

#Finding their respective probabilities
P_HHyes=3/7
P_HHno=4/7
P_HNyes=6/7
P_HNno=1/7

#Calculating Entropy of each child dataset i.e for High and Normal
E_HH=-(P_HHyes * math.log(P_HHyes)) - (P_HHno * math.log(P_HHno))
E_HN=-(P_HNyes * math.log(P_HNyes)) - (P_HNno * math.log(P_HNno))

print("Entropy for High:", E_HH)
print("Entropy for Normal:", E_HN)


#Weighted Entropy for column Humidity
WE_humidity= HH/total * E_HH + HN/total * E_HN
print("Weighted Entropy of Humidity:", WE_humidity)

#Information Gain for column Humidity
IG_humidity=E_P-WE_humidity
print("Information Gain of Humidity:",IG_humidity)
wind=data['wind'].value_counts()
print(wind)
pd.crosstab(data['wind'],data['play'])
#Calculating number of people played in Strong Wind i.e WS and in Weak Wind i.e WW
WW=data['wind'].value_counts()[0]
WS=data['wind'].value_counts()[1]

#Finding their respective probabilities
P_WWyes=6/8
P_WWno=2/8
P_WSyes=3/6
P_WSno=3/6

#Calculating Entropy of each child dataset i.e for Strong Wind and Weak Wind
E_WW=-(P_WWyes * math.log(P_WWyes)) - (P_WWno * math.log(P_WWno))
E_WS=-(P_WSyes * math.log(P_WSyes)) - (P_WSno * math.log(P_WSno))

print("Entropy for Strong Wind:", E_WW)
print("Entropy for Weak Wind:", E_WS)


#Weighted Entropy for column Wind
WE_wind= WW/total * E_WW + WS/total * E_WS
print("Weighted Entropy of Wind:", WE_wind)

#Information Gain for column Wind
IG_wind=E_P-WE_wind
print("Information Gain of Wind:",IG_wind)
data[data["outlook"] == "Overcast"]
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
#Sorting the data on the basis of PetalLengthCm
df.sort_values("PetalLengthCm", inplace=True)
df.head()
df['Species'].value_counts()
total=df['Species'].shape[0]
print("Total is:",total)

#Probaility of being Iris-setosa i.e P_S, Iris-versicolor i.e P_VC and Iris-virginica i.e P_V
P_S=50/total
P_VC=50/total
P_V=50/total

E_S = -(P_S * math.log(P_S))-(P_VC * math.log(P_VC))-(P_V * math.log(P_V))
print("Entropy of Species:", E_S)
M_E = -1
B_P = -1
#First we split the datas on the basis of every rows of PetalLengthCm and make Yes or No table for storing it

def entropy(group):
    df = group.groupby("Species").count()["Id"] # Getting Number Of Occurence Of Each Species
    total = df.sum() # Total Number Of Rows
    entropy_sum = 0 # To store entropy
    for i in df :
        p = i/total
        entropy_sum = entropy_sum - (p*math.log(p))
    return(entropy_sum)


for i in df["PetalLengthCm"].unique():
    
    NO = df[(df["PetalLengthCm"] > i)]
    YES = df[~(df["PetalLengthCm"] > i)]
    
    len_NO = NO.shape[0]
    len_YES = YES.shape[0]
    
    E_NO= entropy(NO)
    E_YES= entropy(YES)
    
    WE = (len_NO/total * E_NO) + (len_YES/total * E_YES) #Finding Weighted Entropy
    IG = E_S - WE #Finding Information Gain
    
    print("Breaking Point is:", i)
    print("Weighted Entropy is:", WE)
    print("Information Gain is:", IG)
    print()
    
    if(IG > M_E): #Finding Maximum Breakpoint 
        M_E = IG
        B_P = i
print("Maximum Information Gain:", B_P)
for i in df[df["PetalLengthCm"] < 1.9]["Species"].tolist():
    print(i)