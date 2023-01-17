# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df=pd.read_csv("/kaggle/input/play-tennis/play_tennis.csv")
df
df.shape
df.groupby(['play']).size()

#Probability of yes and no
pyes=9/14
pno=5/14

Ep=float(-pyes*(math.log(pyes))-pno*(math.log(pno)))
print("The parent entropy is: ",Ep)
#FOR OUTLOOK COLUMN
pd.crosstab(df['outlook'],df['play'])

outlook=df['outlook'].value_counts()
print(outlook)
po=4/14
pr=5/14
ps=5/14
POYES=4/4
PRYES=3/5
PSYES=2/5

PONO=0
PRNO=2/5
PSNO=3/5

EO=float(-POYES*(math.log(POYES))-0)
print("Entropy for overcast is:",EO)

ER=float(-PRYES*(math.log(PRYES))-PRNO*(math.log(PRNO)))
print("Entropy for rainy is:",ER)

ES=float(-PSYES*(math.log(PSYES))-PSNO*(math.log(PSNO)))
print("Entroopy for sunny is:",ES)

WAVG_OUTLOOK=((EO*po)+(ER*pr)+(ES*ps))
print("Weighted Average for the column outlook is:",WAVG_OUTLOOK)

INFO_GAIN_OUTLOOK=Ep-WAVG_OUTLOOK
print("Information gain for the column outlook is:",INFO_GAIN_OUTLOOK)


#FOR THE COLUMN TEMP
pd.crosstab(df['temp'],df['play'])

temp=df['temp'].value_counts()
print(temp)
pm=6/14
pc=4/14
ph=4/14
PMYES=4/6
PCYES=3/4
PHYES=2/4

PMNO=2/6
PCNO=1/4
PHNO=2/4

EM=float(-PMYES*(math.log(PMYES))-PMNO*(math.log(PMNO)))
print("Entropy for mild is:",EM)

EC=float(-PCYES*(math.log(PCYES))-PCNO*(math.log(PCNO)))
print("Entropy for cool is:",EC)

EH=float(-PHYES*(math.log(PHYES))-PHNO*(math.log(PHNO)))
print("Entroopy for hot is:",EH)

WAVG_TEMP=((EM*pm)+(EC*pc)+(EH*ph))
print("Weighted Average for the column temp is:",WAVG_TEMP)

INFO_GAIN_TEMP=Ep-WAVG_TEMP
print("Information gain for the column temp is:",INFO_GAIN_TEMP)


#FOR THE COLUMN HUMIDITY
pd.crosstab(df['humidity'],df['play'])
humidity=df['humidity'].value_counts()
print(humidity)
pn=7/14
ph=7/14
PNYES=6/7
PHYES=3/7

PNNO=1/7
PHNO=4/7

EN=float(-PNYES*(math.log(PNYES))-PNNO*(math.log(PNNO)))
print("Entropy for normal is:",EN)

EH=float(-PHYES*(math.log(PHYES))-PHNO*(math.log(PHNO)))
print("Entroopy for high is:",EH)

WAVG_HUMIDITY=((EN*pn)+(EH*ph))
print("Weighted Average for the column humidity is:",WAVG_HUMIDITY)

INFO_GAIN_HUMIDITY=Ep-WAVG_HUMIDITY
print("Information gain for the column humidity is:",INFO_GAIN_HUMIDITY)


#FOR THE COLUMN WIND
pd.crosstab(df['wind'],df['play'])
wind=df['wind'].value_counts()
print(wind)
ps=6/14
pw=8/14
PWYES=6/8
PSYES=3/6

PWNO=2/8
PSNO=3/6

EW=float(-PWYES*(math.log(PWYES))-PWNO*(math.log(PWNO)))
print("Entropy for weak is:",EN)

ES=float(-PSYES*(math.log(PSYES))-PSNO*(math.log(PSNO)))
print("Entroopy for strong is:",EH)

WAVG_WIND=((EW*pw)+(ES*ps))
print("Weighted Average for the column wind is:",WAVG_WIND)

INFO_GAIN_WIND=Ep-WAVG_WIND
print("Information gain for the column wind is:",INFO_GAIN_WIND)


print("Information gain for the column outlook is:",INFO_GAIN_OUTLOOK)
print("Information gain for the column temp is:",INFO_GAIN_TEMP)
print("Information gain for the column humidity is:",INFO_GAIN_HUMIDITY)
print("Information gain for the column wind is:",INFO_GAIN_WIND)
df[df['outlook']=='Overcast']
df1=pd.read_csv('/kaggle/input/iris/Iris.csv')
df1
df1.shape
df1.sort_values('PetalLengthCm',inplace=True)
df1
df1['Species'].value_counts()
psetosa=50/150
pversicolor=50/150
pvirginica=50/150

E_Species=float(-psetosa*(math.log(psetosa))-pversicolor*(math.log(pversicolor))-pvirginica*(math.log(pvirginica)))
print("Entropy of the species is:",E_Species)
MAX_ENTRO=-1
POINT=-1
def Entropy(data):
    df1 = data.groupby("Species").count()["Id"] 
    total = df1.sum() 
    E_sum = 0 
    for i in df1 :
        p = i/total
        E_sum = E_sum - (p*math.log(p))
    return(E_sum) 
for i in df1["PetalLengthCm"].unique():
    NO = df1[(df1["PetalLengthCm"] > i)]
    YES = df1[~(df1["PetalLengthCm"] > i)]
            
    TNO=NO.shape[0]
    TYES=YES.shape[0]
    
    print("Total length of yes:",TYES)
    print("Total length of no:",TNO)
    
    EYES=Entropy(YES)
    ENO=Entropy(NO)
    
    PYES=TYES/150
    PNO=TNO/150
    
    W_AVG=(EYES*PYES)+(ENO*PNO)
           
    INFOGAIN=E_Species-W_AVG
           
    print("Breaking Point is:",i)        
    print("Weighted average is:",W_AVG)        
    print("Information Gain is:",INFOGAIN)
    print()
    print()
                    
    if(INFOGAIN>MAX_ENTRO):
            MAX_ENTRO=INFOGAIN
            POINT=i 
           
           
           
           
           
           
           
           
    
    
 
   
print("Point where the maximum Information gain occurs is:",POINT)
for i in df1[df1["PetalLengthCm"] < 1.9]["Species"].tolist():
    print(i)
    
    
    
            
