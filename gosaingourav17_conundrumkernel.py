import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')
cities = pd.read_csv("../input/title12/citiesn.csv")
pred801 = pd.read_csv("../input/finalc/genie_lb_price_predictionsn.csv")
real = pd.read_csv("../input/finalc/actual_train_pricen.csv")
pred800 = pd.read_csv("../input/finalc/genie_train_price_predictionsn.csv")
output=pd.read_csv("../input/example1/example1.csv")
result=pd.read_csv("../input/submissions/outputfinal0")
pred802=pd.read_csv("../input/day802/genie_surprise_price_predictions.csv")
print(output.iloc[2][0])
output.insert(1,'price',0,)
output.insert(2,'petrol',0,)
output.price = output.price.astype(float)
output.petrol = output.petrol.astype(float)
output.dtypes


kn=[]
for i in range(0,500):
    kn.append(0)
for i in range(0,500):
    kn[i]=int(output.iloc[i][0])
kn[499]=337
output.city=kn
output.tail()

x=0.0
i=0

f = []

for i in range(0,500):
       f.append(0)

for i in range(0,499):
    x=0
    if(pred802.iloc[i][0]<=3):
        x=(pred802.iloc[i][0])
        f[i]=(x*1.10)
        x=0.0


pred802.head()

p=0
kc=[]
for i in range(0,500):
    kc.append(0)
for i in range(0,500):
    p=int((output.iloc[i][0])-1)
    kc[i]=f[p]
print(kc)






output.price=kc
print(output.head(100))
import math
i=0
l=0
flag=0
x2=0.0
y2=0.0
x1=0.0
y1=0.0
city=0
distance=0.0
petrol=0.0
howmuch=[]
d=0
for d in range(0,500):
    howmuch.append(0.0)
for i in range(0,499):
    if(output.iloc[i][1]>0):
        
        l=i+1
        flag=0
    
        while(flag==0):
            if(output.iloc[l][1]>0):
                city=int(output.iloc[l][0]-1)
                x2=cities.iloc[city][0]
                y2=cities.iloc[city][1]
                x1=cities.iloc[int(output.iloc[i][0])-1][0]
                y1=cities.iloc[int(output.iloc[i][0])-1][1]
                distance=math.sqrt(abs(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1))))
                if((distance/10.0)<30):
                
                    howmuch[i]=distance/10.0
                
                else:
                    howmuch[i]=30.0
                    
                flag=1
            
            else:
                l=l+1
        
        
            
        
            
print(output[420:])
howmuch[499]=30

print(pred801.iloc[237][0])

output.petrol=howmuch
for i in range(0,498):
    if(output.iloc[i][2]>=29):
        print(i)
olo=pred801.iloc[464][0]
for i in range(234,238):
    if((pred801.iloc[int((output.iloc[i][0])-1)][0])<olo):
        olo=(pred801.iloc[int((output.iloc[i][0])-1)][0])
        print(i)
        print(olo)
print(olo)
  

pred801.iloc[426][0]
print(output[445:])

for i in range(0,498):
    distance=0
    city1=int((output.iloc[i][0])-1)
    city2=int((output.iloc[i+1][0])-1)
    xx1=cities.iloc[city1][0]
    yy1=cities.iloc[city1][1]
    xx2=cities.iloc[city2][0]
    yy2=cities.iloc[city2][1]

    distance=math.sqrt(abs(((xx2-xx1)*(xx2-xx1))+((yy2-yy1)*(yy2-yy1))))
    if(distance>450):
        print(i)
        print(distance)

xx1=cities.iloc[361][0]
yy1=cities.iloc[361][1]
xx2=cities.iloc[114][0]
yy2=cities.iloc[114][1]

distance=math.sqrt(abs(((xx2-xx1)*(xx2-xx1))+((yy2-yy1)*(yy2-yy1))))
print(distance)

output.to_csv('8023', index=False)
output.head(50)