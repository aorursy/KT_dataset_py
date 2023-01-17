import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
cwd
NBA=pd.read_csv('../input/NBA_player_of_the_week.csv')
NBA.head()
NBA.info()
# combine positions into G (Guard), F(Forward),C(Center) and creat a column named" new_position
NBA['Position'].value_counts()
new_position=NBA['Position'].replace({'SG': 'G'})
new_position=new_position.replace({'PG': 'G'})
new_position=new_position.replace({'SG': 'G'})
new_position=new_position.replace({'FC': 'C'})
new_position=new_position.replace({'F-C': 'C'})
new_position=new_position.replace({'PF': 'F'})
new_position=new_position.replace({'SF': 'F'})
new_position=new_position.replace({'GF': 'F'})
new_position=new_position.replace({'G-F': 'F'})
new_position.value_counts()
NBA['Positions']=new_position
NBA.head()
###convert Height to number
def convert_Height(item):
    a=int(item.split('-')[0])
    b=int(item.split('-')[1])
    h=a*30.48+b*2.54
    return h
### convert height with cm (eg '200cm') into a int
def remove_cm(item):
    h="".join(list(item)[:-2])
    return int(h)
###convert kg to lb
def convert_to_lb(item):
    h="".join(list(item)[:-2])
    a=int(h)/0.45359237
    return a
height_cm=NBA['Height'].apply(lambda x: convert_Height(x) if '-' in x else remove_cm(x))
weight_lb=NBA['Weight'].apply(lambda x: convert_to_lb(x) if 'kg'in x else int(x))
NBA['height_cm']=height_cm
NBA['weight_lb']=weight_lb
NBA.head()
#distribution of height_cm with Positions
plt.figure(figsize=(10,6))
plt.hist (NBA[NBA['Positions']=='G']['height_cm'],color='red', label='G',bins=10)
plt.hist (NBA[NBA['Positions']=='F']['height_cm'],color='blue',label='F',bins=10)
plt.hist (NBA[NBA['Positions']=='C']['height_cm'],color='yellow',label='C',bins=10)
plt.legend()


plt.figure(figsize=(10,6))
plt.hist (NBA[NBA['Positions']=='G']['weight_lb'],color='red', label='G',bins=10)
plt.hist (NBA[NBA['Positions']=='F']['weight_lb'],color='blue',label='F',bins=10)
plt.hist (NBA[NBA['Positions']=='C']['weight_lb'],color='yellow',label='C',bins=10)
plt.legend()

sns.lmplot(x='height_cm',y='weight_lb',data=NBA)
####machine learning, use random forest to predict positions according to weight and height
X=NBA[['height_cm','weight_lb']]
y=NBA['Positions']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=20)
rfc.fit(X_train, y_train)
pre=rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pre, y_test))
print('\n')
print(confusion_matrix(pre,y_test))
