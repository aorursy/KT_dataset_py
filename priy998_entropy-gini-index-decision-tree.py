## import dependencies

from sklearn import tree #For our Decision Tree

import pandas as pd # For our DataFrame

import numpy as np

df=pd.read_csv("../input/golf-play-dataset/golf_df.csv")

df
df.Play.value_counts()
Entropy_Play = -(9/14)*np.log2(9/14) -(5/14)*np.log2(5/14)

print(Entropy_Play)

df[df.Outlook == 'sunny']
# Entropy(Play|Outlook=Sunny)

Entropy_Play_Outlook_Sunny =-(3/5)*np.log2(3/5) -(2/5)*np.log2(2/5)

print(Entropy_Play_Outlook_Sunny)
df[df.Outlook == 'overcast']
#Entropy will be 0 because homogenous entries

Entropy_Play_Outlook_Overcast =-(0/4)*np.log2(0/4) -(4/4)*np.log2(4/4)

Entropy_Play_Outlook_Overcast
df[df.Outlook == 'rainy']
# Entropy(Play|Outlook=rainy)

Entropy_Play_Outlook_Rain = -(2/5)*np.log2(2/5) - (3/5)*np.log2(3/5)

Entropy_Play_Outlook_Rain
#Gain(Play, Outlook)

Entropy_Play - (5/14)*Entropy_Play_Outlook_Sunny - (4/14)*0 - (5/14) * Entropy_Play_Outlook_Rain
#we will find Gain of Temperature,Humidity,Windy and compare them all.Choose the one with higher gain for root node.

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn import tree 

from sklearn.preprocessing import LabelEncoder



import pandas as pd 

import numpy as np 



df = pd.read_csv('../input/golf-play-dataset/golf_df.csv') 



lb = LabelEncoder() 

df['Outlook_'] = lb.fit_transform(df['Outlook']) 

df['Temperature_'] = lb.fit_transform(df['Temperature'] ) 

df['Humidity_'] = lb.fit_transform(df['Humidity'] ) 

df['Windy_'] = lb.fit_transform(df['Windy'] )   

df['Play_'] = lb.fit_transform(df['Play'] ) 

X = df.iloc[:,5:9] 

Y = df.iloc[:,9]



X_train, X_test , y_train,y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 



clf_entropy = DecisionTreeClassifier(criterion='entropy')

clf_entropy.fit(X_train.astype(int),y_train.astype(int)) 

y_pred_en = clf_entropy.predict(X_test)



print("Accuracy is :{0}".format(accuracy_score(y_test.astype(int),y_pred_en) * 100))
