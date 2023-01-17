from IPython.display import display

from PIL import Image

path="../input/breast-cancerjpg/breast cancer.jpg"

display(Image.open(path))
#Basic library 

import pandas  as pd 

import numpy as np 

import math 

import re 



#visualization 

import seaborn as sns 

import matplotlib.pyplot as plt 

import cufflinks as cf

%matplotlib inline

cf.go_offline()

sns.set_style('whitegrid')

#importing Data

Data = '../input/breast-cancer/Breast_cancer.csv'

df = pd.read_csv(Data)

df.drop('Unnamed: 0', axis = 1 , inplace = True)

df.tail(5)
df.info()
df.describe()
df['Class'].value_counts()   # Data to predict (1=Posstive Cancer , 0 = Negatice Cancer)
df.corr()
sns.heatmap(df.isnull() ,  yticklabels='False' , cmap = 'plasma')

print(df.isnull().any())
# the columns ' ID'  Does not give any important Data or correlation 

data = df.drop('Id' , axis = 1)



#Find Nan Value Index 

Nan_index = data[data.isna().any(axis=1)].index 

data[data.isna().any(axis=1)][0:5]
# Finding cell size Values

size= data[data.isna().any(axis=1)].iloc[:,1].to_list()



# Finding cell Shape Values

shape = data[data.isna().any(axis=1)].iloc[:,2].to_list()

print("size index =",size)

print("shape index =",shape)
#crate a empty list to collect the mode of Bare.nucle

Mode_Bare_nuclei = []  #crate a empty list to collect the mode of Bare.nuclei

for i, j in zip(size,shape):  # collecting Data 

    Mod= data.loc[(data['Cell.size'] == i) & (data['Cell.shape'] == j)]['Bare.nuclei'].mode()[0]

    Mode_Bare_nuclei.append(Mod)
#replacing Data using index 

for i, j in zip(Nan_index,Mode_Bare_nuclei):

    data.loc[i,'Bare.nuclei'] = j



data.iloc[Nan_index][0:3] # Checking We have no more Nan_values
fig, ax1 = plt.subplots (figsize= (20,15))

ax = sns.heatmap(df.corr(),ax=ax1 , linecolor = 'white',linewidths=0.5, 

            annot = True,cmap=sns.diverging_palette(20, 220, n=200))

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
df_re = df[['Cell.size','Cell.shape','Bare.nuclei','Bl.cromatin']]

df_re.iplot(kind = 'box',)

print("I consider these 4 variable important , therefore I was looking for any clue or trend ")
f, ax = plt.subplots(figsize=(20, 10))

sns.violinplot(data=df.drop(["Id","Class"],axis = 1), width = 0.9, inner = 'quartile')

print("Here we have a Visual description of mean and the Data Disstribution")
fig ,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (20,5))

sns.countplot(x ='Cl.thickness', data = df , hue='Class',  ax = ax1 )

sns.countplot(x ='Cell.size', data = df, hue='Class' , ax = ax2)

sns.countplot(x ='Cell.shape', data = df, hue='Class' , ax = ax3)

print('Relation Thickness, size, shape')
fig ,(ax4,ax5,ax6) = plt.subplots(1,3,figsize = (20,5))

sns.countplot(x ='Marg.adhesion', data = df, hue='Class' , ax = ax4 )

sns.countplot(x ='Epith.c.size', data = df, hue='Class' , ax = ax5 )

sns.countplot(x ='Bare.nuclei', data = df, hue='Class' , ax = ax6 )

print('Relation Marg.Adhension, Epith C.Size , Bare.nuclei')
fig ,(ax7,ax8,ax9) = plt.subplots(1,3,figsize = (20,5))

sns.countplot(x ='Bl.cromatin', data = df, hue='Class' , ax = ax7 )

sns.countplot(x ='Normal.nucleoli', data = df, hue='Class' , ax = ax8 )

sns.countplot(x ='Mitoses', data = df, hue='Class' , ax = ax9 )

print('Relation Bl.cromatin', 'Normal.nucleoli' , 'Mitoses')
### applying KNN algorithm

from sklearn.neighbors import KNeighborsClassifier  # Algorithm

from sklearn.metrics import confusion_matrix,f1_score,classification_report # metrics

from sklearn.preprocessing import StandardScaler #scaler to 0-1

from sklearn.model_selection import train_test_split #train and split 
#StandarScale with new Dataframe

scaler = StandardScaler()

scaler_data = scaler.fit_transform(data.drop('Class',axis = 1)) 

data_p = pd.DataFrame(scaler_data , columns= data.columns[:-1])

data_p.head(4) # data_p = Data already Processed
#Splitting Data

X = data_p

y = data ['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Finding the n_neighbors that fits the best 

Error_Rate = []

for i in range (1,50):

    

    KNN_Error = KNeighborsClassifier(n_neighbors=i)

    KNN_Error.fit(X_train,y_train)

    pred_i = KNN_Error.predict(X_test)

    Error_Rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,50), Error_Rate , color = 'blue', linestyle = 'dashed', marker = 'o')

plt.title('Error rate vs K value')

print( " K=3 is the most accurate rate because the error is closest to 0 ")
#KNN Algorithm

KNN = KNeighborsClassifier(n_neighbors=3)

KNN.fit(X_train,y_train)

#Predicting 

y_pred_KNN = KNN.predict(X_test)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression().fit(X_train,y_train)

y_predi_LR = LR.predict(X_test)



print(confusion_matrix(y_test,y_predi_LR))

print(f1_score(y_test,y_predi_LR))

print(classification_report(y_test,y_predi_LR))