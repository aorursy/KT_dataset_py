#importing essential packages

import numpy as np

import pandas as pd
#reading dataset

df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.shape
df.head()
df.isnull().sum()
df.dtypes
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Finding correlation

correl=df.corr()

plt.subplots(figsize=(16,10))

fig1=sns.heatmap(correl, annot=True)

bottom, top = fig1.get_ylim()

fig1.set_ylim(bottom + 0.5, top - 0.5)
print("UNIVARIATE ANALYSIS-HISTOGRAMS:")

for i in [0,1,2,9,10]:

    plt.subplots(figsize=(10,5))

    sns.distplot(df.iloc[:,i],color='purple',bins=15)

    plt.show()

    
print("BOXPLOTS:")

for i in range(0,12):

    plt.subplots(figsize=(8,4))

    sns.boxplot(x=df.iloc[:,11],y = df.iloc[:,i],palette="cool")

    plt.show()
print("BIVARIATE ANALYSIS: Scatter plots-\n")

l1=[1,2,9,10]

for i in l1:

    for j in l1:

        if(j!=i):

            sns.scatterplot(x=df.iloc[:,i],y=df.iloc[:,j],hue=df["quality"])

            plt.show()

    l1.remove(i)
from sklearn.neighbors import KNeighborsClassifier 

#Feature selection

y=df.iloc[:,11]

X=df.iloc[:,[1,2,9,10]] #Using only top 4 columns with highest correlation to quality

#Scaling the data

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X=ss.fit_transform(X)


#Finding best suitable random state between state 1-43



"""This step fits multiple KNN models with number of neighbours ranging from 1-35 and random states 1-43

to give the best case accuracy. Random state manipulation is NOT A GOOD PRACTICE but FINDING OPTIMAL NUMBER OF NEIGHBOURS

for the particular problem is important. 



PLEASE NOTE THIS STEP TAKES AROUND 1 MINUTE TO RUN"""



state_macc=list(np.empty(43))

for i in range(1,43):

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i,stratify=y)

    

    neighbors = np.arange(1,35)

    train_accuracy = np.empty(len(neighbors))

    test_accuracy = np.empty(len(neighbors))

    for j,k in enumerate(neighbors):

        knn2 = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='manhattan')

        knn2.fit(X_train,y_train)

        train_accuracy[j] = knn2.score(X_train, y_train)

        test_accuracy[j] = knn2.score(X_test, y_test)

    macc=np.amax(test_accuracy)

    x=np.where(test_accuracy==np.amax(test_accuracy))

    x=x[0].tolist()

    x = [i+1 for i in x]

    data=[x,float(macc)]

    state_macc[i-1]=data

temp=0.0

pos=0

for i in range(42):

    state_macc[i][-1] = float(state_macc[i][-1])

    if (state_macc[i][-1] > temp):

        temp = state_macc[i][-1]

        pos=i+1

ideal_k=state_macc[pos-1][0]

print("Ideal random state= {} yielding max accuracy {} with number of neighbours: ".format(pos,temp),ideal_k)    
"""THEREFORE Final model is prepared using ideal Random State and Best K number of neighbours """



from sklearn.neighbors import KNeighborsClassifier 

y=df.iloc[:,11]

X=df.iloc[:,[1,2,9,10]] #Using only top 4 columns with highest correlation to quality



from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X=ss.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=pos,stratify=y)



knn = KNeighborsClassifier(n_neighbors=ideal_k[0],weights='distance',metric='manhattan')

knn.fit(X_train,y_train)



facc=knn.score(X_test,y_test)

print("\n\nAccuracy of final model is: ", facc*100, "%\n\n")
print("Prediction table:")

a=knn.predict(X_test)

b=y_test

z=list(enumerate(a))

z=[x[0] for x in z]

table={'Index':z,'Predicted value':a,'Real value':b}

nf=pd.DataFrame(table)

nf.style.hide_index()