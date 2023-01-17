import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualisation library
from IPython.display import display

iris = pd.read_csv("../input/Iris.csv")
iris
Order= iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
Order["SepalLengths"]=pd.qcut(Order.SepalLengthCm,3,labels=[-1,0,1])
Order.head()
Order[['PetalLengthCm','PetalWidthCm']].corr()
#Order[['SepalLengthCm','SepalWidthCm']].corr()
df_seplen = pd.DataFrame([Order[Order.Species=="Iris-setosa"]['SepalLengths'].value_counts(),
                         Order[Order.Species=="Iris-versicolor"]['SepalLengths'].value_counts(),
                         Order[Order.Species=="Iris-virginica"]['SepalLengths'].value_counts()])
df_seplen.index = ['Setosa','Versicolor','Virginica']
display(df_seplen) 

print("Pourcentage de Setosa de classe de longueure de sépale 1:" ,round(df_seplen.iloc[0,0]/df_seplen.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueure de sépale 2:" ,round(df_seplen.iloc[0,1]/df_seplen.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueure de sépale 3:" ,round(df_seplen.iloc[0,2]/df_seplen.iloc[0,:].sum()*100),"%")
#print("Pourcentage de Setosa de classe de longueure de sépale 4:" ,round(df_seplen.iloc[0,3]/df_seplen.iloc[0,:].sum()*100),"%")
#print("Pourcentage de Setosa de classe de longueure de sépale 5:" ,round(df_seplen.iloc[0,4]/df_seplen.iloc[0,:].sum()*100),"%")
#print("Pourcentage de Setosa de classe de longueure de sépale 6:" ,round(df_seplen.iloc[0,5]/df_seplen.iloc[0,:].sum()*100),"%")

print("Pourcentage de Versicolor de classe de longueure de sépale 1:" ,round(df_seplen.iloc[1,0]/df_seplen.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueure de sépale 2:" ,round(df_seplen.iloc[1,1]/df_seplen.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueure de sépale 3:" ,round(df_seplen.iloc[1,2]/df_seplen.iloc[1,:].sum()*100),"%")
#print("Pourcentage de Versicolor de classe de longueure de sépale 4:" ,round(df_seplen.iloc[1,3]/df_seplen.iloc[1,:].sum()*100),"%")
#print("Pourcentage de Versicolor de classe de longueure de sépale 5:" ,round(df_seplen.iloc[1,4]/df_seplen.iloc[1,:].sum()*100),"%")
#print("Pourcentage de Versicolor de classe de longueure de sépale 6:" ,round(df_seplen.iloc[1,5]/df_seplen.iloc[1,:].sum()*100),"%")

print("Pourcentage de Virginia de classe de longueure de sépale 1:" ,round(df_seplen.iloc[2,0]/df_seplen.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueure de sépale 2:" ,round(df_seplen.iloc[2,1]/df_seplen.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueure de sépale 3:" ,round(df_seplen.iloc[2,2]/df_seplen.iloc[2,:].sum()*100),"%")
#print("Pourcentage de Virginia de classe de longueure de sépale 4:" ,round(df_seplen.iloc[2,3]/df_seplen.iloc[2,:].sum()*100),"%")
#print("Pourcentage de Virginia de classe de longueure de sépale 5:" ,round(df_seplen.iloc[2,4]/df_seplen.iloc[2,:].sum()*100),"%")
#print("Pourcentage de Virginia de classe de longueure de sépale 6:" ,round(df_seplen.iloc[2,5]/df_seplen.iloc[2,:].sum()*100),"%")

df_seplen.plot(kind='bar', figsize=(14,6) )
Order= Order[['SepalLengths','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
Order["SepalWidth"]=pd.qcut(Order.SepalWidthCm,3,labels=[-1,0,1])
Order.head()

df_sepwid = pd.DataFrame([Order[Order.Species=="Iris-setosa"]['SepalWidth'].value_counts(),
                         Order[Order.Species=="Iris-versicolor"]['SepalWidth'].value_counts(),
                         Order[Order.Species=="Iris-virginica"]['SepalWidth'].value_counts()])
df_sepwid.index = ['Setosa','Versicolor','Virginica']
display(df_sepwid) 

print("Pourcentage de Setosa de classe de largeur de sépale 1:" ,round(df_sepwid.iloc[0,0]/df_sepwid.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de largeur de sépale 2:" ,round(df_sepwid.iloc[0,1]/df_sepwid.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de largeur de sépale 3:" ,round(df_sepwid.iloc[0,2]/df_sepwid.iloc[0,:].sum()*100),"%")

print("Pourcentage de Versicolor de classe de largeur de sépale 1:" ,round(df_sepwid.iloc[1,0]/df_sepwid.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de largeur de sépale 2:" ,round(df_sepwid.iloc[1,1]/df_sepwid.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de largeur de sépale 3:" ,round(df_sepwid.iloc[1,2]/df_sepwid.iloc[1,:].sum()*100),"%")

print("Pourcentage de Virginia de classe de largeur de sépale 1:" ,round(df_sepwid.iloc[2,0]/df_sepwid.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de largeur de sépale 2:" ,round(df_sepwid.iloc[2,1]/df_sepwid.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de largeur de sépale 3:" ,round(df_sepwid.iloc[2,2]/df_sepwid.iloc[2,:].sum()*100),"%")

df_sepwid.plot(kind='bar', figsize=(14,6) )
Order= Order[['SepalLengths','SepalWidth','PetalLengthCm','PetalWidthCm','Species']]
Order["PetalLength"]=pd.qcut(Order.PetalLengthCm,3,labels=[-1,0,1])

df_petlen = pd.DataFrame([Order[Order.Species=="Iris-setosa"]['PetalLength'].value_counts(),
                         Order[Order.Species=="Iris-versicolor"]['PetalLength'].value_counts(),
                         Order[Order.Species=="Iris-virginica"]['PetalLength'].value_counts()])
df_petlen.index = ['Setosa','Versicolor','Virginica']
display(df_petlen) 

print("Pourcentage de Setosa de classe de longueur de pétale 1:" ,round(df_petlen.iloc[0,0]/df_petlen.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueur de pétale 2:" ,round(df_petlen.iloc[0,1]/df_petlen.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueur de pétale 3:" ,round(df_petlen.iloc[0,2]/df_petlen.iloc[0,:].sum()*100),"%")

print("Pourcentage de Versicolor de classe de longueur de pétale 1:" ,round(df_petlen.iloc[1,0]/df_petlen.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueur de pétale 2:" ,round(df_petlen.iloc[1,1]/df_petlen.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueur de pétale 3:" ,round(df_petlen.iloc[1,2]/df_petlen.iloc[1,:].sum()*100),"%")

print("Pourcentage de Virginia de classe de longueur de pétale 1:" ,round(df_petlen.iloc[2,0]/df_petlen.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueur de pétale 2:" ,round(df_petlen.iloc[2,1]/df_petlen.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueur de pétale 3:" ,round(df_petlen.iloc[2,2]/df_petlen.iloc[2,:].sum()*100),"%")

df_petlen.plot(kind='bar', figsize=(14,6) )

#ça parait quand meme bien
Order= Order[['SepalLengths','SepalWidth','PetalLength','PetalWidthCm','Species']]
Order["PetalWidth"]=pd.qcut(Order.PetalWidthCm,3,labels=[-1,0,1])
Order.head()


df_petwid = pd.DataFrame([Order[Order.Species=="Iris-setosa"]['PetalWidth'].value_counts(),
                         Order[Order.Species=="Iris-versicolor"]['PetalWidth'].value_counts(),
                         Order[Order.Species=="Iris-virginica"]['PetalWidth'].value_counts()])
df_petwid.index = ['Setosa','Versicolor','Virginica']
display(df_petwid)


print("Pourcentage de Setosa de classe de longueur de pétale 1:" ,round(df_petwid.iloc[0,0]/df_petwid.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueur de pétale 2:" ,round(df_petwid.iloc[0,1]/df_petwid.iloc[0,:].sum()*100),"%")
print("Pourcentage de Setosa de classe de longueur de pétale 3:" ,round(df_petwid.iloc[0,2]/df_petwid.iloc[0,:].sum()*100),"%")

print("Pourcentage de Versicolor de classe de longueur de pétale 1:" ,round(df_petwid.iloc[1,0]/df_petwid.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueur de pétale 2:" ,round(df_petwid.iloc[1,1]/df_petwid.iloc[1,:].sum()*100),"%")
print("Pourcentage de Versicolor de classe de longueur de pétale 3:" ,round(df_petwid.iloc[1,2]/df_petwid.iloc[1,:].sum()*100),"%")

print("Pourcentage de Virginia de classe de longueur de pétale 1:" ,round(df_petwid.iloc[2,0]/df_petwid.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueur de pétale 2:" ,round(df_petwid.iloc[2,1]/df_petwid.iloc[2,:].sum()*100),"%")
print("Pourcentage de Virginia de classe de longueur de pétale 3:" ,round(df_petwid.iloc[2,2]/df_petwid.iloc[2,:].sum()*100),"%")

df_petwid.plot(kind='bar', figsize=(14,6) )

#FAIRE .CORE() POUR TESTER SI CEST DOUBL
Order= Order[['PetalLength','SepalLengths','PetalWidth','SepalWidth','Species']]

Ord=pd.concat([Order,pd.get_dummies(Order.Species)],axis=1).drop(['Species'],axis=1)
Ord['Setosa']=Ord['Iris-setosa']
Ord['Versicolor']=Ord['Iris-versicolor']
Ord['Virginica']=Ord['Iris-virginica']
Ord=Ord[['SepalLengths','PetalWidth','Setosa','Versicolor','Virginica']]


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
Result=[[0 for x in range(0,100)] for y in range(0,3)] 
for i in range(100) :
    percent = 0.7
    in_train = np.random.binomial(1, percent, size=len(Ord)).astype('bool')
    Ord_train = Ord[in_train]
    Ord_test = Ord[~in_train]
    
    neigh.fit(Ord_train[['PetalWidth','SepalLengths']],Ord_train.Setosa)
    Result[0][i]=neigh.score(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Setosa)
    neigh.fit(Ord_train[['PetalWidth','SepalLengths']],Ord_train.Versicolor)
    Result[1][i]=neigh.score(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Versicolor)
    neigh.fit(Ord_train[['PetalWidth','SepalLengths']],Ord_train.Virginica)
    Result[2][i]=neigh.score(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Virginica)
print( "Setosa:", sum(Result[0]),"%")
print( "Versicolor:", sum(Result[1]),"%")
print( "Virginica:", sum(Result[2]),"%")
from sklearn.neural_network import MLPClassifier

Result=[[0 for x in range(0,100)] for y in range(0,3)] 
for i in range(100) :
    percent = 0.7
    in_train = np.random.binomial(1, percent, size=len(Ord)).astype('bool')
    Ord_train = Ord[in_train]
    Ord_test = Ord[~in_train]
    ann = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ann.fit(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Versicolor)
    ann.predict(Ord_train[['PetalWidth','SepalLengths']])
    Ord_test['Versiresult']=ann.predict(Ord_test[['PetalWidth','SepalLengths']])
    df_result = pd.DataFrame([Ord_test[Ord_test.Versicolor==1]['Versiresult'].value_counts()])
    df_result.index = ['right']
    df_result
    Result[1][i]=df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100
    ann.fit(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Setosa)
    Ord_test['Setoresult']=ann.predict(Ord_test[['PetalWidth','SepalLengths']])
    df_result = pd.DataFrame([Ord_test[Ord_test.Setosa==1]['Setoresult'].value_counts()])
    df_result.index = ['right']
    df_result
    Result[0][i]=df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100
    ann.fit(Ord_test[['PetalWidth','SepalLengths']],Ord_test.Virginica)
    Ord_test['Virgiresult']=ann.predict(Ord_test[['PetalWidth','SepalLengths']])
    df_result = pd.DataFrame([Ord_test[Ord_test.Virginica==1]['Virgiresult'].value_counts()])
    df_result.index = ['right']
    df_result
    Result[2][i]=df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100
print( "Setosa:", sum(Result[0])/100,"%")
print( "Versicolor:", sum(Result[1])/100,"%")
print( "Virginica:", sum(Result[2])/100,"%")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
seto_pred = gnb.fit(Ord[['PetalWidth','SepalLengths']], Ord.Setosa).predict(Ord[['PetalWidth','SepalLengths']])

seto_pred = gnb.fit(Ord[['PetalWidth','SepalLengths']], Ord.Setosa).predict(Ord[['PetalWidth','SepalLengths']])
Ord_result= Ord
Ord_result['setoRes']=seto_pred

df_result = pd.DataFrame([Ord_result[Ord_result.Setosa==1]['setoRes'].value_counts()])
df_result.index = ['right']

print("Result:" ,round(df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100),"%")

versi_pred = gnb.fit(Ord[['PetalWidth','SepalLengths']], Ord.Versicolor).predict(Ord[['PetalWidth','SepalLengths']])
Ord_result= Ord
Ord_result['versiRes']=versi_pred
Ord_result
df_result = pd.DataFrame([Ord_result[Ord_result.Versicolor==1]['versiRes'].value_counts()])
df_result.index = ['right']
df_result
print("Result:" ,round(df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100),"%")
virg_pred = gnb.fit(Ord[['PetalWidth','SepalLengths']], Ord.Virginica).predict(Ord[['PetalWidth','SepalLengths']])
Ord_result= Ord
Ord_result['virgRes']=virg_pred
Ord_result
df_result = pd.DataFrame([Ord_result[Ord_result.Virginica==1]['virgRes'].value_counts()])
df_result.index = ['right']
df_result
print("Result:" ,round(df_result.iloc[0,0]/df_result.iloc[0,:].sum()*100),"%")

