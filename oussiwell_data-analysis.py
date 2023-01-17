import pandas as pd
from matplotlib import pyplot as plt
data=pd.read_csv('../input/DJ_Mag.csv')
print("That's the head of this dataset")
data.head()
print("That's the tail of this dataset")
data.tail()
print("These are the results of the last '2017' edition")
data[data.Year==2017]
DJ={}
for i in data.DJ:
    DJ[i]=0
print("The Full List of Djs:")
for i in DJ.keys():
    print(i)
Djfav=input("Please enter your favourite DJ name carefully")
djfavdf=data[data.DJ==Djfav]
plt.plot(djfavdf.Year,djfavdf.Rank, 'ro')
plt.title("Progression of %s" %Djfav)
plt.show()
etat=True
while etat==True:
    rep=input("Do you want to compare it with an another DJ(Y/N)")
    if rep=='Y':
        Dj2=input("Please enter That DJ's name carefully")
        dj2df=data[data.DJ==Dj2]
        plt.plot(djfavdf.Year,djfavdf.Rank, 'ro')
        plt.plot(dj2df.Year,dj2df.Rank, 'bo')
        plt.legend([Djfav,Dj2])
        plt.show()
    if rep=='N':
        etat=False
Armin=data[data.DJ=="Armin Van Buuren"]
Hardwell=data[data.DJ=="Hardwell"]
plt.plot(Armin.Year,Armin.Rank, 'ro' )
plt.plot(Hardwell.Year,Hardwell.Rank, 'bo')
plt.legend(["Armin Van Buuren","Hardwell"])
plt.title("La progression du Armin Van Buuren(Bleu) et Hardwell(Rouge)")
plt.show()
dates={}
for i in data.Year:
    dates[i]=0
stat=["New Entry","No change","Re-entry"]
for k in stat:
    for i in dates.keys():
        dates[i]=0
        cell=data[data.Year==i]
        for j in cell.Change:
            if j==k:
                dates[i]+=1
    plt.bar(dates.keys(),dates.values())
plt.legend(stat)
plt.show()
DJ={}
labels=[]
for i in DJ.keys():
    labesl:append(i)
for i in data.DJ:
    DJ[i]=0
for k in [1.0,2.0,3.0]:
    cell2=data[data.Rank==k]
    for i in cell2.DJ:
        DJ[i]+=1
    plt.pie(DJ.values(),labels=DJ.keys(),
        startangle=90,
        shadow= True, autopct='%1.1f%%')
    plt.title("Les pourcentages pour la place %s" %k )
    plt.show()