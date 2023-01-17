#Take a Sample of the Data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle as pickle;

import seaborn as sns;

import matplotlib.pyplot as plt;

import matplotlib as mtl;



Pokedata=pd.read_csv("../input/pokemon.csv");

battledata=pd.read_csv("../input/combats.csv");



#classify winners and losers for sample 1

TestData1=pd.DataFrame(battledata[battledata["First_pokemon"]==1]);

PokedataforTestData1=Pokedata;

def result1(x):

    for t in zip(TestData1["Winner"],TestData1["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "Winner"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "Loser"

PokedataforTestData1["Result"]=PokedataforTestData1["#"].apply(result1);

winners=pd.DataFrame(PokedataforTestData1[PokedataforTestData1["Result"]=="Winner"]);

losers=pd.DataFrame(PokedataforTestData1[PokedataforTestData1["Result"]=="Loser"]);

allforData1=pd.DataFrame();

allforData1=allforData1.append(winners);

allforData1=allforData1.append(losers);



#classify winners and losers for sample 2

TestData2=pd.DataFrame(battledata[battledata["First_pokemon"]==2]);

PokedataforTestData2=Pokedata;

def result2(x):

    for t in zip(TestData2["Winner"],TestData2["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "Winner"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "Loser"

PokedataforTestData2["Result"]=PokedataforTestData2["#"].apply(result2);

winners=pd.DataFrame(PokedataforTestData2[PokedataforTestData2["Result"]=="Winner"]);

losers=pd.DataFrame(PokedataforTestData2[PokedataforTestData2["Result"]=="Loser"]);

allforData2=pd.DataFrame();

allforData2=allforData2.append(winners);

allforData2=allforData2.append(losers);



#classify winners and losers for sample 3

TestData3=pd.DataFrame(battledata[battledata["First_pokemon"]==3]);

PokedataforTestData3=Pokedata;

def result3(x):

    for t in zip(TestData3["Winner"],TestData3["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "Winner"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "Loser"

PokedataforTestData3["Result"]=PokedataforTestData3["#"].apply(result3);

winners=pd.DataFrame(PokedataforTestData3[PokedataforTestData3["Result"]=="Winner"]);

losers=pd.DataFrame(PokedataforTestData3[PokedataforTestData3["Result"]=="Loser"]);

allforData3=pd.DataFrame();

allforData3=allforData3.append(winners);

allforData3=allforData3.append(losers);



#classify winners and losers for sample 4

TestData4=pd.DataFrame(battledata[battledata["First_pokemon"]==4]);

PokedataforTestData4=Pokedata;

def result4(x):

    for t in zip(TestData4["Winner"],TestData4["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "Winner"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "Loser"

PokedataforTestData4["Result"]=PokedataforTestData4["#"].apply(result4);

winners=pd.DataFrame(PokedataforTestData4[PokedataforTestData4["Result"]=="Winner"]);

losers=pd.DataFrame(PokedataforTestData4[PokedataforTestData4["Result"]=="Loser"]);

allforData4=pd.DataFrame();

allforData4=allforData4.append(winners);

allforData4=allforData4.append(losers);



alldatatogether=pd.DataFrame();

alldatatogether=alldatatogether.append(allforData1);

alldatatogether=alldatatogether.append(allforData2);

alldatatogether=alldatatogether.append(allforData3);

alldatatogether=alldatatogether.append(allforData4);



sns.factorplot(data=alldatatogether,x="Type 1",size=7,aspect=2,kind="count");

plt.show();

%matplotlib inline

#def result():

    #for x in zip(TestData1["Second_pokemon"],TestData1["Winner"]):

    # if x[0]==x[1]:

     # return Winner

     #elif x[0]!=x[1]:

      #return Loser

            

#print(allforData1)

a=sns.FacetGrid(data=alldatatogether,col_wrap=4,col="Type 1",hue="Result");

a=a.map(plt.scatter,"Attack","Speed")

plt.show();

a=sns.FacetGrid(data=alldatatogether,col_wrap=4,col="Type 1",hue="Result");

a=a.map(plt.scatter,"Attack","Defense")

plt.show();
a=sns.FacetGrid(data=alldatatogether,col_wrap=4,col="Type 1",hue="Result");

a=a.map(plt.scatter,"Sp. Def","Defense")

plt.show();
a=sns.FacetGrid(data=alldatatogether,col_wrap=4,col="Type 1",hue="Result");

a=a.map(plt.scatter,"Sp. Def","HP")

plt.show();
#show all stats

alldatatogetherbxplt=Pokedata;

alldatatogetherbxplt=alldatatogetherbxplt.drop(["#","Generation"],axis=1)

sns.factorplot(data=alldatatogetherbxplt,size=7,kind="box",aspect=2);

plt.show();
##riunning the data through a decision ntree

from sklearn.tree import DecisionTreeClassifier

#Take a Sample of the Data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle as pickle;

import seaborn as sns;

import matplotlib.pyplot as plt;

import matplotlib as mtl;



Pokedata=pd.read_csv("../input/pokemon.csv");

battledata=pd.read_csv("../input/combats.csv");



#classify winners and losers for sample 1

TestData1=pd.DataFrame(battledata[battledata["First_pokemon"]==1]);

PokedataforTestData1=Pokedata;

def result1(x):

    for t in zip(TestData1["Winner"],TestData1["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "1"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "0"

PokedataforTestData1["Result"]=PokedataforTestData1["#"].apply(result1);

winners=pd.DataFrame(PokedataforTestData1[PokedataforTestData1["Result"]=="0"]);

losers=pd.DataFrame(PokedataforTestData1[PokedataforTestData1["Result"]=="1"]);

allforData1=pd.DataFrame();

allforData1=allforData1.append(winners);

allforData1=allforData1.append(losers);



#classify winners and losers for sample 2

TestData2=pd.DataFrame(battledata[battledata["First_pokemon"]==2]);

PokedataforTestData2=Pokedata;

def result2(x):

    for t in zip(TestData2["Winner"],TestData2["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "1"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "0"

PokedataforTestData2["Result"]=PokedataforTestData2["#"].apply(result2);

winners=pd.DataFrame(PokedataforTestData2[PokedataforTestData2["Result"]=="1"]);

losers=pd.DataFrame(PokedataforTestData2[PokedataforTestData2["Result"]=="0"]);

allforData2=pd.DataFrame();

allforData2=allforData2.append(winners);

allforData2=allforData2.append(losers);



#classify winners and losers for sample 3

TestData3=pd.DataFrame(battledata[battledata["First_pokemon"]==3]);

PokedataforTestData3=Pokedata;

def result3(x):

    for t in zip(TestData3["Winner"],TestData3["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "1"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "0"

PokedataforTestData3["Result"]=PokedataforTestData3["#"].apply(result3);

winners=pd.DataFrame(PokedataforTestData3[PokedataforTestData3["Result"]=="1"]);

losers=pd.DataFrame(PokedataforTestData3[PokedataforTestData3["Result"]=="0"]);

allforData3=pd.DataFrame();

allforData3=allforData3.append(winners);

allforData3=allforData3.append(losers);



#classify winners and losers for sample 4

TestData4=pd.DataFrame(battledata[battledata["First_pokemon"]==4]);

PokedataforTestData4=Pokedata;

def result4(x):

    for t in zip(TestData4["Winner"],TestData4["Second_pokemon"]):

     if t[0]==t[1]:

      if x==t[0]:

       return "1"

     elif t[1]!=t[0]:

      #print(x,"x",t[1],"t[0]")

      if x==t[1]:

       return "0"

PokedataforTestData4["Result"]=PokedataforTestData4["#"].apply(result4);

winners=pd.DataFrame(PokedataforTestData4[PokedataforTestData4["Result"]=="1"]);

losers=pd.DataFrame(PokedataforTestData4[PokedataforTestData4["Result"]=="0"]);

allforData4=pd.DataFrame();

allforData4=allforData4.append(winners);

allforData4=allforData4.append(losers);



alldatatogether=pd.DataFrame();

alldatatogether=alldatatogether.append(allforData1);

alldatatogether=alldatatogether.append(allforData2);

alldatatogether=alldatatogether.append(allforData3);

alldatatogether=alldatatogether.append(allforData4);

#print(alldatatogether)



fordescisiontree=alldatatogether.drop(["Generation","Legendary","#","Type 2"],axis=1)

fordescisiontree=fordescisiontree.reset_index(level=0)

fordescisiontree=fordescisiontree.drop(["index","Name"],axis=1)

#print(fordescisiontree)

type_={"Bug":0,"Dark":1,"Dragon":2,"Electric":3,"Fairy":4,"Fighting":5,"Fire":6,"Flying":7,"Ghost":8,"Grass":9,"Ground":10,"Ice":11,"Normal":12,"Poison":13,"Psychic":14,"Rock":15,"Steel":16,"Water":17};

def changetype(x):

 for a in type_:

  if x==a:

    return type_[a]

fordescisiontree["Type 1"]=fordescisiontree["Type 1"].apply(changetype)

labels=fordescisiontree["Result"].tolist()

labels=np.array(labels)

#print(labels)

fordescisiontree=fordescisiontree.drop(["Result"],axis=1)

#print(fordescisiontree)

clif=DecisionTreeClassifier()

clif.fit(fordescisiontree,labels)

importances=clif.feature_importances_

columns_=list(fordescisiontree.columns.values)

#print(importances,columns_)

sns.pointplot(x=columns_,y=importances,size=10);

plt.show();


