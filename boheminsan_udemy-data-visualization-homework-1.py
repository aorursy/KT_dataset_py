# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from decimal import Decimal

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#reading csv files
data = pd.read_csv('../input/mushrooms.csv')
data.head()
data.info()
data.columns = [c.replace(' ', '') for c in data.columns]
data.columns = [c.replace('-', '') for c in data.columns]
data.columns
descr=data.describe()
descr
a=data['class']
for i in range(0,len(a)):
    if a[i]=="p":
        a[i]="poisonous"
    elif a[i]=="e":
        a[i]="edible"

a=data['capshape']
for i in range(0,len(a)):
    if a[i]=="b":
        a[i]="bell"
    elif a[i]=="c":
        a[i]="conical"
    elif a[i]=="x":
        a[i]="convex"
    elif a[i]=="f":
        a[i]="flat"
    elif a[i]=="k":
        a[i]="knobbed"
    elif a[i]=="s":
        a[i]="sunken"

a=data['capsurface']
for i in range(0,len(a)):
    if a[i]=="f":
        a[i]="fibrous"
    elif a[i]=="g":
        a[i]="grooves"
    elif a[i]=="y":
        a[i]="scaly"
    elif a[i]=="s":
        a[i]="smooth"

a=data['capcolor']
for i in range(0,len(a)):
    if a[i]=="n":
        a[i]="brown"
    elif a[i]=="b":
        a[i]="buff"
    elif a[i]=="c":
        a[i]="cinnamon"
    elif a[i]=="g":
        a[i]="gray"
    elif a[i]=="r":
        a[i]="green"
    elif a[i]=="p":
        a[i]="pink"
    elif a[i]=="u":
        a[i]="purple"
    elif a[i]=="e":
        a[i]="red"
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"

a=data['bruises']
for i in range(0,len(a)):
    if a[i]=="t":
        a[i]="yes"
    elif a[i]=="f":
        a[i]="no"

a=data['odor']
for i in range(0,len(a)):
    if a[i]=="a":
        a[i]="almond"
    elif a[i]=="l":
        a[i]="anise"
    elif a[i]=="c":
        a[i]="creosote"
    elif a[i]=="y":
        a[i]="fishy"
    elif a[i]=="f":
        a[i]="foul"
    elif a[i]=="m":
        a[i]="musty"
    elif a[i]=="n":
        a[i]="none"
    elif a[i]=="p":
        a[i]="pungent"
    elif a[i]=="s":
        a[i]="spicy"

a=data['gillattachment']
for i in range(0,len(a)):
    if a[i]=="a":
        a[i]="attachment"
    elif a[i]=="d":
        a[i]="descending"
    elif a[i]=="f":
        a[i]="free"
    elif a[i]=="n":
        a[i]="notched"

a=data['gillspacing']
for i in range(0,len(a)):
    if a[i]=="c":
        a[i]="close"
    elif a[i]=="w":
        a[i]="crowded"
    elif a[i]=="d":
        a[i]="distant"

a=data['gillsize']
for i in range(0,len(a)):
    if a[i]=="b":
        a[i]="broad"
    elif a[i]=="n":
        a[i]="narrow"

a=data['gillcolor']
for i in range(0,len(a)):
    if a[i]=="k":
        a[i]="black"
    elif a[i]=="n":
        a[i]="brown"
    elif a[i]=="b":
        a[i]="buff"
    elif a[i]=="h":
        a[i]="chocolate"
    elif a[i]=="g":
        a[i]="gray"
    elif a[i]=="r":
        a[i]="green"
    elif a[i]=="o":
        a[i]="orange"    
    elif a[i]=="p":
        a[i]="pink"
    elif a[i]=="u":
        a[i]="purple"
    elif a[i]=="e":
        a[i]="red"
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"
    
a=data['gillsize']
for i in range(0,len(a)):
    if a[i]=="b":
        a[i]="broad"
    elif a[i]=="n":
        a[i]="narrow"

a=data['stalkshape']
for i in range(0,len(a)):
    if a[i]=="e":
        a[i]="enlarging"
    elif a[i]=="t":
        a[i]="taping"

a=data['stalkroot']
for i in range(0,len(a)):
    if a[i]=="b":
        a[i]="bulbous"
    elif a[i]=="c":
        a[i]="club"
    elif a[i]=="u":
        a[i]="cup"
    elif a[i]=="e":
        a[i]="equal"
    elif a[i]=="z":
        a[i]="rhizomorphs"
    elif a[i]=="r":
        a[i]="rooted"
    elif a[i]=="?":
        a[i]="missing"    
    
a=data['stalksurfaceabovering']
for i in range(0,len(a)):
    if a[i]=="f":
        a[i]="fibrous"
    elif a[i]=="y":
        a[i]="scaly"
    elif a[i]=="k":
        a[i]="silky"
    elif a[i]=="s":
        a[i]="smooth"
    
a=data['stalksurfacebelowring']
for i in range(0,len(a)):
    if a[i]=="f":
        a[i]="fibrous"
    elif a[i]=="y":
        a[i]="scaly"
    elif a[i]=="k":
        a[i]="silky"
    elif a[i]=="s":
        a[i]="smooth"

a=data['stalkcolorabovering']
for i in range(0,len(a)):
    if a[i]=="n":
        a[i]="brown"
    elif a[i]=="b":
        a[i]="buff"
    elif a[i]=="c":
        a[i]="cinnamon"
    elif a[i]=="g":
        a[i]="gray"
    elif a[i]=="o":
        a[i]="orange"    
    elif a[i]=="p":
        a[i]="pink"
    elif a[i]=="e":
        a[i]="red"
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"

a=data['stalkcolorbelowring']
for i in range(0,len(a)):
    if a[i]=="n":
        a[i]="brown"
    elif a[i]=="b":
        a[i]="buff"
    elif a[i]=="c":
        a[i]="cinnamon"
    elif a[i]=="g":
        a[i]="gray"
    elif a[i]=="o":
        a[i]="orange"    
    elif a[i]=="p":
        a[i]="pink"
    elif a[i]=="e":
        a[i]="red"
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"

a=data['veiltype']
for i in range(0,len(a)):
    if a[i]=="p":
        a[i]="partial"
    elif a[i]=="u":
        a[i]="genel"

a=data['veilcolor']
for i in range(0,len(a)):
    if a[i]=="n":
        a[i]="brown"
    elif a[i]=="o":
        a[i]="orange"    
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"

a=data['ringnumber']
for i in range(0,len(a)):
    if a[i]=="n":
        a[i]="none"
    elif a[i]=="o":
        a[i]="one"
    elif a[i]=="t":
        a[i]="two"

a=data['ringtype']
for i in range(0,len(a)):
    if a[i]=="c":
        a[i]="cobwebby"
    elif a[i]=="e":
        a[i]="evanescent"
    elif a[i]=="f":
        a[i]="flaring"
    elif a[i]=="l":
        a[i]="large"
    elif a[i]=="n":
        a[i]="none"    
    elif a[i]=="p":
        a[i]="pendant"
    elif a[i]=="s":
        a[i]="sheathing"
    elif a[i]=="z":
        a[i]="zone"

a=data['sporeprintcolor']
for i in range(0,len(a)):
    if a[i]=="k":
        a[i]="black"
    elif a[i]=="n":
        a[i]="brown"
    elif a[i]=="b":
        a[i]="buff"
    elif a[i]=="h":
        a[i]="chocolate"
    elif a[i]=="r":
        a[i]="green"
    elif a[i]=="o":
        a[i]="orange"    
    elif a[i]=="u":
        a[i]="purple"
    elif a[i]=="w":
        a[i]="white"
    elif a[i]=="y":
        a[i]="yellow"

a=data['population']
for i in range(0,len(a)):
    if a[i]=="a":
        a[i]="abundant"
    elif a[i]=="c":
        a[i]="clustered"
    elif a[i]=="n":
        a[i]="numerous"
    elif a[i]=="s":
        a[i]="scattered"
    elif a[i]=="v":
        a[i]="several"    
    elif a[i]=="y":
        a[i]="solitary"

a=data['habitat']
for i in range(0,len(a)):
    if a[i]=="g":
        a[i]="grasses"
    elif a[i]=="l":
        a[i]="leaves"
    elif a[i]=="m":
        a[i]="meadows"
    elif a[i]=="p":
        a[i]="paths"
    elif a[i]=="u":
        a[i]="urban"    
    elif a[i]=="w":
        a[i]="waste"
    elif a[i]=="d":
        a[i]="woods"
        
#ahh... finally we have done it... 
data.head(10)
#seperating classes
edata=data.loc[data['class']=='edible']
pdata=data.loc[data['class']=='poisonous']
edata.head()
#we will use them later
#edata['capsurface'].value_counts(normalize=True)*100
columns=data.columns
edatas={}
pdatas={}
for i in columns:
    a=edata[i].value_counts().T
    b=pdata[i].value_counts().T
    edatas[i]=a
    pdatas[i]=b
types=data['capsurface'].unique()
edible=edatas["capsurface"]
edible['grooves']=0.0
edible=edible.astype(int)
poisonous=pdatas["capsurface"]
poisonous=poisonous.astype(int)
df=pd.DataFrame({'Edible':edible,'Poisonous':poisonous},index=types)
df.fillna(0)
edible2=pd.Series.to_frame(edible)
edible2['class']='edible'
edible2=edible2.T

poisonous2=pd.Series.to_frame(poisonous)
poisonous2['class']='poisonous'
poisonous2=poisonous2.T

df2=pd.concat([edible2, poisonous2], axis=1 , sort=True)
df2=df2.T
df2=df2.reset_index(inplace=False)
df2_ren2={'index':'capsurface','capsurface':'count'}
df2=df2.rename(columns=df2_ren2)
df2
with sns.axes_style("whitegrid", {'axes.grid' : True}):
    fig, ax = plt.subplots(1,1, figsize=(14,6))
g=sns.barplot(x = 'capsurface', y = 'count', hue = 'class', data=df2,palette="winter")
ax.legend()
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.show()
def listele(dframe,list1,list2):
    columns=dframe.columns
    for i in columns:
        a=dframe[i]
        nc = Counter(a)         
        ecr = nc.most_common(1)
        count=len(a)
        attrname=ecr[0][0]
        rate=(ecr[0][1])/count*100
        list1.append(attrname)
        list2.append(round(rate,2))
prates=[]
pnames=[]
erates=[]
enames=[]
#eratesall=pd.Series.to_frame(edata['capsurface'].value_counts(normalize=True)*100)
#pratesall=pdata['capsurface'].value_counts(normalize=True)*100
#type(eratesall)
listele(pdata,pnames, prates)
listele(edata,enames, erates)
columns=data.columns
names=enames+pnames
f,ax = plt.subplots(figsize = (12,16))
sns.barplot(x=erates,y=columns,color='lime',alpha = 0.8,label='Edible')
sns.barplot(x=prates,y=columns,color='green',alpha = 0.4,label='Poisonous')
#for p in ax.patches:
    #ax.annotate('{}'.format(p.get_width()), (p.get_y()+0.1, p.get_height()+1))
sayac=0
itere=0
for p in ax.patches:
    width=p.get_width()
    #print(p.get_width())
    if sayac%2==0:
        a=5
        clr = 'black'
    else:
        a=25
        clr = 'blue'
    k=names[itere].capitalize()
    plt.text(a, p.get_y()+0.55*p.get_height(),'{}:{:1.2f}'.format(k,width),color='black', va='center')
    itere=itere+1
    sayac=sayac+1
ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Rates', ylabel='Columns',title = "Most frequently encountered attributes")
plt.show()
def convert_df(ep):
    result= pd.DataFrame()
    for i in data.columns:
        dafr=pd.Series.to_frame(ep[i].value_counts(normalize=True)*100)
        dafr.reset_index(inplace=True)
        dafr.rename(index=str,columns={"index": i + " name",i:i+" value"},inplace=True)
        result=pd.concat([result,dafr],axis=1, sort=False)
    result.reset_index(inplace=True)
    result.rename(index=str,columns={"index": "name"},inplace=True)
    result.index = np.arange(1, len(result)+1)
    result=result.drop(['name','class name','class value'], axis=1)
    return result
datae=convert_df(edata).head(1)
datap=convert_df(pdata).head(1)
datae
datap
datae=convert_df(edata).head(1)
datae=datae.drop(['capshape name','capshape value','capcolor name','capcolor value', 'gillattachment name', 'gillattachment value','gillspacing name','gillspacing value','stalkshape name','stalkshape value','stalkroot name','stalkroot value','stalkcolorabovering name','stalkcolorabovering value','stalkcolorabovering name','stalkcolorabovering value','stalkcolorbelowring name','stalkcolorbelowring value','veiltype name','veiltype value','veilcolor name','veilcolor value','ringnumber name','ringnumber value','habitat name','habitat value'], axis=1)
datap=convert_df(pdata).head(1)
datap=datap.drop(['capshape name','capshape value','capcolor name','capcolor value', 'gillattachment name', 'gillattachment value','gillspacing name','gillspacing value','stalkshape name','stalkshape value','stalkroot name','stalkroot value','stalkcolorabovering name','stalkcolorabovering value','stalkcolorabovering name','stalkcolorabovering value','stalkcolorbelowring name','stalkcolorbelowring value','veiltype name','veiltype value','veilcolor name','veilcolor value','ringnumber name','ringnumber value','habitat name','habitat value'], axis=1)

liste=[]
eliste=[]
pliste=[]
for i in datae.columns:
    if i.endswith('name'):
        gec=i.replace(" name","")
        liste.append(gec)
        eliste.append(np.ndarray.tolist(datae[i].values)[0])
        pliste.append(np.ndarray.tolist(datap[i].values)[0])
        datae=datae.drop(i,axis=1)
        datap=datap.drop(i,axis=1)
epliste=list(np.ndarray.tolist(datae.values)[0])+list(np.ndarray.tolist(datap.values)[0])
datae2=np.ndarray.tolist(datae.values[0])
datap2=np.ndarray.tolist(datap.values[0])
ep2liste=eliste+pliste
dataep=list(datae2)+list(datap2)
f,ax = plt.subplots(figsize =(16,8))
sns.pointplot(x=liste, y=datae2, color='cyan', alpha=0.8)
sns.pointplot(x=liste, y=datap2, color='orange', alpha=0.8)
sayac=0
ofs=0
for c in ax.collections:
    for of in c.get_offsets():
        if of[0]==0:
            ofs=ofs+1
        if ofs==2 and of[0]>=0:
            renk='blue'
        else:
            renk='black'
        yazi=str(ep2liste[sayac])+": \n %"+str(round(dataep[sayac],2))
        ax.annotate(yazi,of,of,color=renk,fontsize = 13)
        sayac=sayac+1
plt.xlabel('Attribute Names',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Comparison',fontsize = 20,color='blue')
plt.grid()
plt.show()
labels = data.gillcolor.value_counts().index
colors = ['orange','lime','blue','yellow','brown','green','gray','cyan','red','magenta','purple','pink']
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes=data.gillcolor.value_counts().values

plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',textprops={'fontsize': 12})
plt.title('Mushrooms Chart by Gill Color',color = 'blue',fontsize = 15)
dataf=convert_df(data)
dataf=dataf.drop("veiltype name",axis=1)
dataf=dataf.drop("veiltype value", axis=1)
dataf

edataf=convert_df(edata)
edataf=edataf.drop("veiltype name",axis=1)
edataf=edataf.drop("veiltype value", axis=1)
edataf

pdataf=convert_df(pdata)
pdataf=pdataf.drop("veiltype name",axis=1)
pdataf=pdataf.drop("veiltype value", axis=1)
pdataf
g = (sns.jointplot("sporeprintcolor value", "habitat value", height=6, data=edataf, ratio=6, kind="kde",space=0, color="g")).set_axis_labels("Spore Color", "Population")
plt.show()
sns.lmplot(x="gillcolor value", y="capcolor value", data=dataf)
plt.show()
datach=pd.concat([dataf['gillcolor value'], dataf['capcolor value'],dataf['habitat value'],dataf['odor value']], axis=1).head(10)
datach
sns.kdeplot(datach['gillcolor value'], datach['capcolor value'], shade=True, cut=3)
plt.show()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.violinplot(data=datach, palette=flatui, inner="points")
plt.show()
#dataf.corr()
if 'bruises name' in dataf.columns:
    dataf=dataf.drop(['bruises name', 'bruises value','gillattachment name', 'gillattachment value','gillsize name', 'gillsize value', 'gillspacing name', 'gillspacing value','stalkshape name', 'stalkshape value'], axis=1)
dataf.fillna(0)
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(dataf.corr(), annot=True, linewidths=1,linecolor="cyan", fmt= '.2f',ax=ax)
plt.show()
data.head()
a=data['capcolor']
for i in range(0,len(a)):
    if a[i]=="brown":
        a[i]="21"
    elif a[i]=="buff":
        a[i]="22"
    elif a[i]=="cinnamon":
        a[i]="23"
    elif a[i]=="gray":
        a[i]="24"
    elif a[i]=="green":
        a[i]="25"
    elif a[i]=="pink":
        a[i]="26"
    elif a[i]=="purple":
        a[i]="27"
    elif a[i]=="red":
        a[i]="28"
    elif a[i]=="white":
        a[i]="29"
    elif a[i]=="yellow":
        a[i]="30"
a=a.astype(int)
dfbox=pd.concat([data['bruises'],a,data['stalkshape'],data['ringnumber']],axis=1)
dfbox.head()
sns.boxplot(x="ringnumber", y="capcolor", hue="bruises", data=dfbox, palette="BuGn")
plt.show()
dfbox2=pd.concat([dfbox.tail(800),dfbox.head(800)],axis=0)
sns.swarmplot(x="ringnumber", y="capcolor", hue="bruises", data=dfbox2)
plt.show()
datapair=pd.concat([dataf['population value'], dataf['habitat value']], axis=1).head(7).fillna(0)
datapair
sns.pairplot(datapair)
plt.show()
sns.countplot(data.stalkcolorabovering,palette="Paired")
#sns.countplot(kill.manner_of_death)
plt.title("stalkcolorabovering",color = 'blue',fontsize=15)