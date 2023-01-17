import numpy as np

import pandas as pd

sgdp=pd.read_csv("../input/gdpdatasets/Part I-A.csv")

sgdp
sgdp=sgdp.drop(index=[5,10])

sgdp
sgdp=sgdp.drop(columns=["West Bengal1"])

sgdp.isnull().sum()
sgdp=sgdp.transpose()

sgdp
sgdp.isnull().sum()
sgdp[4].fillna(1.36753e+07/32, inplace=True)
sgdp[9].fillna(9.99, inplace=True)
sgdp
sgdp.columns = ['GSDPCP 11-12','GSDPCP 12-13','GSDPCP 13-14','GSDPCP 14-15','GSDPCP 15-16',

                'Growth % 12-13','Growth % 13-14','Growth % 14-15','Growth % 15-16']

sgdp
sgdp=sgdp.drop(["Duration","Items  Description"],axis=0)
sgdp
sgdp['average growth %']=(sgdp['Growth % 13-14']+sgdp['Growth % 14-15']+sgdp['Growth % 15-16'])/3

sgdp
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.figure(figsize=(20,10))

plt.plot(sgdp['average growth %'],sgdp.index,'ro')

plt.xlabel("Average Growth %")

plt.ylabel("States")

plt.xlim([0,15])

plt.grid(True)
sgdp.sort_values(by=['average growth %'],ascending=False)
sgdp1=sgdp.drop("All_India GDP",axis=0)

plt.figure(figsize=(20,10))

plt.plot(sgdp1['GSDPCP 15-16'],sgdp1.index,'g^')

plt.xlabel("total GDP of the states for the year 2015-16")

plt.ylabel("States")

plt.grid(True)
sgdp1.sort_values(by=['GSDPCP 15-16'],ascending=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

ap=pd.read_csv(r"../input/gdpdatasets/NAD-Andhra_Pradesh-GSVA_cur_2016-17.csv")

ap
ap=ap.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

ap
ap.columns=['Items','Andhra Pradesh']

ap
arp=pd.read_csv(r"../input/gdpdatasets/NAD-Arunachal_Pradesh-GSVA_cur_2015-16.csv")

arp
arp=arp.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

arp.columns=['Items','Arunanchal Pradesh']

arp
ass=pd.read_csv(r"../input/gdpdatasets/NAD-Assam-GSVA_cur_2015-16.csv")

ass
ass=ass.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

ass.columns=['Items','Assam']

ass
bh=pd.read_csv(r"../input/gdpdatasets/NAD-Bihar-GSVA_cur_2015-16.csv")

bh
bh=bh.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

bh.columns=['Items','Bihar']

bh
ch=pd.read_csv(r"../input/gdpdatasets/NAD-Chhattisgarh-GSVA_cur_2016-17.csv")

ch
ch=ch.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

ch.columns=['Items','Chhattisgarh']

ch
ga=pd.read_csv(r"../input/gdpdatasets/NAD-Goa-GSVA_cur_2015-16.csv")

ga
ga=ga.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

ga.columns=['Items','Goa']

ga
gj=pd.read_csv(r"../input/gdpdatasets/NAD-Gujarat-GSVA_cur_2015-16.csv")

gj
#We may notice that certain values have an extra asterix placed in front of them, these should be removed.

gj=gj.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

gj.columns=['Items','Gujarat']

gj['Items']=gj['Items'].str.replace("*","")

gj
hr=pd.read_csv(r"../input/gdpdatasets/NAD-Haryana-GSVA_cur_2016-17.csv")

hr
hr=hr.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

hr.columns=['Items','Haryana']

hr
hp=pd.read_csv(r"../input/gdpdatasets/NAD-Himachal_Pradesh-GSVA_cur_2014-15.csv")

hp
hp=hp.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

hp.columns=['Items','Himachal Pradesh']

hp
jk=pd.read_csv(r"../input/eda-gdp-analysis-india/NAD-Jammu_Kashmir-GSVA_cur_2015-16.csv")

jk
jk=jk.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

jk.columns=['Items','Jammu Kashmir']

jk
jh=pd.read_csv(r"../input/gdpdatasets/NAD-Jharkhand-GSVA_cur_2015-16.csv")

jh
jh=jh.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

jh.columns=['Items','Jharkhand']

jh
ka=pd.read_csv(r"../input/gdpdatasets/NAD-Karnataka-GSVA_cur_2015-16.csv")

ka
ka=ka.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

ka.columns=['Items','Karnataka']

ka
kl=pd.read_csv(r"../input/gdpdatasets/NAD-Kerala-GSVA_cur_2015-16.csv")

kl
kl=kl.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

kl.columns=['Items','Kerala']

kl['Items']=kl['Items'].str.replace("*","")

kl
mp=pd.read_csv(r"../input/gdpdatasets/NAD-Madhya_Pradesh-GSVA_cur_2016-17.csv")

mp
mp=mp.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

mp.columns=['Items','Madhya Pradesh']

mp
mh=pd.read_csv(r"../input/gdpdatasets/NAD-Maharashtra-GSVA_cur_2014-15.csv")

mh
mh=mh.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

mh.columns=['Items','Maharashtra']

mh
ma=pd.read_csv(r"../input/gdpdatasets/NAD-Manipur-GSVA_cur_2014-15.csv", engine='python')

ma
ma=ma.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

ma.columns=['Items','Manipur']

ma
me=pd.read_csv(r"../input/gdpdatasets/NAD-Meghalaya-GSVA_cur_2016-17.csv")

me
me=me.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

me.columns=['Items','Meghalaya']

me
mi=pd.read_csv(r"../input/gdpdatasets/NAD-Mizoram-GSVA_cur_2014-15.csv")

mi
mi=mi.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

mi.columns=['Items','Mizoram']

mi
ng=pd.read_csv(r"../input/gdpdatasets/NAD-Nagaland-GSVA_cur_2014-15.csv")

ng
ng=ng.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

ng.columns=['Items','Nagaland']

ng
od=pd.read_csv(r"../input/gdpdatasets/NAD-Odisha-GSVA_cur_2016-17.csv")

od
od=od.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

od.columns=['Items','Odisha']

od
pj=pd.read_csv(r"../input/gdpdatasets/NAD-Punjab-GSVA_cur_2014-15.csv")

pj
pj=pj.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

pj.columns=['Items','Punjab']

pj
rj=pd.read_csv(r"../input/gdpdatasets/NAD-Rajasthan-GSVA_cur_2014-15.csv")

rj
rj=rj.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

rj.columns=['Items','Rajasthan']

rj
sk=pd.read_csv(r"../input/gdpdatasets/NAD-Sikkim-GSVA_cur_2015-16.csv")

sk
sk=sk.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

sk.columns=['Items','Sikkim']

sk
tn=pd.read_csv(r"../input/gdpdatasets/NAD-Tamil_Nadu-GSVA_cur_2016-17.csv")

tn
tn=tn.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

tn.columns=['Items','Tamil Nadu']

tn['Items']=tn['Items'].str.replace("*","")

tn
ts=pd.read_csv(r"../input/gdpdatasets/NAD-Telangana-GSVA_cur_2016-17.csv")

ts
ts=ts.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

ts.columns=['Items','Telangana']

ts
tr=pd.read_csv(r"../input/gdpdatasets/NAD-Tripura-GSVA_cur_2014-15.csv")

tr
tr=tr.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

tr.columns=['Items','Tripura']

tr['Items']=tr['Items'].str.replace("*","")

tr
up=pd.read_csv(r"../input/gdpdatasets/NAD-Uttar_Pradesh-GSVA_cur_2015-16.csv")

up
up=up.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

up.columns=['Items','Uttar Pradesh']

up
uk=pd.read_csv(r"../input/gdpdatasets/NAD-Uttarakhand-GSVA_cur_2015-16.csv")

uk
uk=uk.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

uk.columns=['Items','Uttarakhand']

uk['Items']=uk['Items'].str.replace("*","")

uk
ani=pd.read_csv(r"../input/eda-gdp-analysis-india/NAD-Andaman_Nicobar_Islands-GSVA_cur_2014-15.csv")

ani
ani=ani.drop(columns=['S.No.','2011-12','2012-13','2013-14'])

ani.columns=['Items','Andaman Nicobar Islands']

ani['Items']=ani['Items'].str.replace("*","")

ani
cg=pd.read_csv(r"../input/gdpdatasets/NAD-Chandigarh-GSVA_cur_2015-16.csv")

cg
cg=cg.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16'])

cg.columns=['Items','Chandigarh']

cg['Items']=cg['Items'].str.replace("*","")

cg
dl=pd.read_csv(r"../input/gdpdatasets/NAD-Delhi-GSVA_cur_2016-17.csv")

dl
dl=dl.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

dl.columns=['Items','Delhi']

dl['Items']=dl['Items'].str.replace("*","")

dl
py=pd.read_csv(r"../input/gdpdatasets/NAD-Puducherry-GSVA_cur_2016-17.csv")

py
py=py.drop(columns=['S.No.','2011-12','2012-13','2013-14','2015-16','2016-17'])

py.columns=['Items','Puducherry']

py['Items']=py['Items'].str.replace("*","")

py
from functools import reduce

newdf=reduce(lambda x,y: pd.merge(x,y, on='Items', how='inner'), [ap,arp,ass,bh,ch,ga,gj,hr,hp,jk,jh,ka,kl,mp,mh,ma,me,mi,ng,od,pj,rj,sk,tn,ts,tr,up,uk,ani,cg,dl,py])

newdf
newdf=newdf.drop(columns=['Andaman Nicobar Islands','Chandigarh','Delhi','Puducherry'])

newdf
newdf2=newdf.transpose()

newdf2
newdf2.columns=newdf2.iloc[0]

newdf2.drop('Items',axis=0,inplace=True)
newdf2.rename_axis('Items')

newdf2
cols=[i for i in newdf2.columns if i not in ["Items"]]

for col in cols:

    newdf2[col]=newdf2[col].astype(float)
newdf2.info()
newdf2.fillna(0)
sns.set(rc={'figure.figsize':(12,9)})

sns.barplot(x=newdf2['Per Capita GSDP (Rs.)'],y=newdf2.index)
newdf2.sort_values(by=['Per Capita GSDP (Rs.)'],ascending=False)
(newdf2['Per Capita GSDP (Rs.)'].max())/(newdf2['Per Capita GSDP (Rs.)'].min())
newdf3=newdf2[['Primary','Secondary','Tertiary']]

newdf3
newdf4=newdf3.transpose()
newdf4
plot = newdf4.plot.pie(subplots=True, figsize=(100, 100), autopct='%1.1f%%', startangle=0, fontsize=17, layout=(7,4))
p=np.percentile(newdf2['Per Capita GSDP (Rs.)'],[20,50,85,100])

p
C1=newdf2[(newdf2['Per Capita GSDP (Rs.)']<=p[3]) & (newdf2['Per Capita GSDP (Rs.)']>p[2])]

C2=newdf2[(newdf2['Per Capita GSDP (Rs.)']<=p[2]) & (newdf2['Per Capita GSDP (Rs.)']>p[1])]

C3=newdf2[(newdf2['Per Capita GSDP (Rs.)']<=p[1]) & (newdf2['Per Capita GSDP (Rs.)']>p[0])]

C4=newdf2[(newdf2['Per Capita GSDP (Rs.)']<=p[0])]
C1
C2
C3
C4
NewC1=C1[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services','Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services','Taxes on Products','Subsidies on products']]

NewC1
pd.options.mode.chained_assignment = None #to avoid false positives warnings while creating new columns

NewC1['Taxes-Subsidies']=NewC1['Taxes on Products']-NewC1['Subsidies on products']

NewC1
NewC1.drop(['Taxes on Products','Subsidies on products'],axis=1,inplace=True)
NewC1=NewC1.transpose()
NewC1
NewC1["Total"]=NewC1["Goa"]+NewC1["Haryana"]+NewC1["Kerala"]+NewC1["Sikkim"]+NewC1["Uttarakhand"]
NewC1.sort_values(by='Total',ascending=False,inplace=True)

NewC1
NewC1['Goa'].sum() #Validity check for the GSDP from the original Data
plot = NewC1.plot.pie(y='Total', figsize=(100, 100), autopct='%1.1f%%', startangle=0, fontsize=50)
NewC2=C2[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services','Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services','Taxes on Products','Subsidies on products']]

NewC2['Taxes-Subsidies']=NewC2['Taxes on Products']-NewC2['Subsidies on products']

NewC2.drop(['Taxes on Products','Subsidies on products'],axis=1,inplace=True)

NewC2=NewC2.transpose()
NewC2
NewC2['Telangana'].sum() #Validity check for the GSDP from the original Data
NewC2["Total"]=NewC2["Andhra Pradesh"]+NewC2["Arunanchal Pradesh"]+NewC2["Gujarat"]+NewC2["Himachal Pradesh"]+NewC2["Karnataka"]+NewC2["Maharashtra"]+NewC2["Punjab"]+NewC2["Tamil Nadu"]+NewC2["Telangana"]

NewC2.sort_values(by='Total',ascending=False,inplace=True)

NewC2
plot = NewC2.plot.pie(y='Total', figsize=(100, 100), autopct='%1.1f%%', startangle=0, fontsize=50)
#The same process is repeated for all the categories.

NewC3=C3[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services','Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services','Taxes on Products','Subsidies on products']]

NewC3['Taxes-Subsidies']=NewC3['Taxes on Products']-NewC3['Subsidies on products']

NewC3.drop(['Taxes on Products','Subsidies on products'],axis=1,inplace=True)

NewC3=NewC3.transpose()

NewC3
NewC3['Nagaland'].sum() #Validity check for the GSDP from the original Data
NewC3["Total"]=NewC3["Chhattisgarh"]+NewC3["Jammu Kashmir"]+NewC3["Meghalaya"]+NewC3["Mizoram"]+NewC3["Nagaland"]+NewC3["Odisha"]+NewC3["Rajasthan"]+NewC3["Tripura"]

NewC3.sort_values(by='Total',ascending=False,inplace=True)

NewC3
plot = NewC3.plot.pie(y='Total', figsize=(100, 100), autopct='%1.1f%%', startangle=0, fontsize=50)
#The same process is repeated for all the categories.

NewC4=C4[['Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services','Construction','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services','Taxes on Products','Subsidies on products']]

NewC4['Taxes-Subsidies']=NewC4['Taxes on Products']-NewC4['Subsidies on products']

NewC4.drop(['Taxes on Products','Subsidies on products'],axis=1,inplace=True)

NewC4=NewC4.transpose()

NewC4=NewC4.fillna(0)

NewC4
NewC4['Manipur'].sum() #Validity check for the GSDP from the original Data
NewC4["Total"]=NewC4["Assam"]+NewC4["Bihar"]+NewC4["Jharkhand"]+NewC4["Madhya Pradesh"]+NewC4["Manipur"]+NewC4["Uttar Pradesh"]

NewC4.sort_values(by='Total',ascending=False,inplace=True)

NewC4
plot = NewC4.plot.pie(y='Total', figsize=(100, 100), autopct='%1.1f%%', startangle=0, fontsize=50)
dprate=pd.read_csv(r"../input/eda-gdp-analysis-india/Dropout rate dataset.csv")

dprate
dprate=dprate[['Level of Education - State','Primary - 2014-2015.1','Upper Primary - 2014-2015','Secondary - 2014-2015']]

dprate.columns=['State','Primary 14-15','Upper Primary 14-15','Secondary 14-15']

dprate
dprate.at[2,'State'] = 'Arunanchal Pradesh'

dprate.at[6,'State'] = 'Chhattisgarh'

dprate.at[14,'State'] = 'Jammu Kashmir'

dprate.at[34,'State'] = 'Uttarakhand'

dprate.at[36,'State'] = 'All_India GDP'

dprate
dprate.set_index('State',inplace=True)

dprate
PCGSDP=newdf2[['Per Capita GSDP (Rs.)']]

PCGSDP
newdprate=pd.merge(dprate, PCGSDP, left_index=True, right_index=True)

newdprate
newdprate.corr()
f = plt.figure(figsize=(19, 15))

plt.matshow(newdprate.corr(), fignum=f.number)

plt.xticks(range(newdprate.shape[1]), newdprate.columns, fontsize=14, rotation=45)

plt.yticks(range(newdprate.shape[1]), newdprate.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);