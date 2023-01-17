import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
xls = pd.ExcelFile('../input/businesscasestudy-vindiata/Vindiata Case study.xlsx')
df1 = pd.read_excel(xls, 'Operations', )
df2 = pd.read_excel(xls, 'AC characteristics')
df3=df1.groupby("Aircraft Type").sum()
df3['totalhrs']=df3.sum(axis = 1, skipna = True) 
df3
df3.drop(['JAN', 'FEB', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUG', 'SEPT',
       'OCT', 'NOV', 'DEC'],axis=1,inplace=True)
df3.reset_index()
df2=pd.merge(df2,df3,left_on='Aircraft Type',right_on='Aircraft Type')
df2
df2['total_cost']=df2['Costs per flight hour']*df2['totalhrs']
df2['lwst_cost_per_seat']=(df2['total_cost']/((df2['totalhrs']*df2['Ave. Speed (km/h)'])*df2['Number of Seats']))
df2
Total_hours=df2['totalhrs'].sum()
Total_hours
#Which aircraft type  has the lowest cost per seat per km flown?

df2.loc[df2['lwst_cost_per_seat'].idxmin(),"Aircraft Type"]
sns.heatmap(df2.corr(),annot=True)
#no. of flights of a particular type
df5 = pd.read_excel(xls, 'Operations', )
airc=df5['Aircraft Type'].value_counts()
airc
df6 = pd.read_excel(xls, 'City pairs')
df6=df6.sort_values(by=['Distance (km)'],ascending=False).reset_index().drop(['index'],axis=1)
a_dictionary = {}
for i in range(len(df6["Distance (km)"])) :
    a_dictionary[str(df6.loc[i,"Origin City"]) +'-'+str(df6.loc[i,"Desitnation City"])]=df2.loc[(df2["Range (Km)"]>=df6.loc[i,"Distance (km)"]),"Aircraft Type"]
a_dictionary
p1=pd.DataFrame.from_dict(a_dictionary, orient='index')
df2
df6
p1
ls=[]
totallst=[]
for k in range(p1.shape[0]):
    for i in range(p1.shape[1]) :
        if pd.isna(p1.iloc[k,i]) == True :             
            continue
        p=((df6.loc[k,'Distance (km)']/df2.loc[p1.columns[i],'Ave. Speed (km/h)'])*df2.loc[p1.columns[i],'Costs per flight hour'])*math.ceil(df6.loc[k,'Pass. Demand \n(per day)']/df2.loc[p1.columns[i],'Number of Seats'])
        ls.append(p)                                                                                                                                           
    totallst.append(ls)
    ls=[]
rflights=pd.DataFrame(totallst,columns=['A330','B747','A320','B737','Q400','ATR72'])
rflights
rflights.min(axis=1)
'''
Which aircraft types are best suited for their operation?
so for path BB-CC : A330 
for path AA-BB: A330 
for path AA-DD: A320
for path CC-AA: A320
'''
#So now if you want to know how many flights of a particular aircraft type will be required
#for BB-CC A330
print(math.ceil(df6.loc[0,'Pass. Demand \n(per day)']/df2.loc[1,'Number of Seats']))
#for AA-BB A330
print(math.ceil(df6.loc[1,'Pass. Demand \n(per day)']/df2.loc[1,'Number of Seats']))
#for AA-DD A320
print(math.ceil(df6.loc[2,'Pass. Demand \n(per day)']/df2.loc[0,'Number of Seats']))
#for CC-AA A320
print(math.ceil(df6.loc[3,'Pass. Demand \n(per day)']/df2.loc[0,'Number of Seats']))
