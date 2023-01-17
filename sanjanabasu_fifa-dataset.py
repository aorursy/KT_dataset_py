import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv('../input/fifa19/data.csv')
data.head()
def clean_value(x):
    
    if isinstance(x,str):
        a=x.replace('€','')
        
        if 'M' in a:
            a=float(a.replace('M', ''))*1000000
        elif 'K' in a:
            a=float(a.replace('K', ''))*1000
        return float(a)
       
data['Value'] = data['Value'].apply(clean_value)
data['Wage'] = data['Wage'].apply(clean_value)
def clean_date(x):
    l=[]
    if isinstance(x,str):
        l=x.split(", ")
        x=l[-1]
    return(x)

data['Joined']=data['Joined'].replace(np.nan,0)
data['Joined'] = data['Joined'].apply(clean_date).astype('int')
import datetime
data['Contract Valid Until']=pd.to_datetime(data['Contract Valid Until'])
data['year'] = pd.DatetimeIndex(data['Contract Valid Until']).year.astype(str)
def clean_height(x):
    l=[]
    s=1
    if isinstance(x,str):
        l=x.split("'")
        i=int(l[0])
        j=int(l[1])
        s=((i*12)+j)/12
    return(s)


data['Height'] = data['Height'].apply(clean_height)
def clean_weight(x):
    
    if isinstance(x,str):
        return(x.replace('lbs', ''))
    return(x)

data['Weight'] = data['Weight'].apply(clean_weight).astype('float')
data['Release Clause'] = data['Release Clause'].apply(clean_value)
data['Release Clause']=data['Release Clause'].replace(np.nan,data['Release Clause'].mean())
data=data[['ID', 'Name','Overall','Potential','Value', 'Wage','Age','International Reputation','Skill Moves','Position','Joined','Contract Valid Until', 'Height', 'Weight','Release Clause', 'year']]
data
data['Weight'].fillna((data['Weight'].mean()), inplace = True) 
data['Height'].fillna((data['Height'].mean()), inplace = True) 
data['International Reputation'].fillna(data['International Reputation'].mean(), inplace = True) 
data['Skill Moves'].fillna(data['Skill Moves'].mean(), inplace = True)  
data['Position'].fillna('Not available', inplace = True) 
data['Joined'].fillna('Not available', inplace = True) 
data['Contract Valid Until'].fillna('Not available', inplace = True) 
data['year'].fillna('Not available', inplace = True) 
data['Release Clause'].fillna(data['Release Clause'].mean(), inplace = True) 
data['Value'].fillna((data['Value'].mean()), inplace = True) 
data['Wage'].fillna((data['Wage'].mean()), inplace = True) 
data['Overall'].plot(kind='kde')
data['Overall'].skew()
sns.pairplot(data[['Overall', 'Value', 'Wage', 'International Reputation', 'Height', 'Weight', 'Release Clause']])
data1=data[data['year']=='2020.0'].sort_values(by='Overall',ascending=False).head(20)
data1[['ID', 'Name','Overall','Potential','Value', 'Wage','Age','International Reputation','Skill Moves','Position','Height', 'Weight','Release Clause']]
print("The average wage of top 20 players by overall rating is=",data1['Wage'].mean())
print("The average age of top 20 players by overall rating is=",data1['Age'].mean())
data3=data[['Overall','Value']]
cor=data3.corr()
print('Yes it is-',cor.iloc[0,1])
data4=data[['Name','Position','Overall']]
data5=pd.pivot_table(data4,index=['Position','Name'])
data5
d1=data.groupby(['Position']).mean().T
l1=d1.columns
a=data5.loc[l1[0]].sort_values('Overall',ascending=False).head(5)
for i in range(1,len(l1)):
    b=data5.loc[l1[i]].sort_values('Overall',ascending=False).head(5)
    a=pd.concat([a,b])
c=pd.merge(a,data[['ID','Name','Position','Overall','Wage']],on=['Name','Overall'],how='left')                 
final_c=c.groupby(['Position','Name']).mean()

for i in range(0,len(l1)):
    print("For",l1[i])
    print(final_c.loc[l1[i]])
    print()
    
dup1=final_c[final_c.duplicated(keep=False)]
dup2=dup1.drop_duplicates('ID')
dup2.count()

dup2
wage=final_c.groupby(['Position']).mean()
print('Average wage to be paid to top 5 players by position is as follows:')
wage.loc[:,('Wage')]
#wage.drop('Overall',axis=1,inplace=True)
striker=data[((data['Position']=='CF')|(data['Position']=='ST'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
striker_top=data[(data['Position']=='CF')|(data['Position']=='ST')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
Right_forward=data[((data['Position']=='RF')|(data['Position']=='RS')|(data['Position']=='RW')|(data['Position']=='RAM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
R_striker=data[(data['Position']=='RF')|(data['Position']=='RS')|(data['Position']=='RW')|(data['Position']=='RAM')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
Left_forward=data[((data['Position']=='LF')|(data['Position']=='LS')|(data['Position']=='LW')|(data['Position']=='LAM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
L_striker=data[(data['Position']=='LF')|(data['Position']=='LAM')|(data['Position']=='LW')|(data['Position']=='LS')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
C_mid=data[((data['Position']=='CAM')|(data['Position']=='CM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
L_back=data[((data['Position']=='LWB')|(data['Position']=='LB')|(data['Position']=='LCB'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
R_back=data[((data['Position']=='RB')|(data['Position']=='RWB')|(data['Position']=='RCB'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
C_stopper=data[((data['Position']=='CB')|(data['Position']=='CDM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
C_stopper1=data[(data['Position']=='CB')|(data['Position']=='CDM')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
Left_mid=data[((data['Position']=='LM')|(data['Position']=='LCM')|(data['Position']=='LDM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
Right_mid=data[((data['Position']=='RM')|(data['Position']=='RCM')|(data['Position']=='RDM'))&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)
GK=data[(data['Position']=='GK')&(data['year']<='2020')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)
GK1=data[(data['Position']=='GK')][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)

l2=[striker,striker_top,Right_forward,R_striker,Left_forward,L_striker,C_mid,L_back,R_back,C_stopper,C_stopper1,Left_mid,Right_mid,GK,GK1]
final_20=pd.concat(l2)
final_20=final_20.drop_duplicates()
print(final_20.info())
final_20
