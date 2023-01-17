import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/Absenteeism_at_work.csv')
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)
v=df[df['Absenteeism time in hours']==0].count()

print(v)
print("% of No Absentees="+str(v[0]/df.shape[0]))
df['Absenteeism time in hours'].mean()
df['Absenteeism time in hours'].median()
#Function for defing bar text

def barlabels(bars,c='black'):

    for bar in bars:

        yval = round(bar.get_height(),2)

        plt.text(bar.get_x()+0.25,yval, yval,verticalalignment='bottom',color=c,fontweight='bold')
r= df.groupby('Reason for absence')

r['Absenteeism time in hours'].count()
arr=r['Absenteeism time in hours'].count()

arr=np.array(arr)

print(arr)

fig, ax = plt.subplots(figsize=(10,10))

table=[

    'No Reason given','Certain infectious and parasitic diseases', 

'Neoplasms', 

'blood-forming organs and involving the immune mechanism', 

'Endocrine, nutritional and metabolic diseases', 

'Mental and behavioural disorders', 

'Diseases of the nervous system', 

'Diseases of the eye and adnexa', 

'Diseases of the ear and mastoid process', 

'Diseases of the circulatory system', 

'Diseases of the respiratory system', 

'Diseases of the digestive system', 

'Diseases of the skin and subcutaneous tissue', 

'Diseases of the musculoskeletal system and connective tissue', 

'Diseases of the genitourinary system',

'Pregnancy, childbirth and the puerperium',

'Certain conditions originating in the perinatal period',

'Congenital malformations, deformations and chromosomal abnormalities', 

'Symptoms, signs and abnormal clinical and laboratory findings', 

'Injury, poisoning and certain other consequences of external causes', 

'Factors influencing health status and contact with health services.',

'patient follow-up',

'medical consultation',

'blood donation',

'laboratory examination',

'unjustified absence',

'physiotherapy',

'dental consultation']

plt.barh(y=np.arange(len(arr)),width=arr,label='No. of people',color='#00C6C5')

plt.yticks(np.arange(len(arr)),table,rotation=0)

plt.ylabel('Reason of Absence')

plt.xlabel('Count of people')

plt.title('Reason vs Count',fontweight='bold')

plt.legend()

for i, v in enumerate(arr):

    ax.text(v+2, i, str(v), color='black',fontweight='bold')

plt.show()
plt.figure(figsize=(10,5))

r= df.groupby('Month of absence')

arr=r['Absenteeism time in hours'].count()

arr=np.array(arr)

arr=arr[1:]

arr1=r['Absenteeism time in hours'].mean()

arr1=np.array(arr1)

arr1=arr1[1:]

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='#E6D3E3',label='Count')

bars1=plt.bar(x=np.arange(len(arr)),height=arr1,color='#461C7C',label='Mean')

plt.xlabel('Month of year')

plt.xticks(np.arange(len(arr)),['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])

plt.title('Variation due to month')

plt.legend()

barlabels(bars)

barlabels(bars1,'red')

plt.show()
r['Absenteeism time in hours'].mean()
plt.figure(figsize=(7,7))

r= df.groupby('Day of the week')

arr=r['Absenteeism time in hours'].count()

arr=np.array(arr)

print(arr)

arr1=r['Absenteeism time in hours'].mean()

arr1=np.array(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='grey',label='Count')

bars1=plt.bar(x=np.arange(len(arr)),height=arr1,color='k',label='Mean')

plt.xlabel('Days of week')

plt.xticks(np.arange(len(arr)),['MON','TUE','WED','THUR','FRI'])

plt.title('Days of week variation')

plt.legend()

barlabels(bars)

barlabels(bars1,'white')

plt.show()
r['Absenteeism time in hours'].mean()
plt.figure(figsize=(7,7))

r= df.groupby('Seasons')

arr=r['Absenteeism time in hours'].count()

arr=np.array(arr)

print(arr)

arr1=r['Absenteeism time in hours'].mean()

arr1=np.array(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='k',label='Count')

bars1=plt.bar(x=np.arange(len(arr)),height=arr1,color='blue',label='Mean')

plt.xlabel('Season')

plt.xticks(np.arange(len(arr)),['Summer','Autumn','Winter','Spring'])

plt.title('Season variation')

plt.legend()

barlabels(bars)

barlabels(bars1,'white')

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x='Month of absence',y='Absenteeism time in hours',data=df,hue='Seasons')

sns.jointplot(x='Seasons',y='Month of absence',data=df,kind='reg',color='m')
r['Absenteeism time in hours'].mean()
df['Transportation expense'].mean()
df2=df.assign(avg=df['Transportation expense']/df['Distance from Residence to Work'])

#print(df1)
plt.figure(figsize=(10,10))

sns.jointplot(x='avg',y='Absenteeism time in hours',data=df2,kind='reg',color='crimson')

plt.title('Expense/Distance ratio')

plt.xlabel('Ratio Expense/Dist')

plt.show()
errorwindow=0.15

maximum=df2['avg'].max()

minimum=df2['avg'].min()

mean=df2['avg'].mean()

mean1=mean-errorwindow*mean

mean2=mean+errorwindow*mean

bins=list([minimum-1,mean1,mean2,maximum+1])

print(bins)
df2=df2.assign(avg1=(pd.cut(df2['avg'], bins=bins)))

print(df2['avg1'].value_counts())
plt.figure(figsize=(5,7))

r= df2.groupby('avg1')

arr=r['Absenteeism time in hours'].count()

arr1=r['Absenteeism time in hours'].mean()

print(arr)

print(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='orange',label='Count')

bars1=plt.bar(x=np.arange(len(arr)),height=arr1,color='blue',label='Mean')

plt.xlabel('Expense/Distance')

plt.ylabel('Values')

plt.xticks(np.arange(len(arr)),['Below Mean','Approx Mean','Above Mean'])

plt.title('Expense/Distance ratio')

plt.legend()

barlabels(bars)

barlabels(bars1)

plt.show()
sns.jointplot(x='Transportation expense',y='Month of absence',data=df,kind='hex',color='green')
sns.jointplot(x='Transportation expense',y='Distance from Residence to Work',kind='reg',data=df)
r= df.groupby('Distance from Residence to Work')

r['Absenteeism time in hours'].count()

#r['Absenteeism time in hours'].median()
maximum=df['Distance from Residence to Work'].max()

minimum=df['Distance from Residence to Work'].min()

bins=list(np.linspace(minimum-1,maximum+1,num=8))

print(bins)
df1=df.assign(dist=pd.cut(df['Distance from Residence to Work'], bins=bins))

print(df1['dist'].value_counts())
plt.figure(figsize=(10,10))

r= df1.groupby('dist')

arr=r['Absenteeism time in hours'].mean()

print(arr)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='purple',label='Mean')

plt.xlabel('Distance')

plt.ylabel('Absent Time')

z=list(arr.index)

plt.xticks(np.arange(len(z)),z)

plt.title('Distance Variation')

plt.legend()

barlabels(bars)

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(x=df['Age'],y=df['Absenteeism time in hours'])

plt.ylabel('Absenteeism time in hours')

plt.xlabel('Age')

plt.title('Time vs Age')

plt.show()
maximum=df['Age'].max()

minimum=df['Age'].min()

bins=list(np.linspace(minimum-1,maximum+1,num=7))

print(bins)
df1=df.assign(age=pd.cut(df['Age'], bins=bins))

print(df1['age'].value_counts())
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

r= df1.groupby('age')

arr=r['Absenteeism time in hours'].count()

#arr=np.array(arr)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='green',label='Count')

plt.xlabel('Age')

plt.ylabel('Frequency')

z=list(arr.index)

plt.xticks(np.arange(len(z)),z)

plt.title('Age Variation(Count)')

plt.legend()

barlabels(bars)

plt.show()



plt.figure(figsize=(10,10))

plt.subplot(2,1,2)

arr1=r['Absenteeism time in hours'].mean()

print(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr1,color='black',label='Mean')

plt.xlabel('Age')

plt.ylabel('Absent Time')

z=list(arr.index)

plt.xticks(np.arange(len(z)),z)

plt.title('Age Variation(Mean)')

plt.legend()

barlabels(bars)

plt.show()
plt.figure(figsize=(12,6))

sns.lmplot(x='Age',y='Absenteeism time in hours',data=df,hue='Day of the week',height=5,aspect=2)
#plt.figure(figsize=(5,5))

r= df.groupby('Work load Average/day ')

arr=r['Absenteeism time in hours'].count()

arr=np.array(arr)

print(arr)
r['Absenteeism time in hours'].count()
r['Absenteeism time in hours'].mean()
plt.figure(figsize=(10,10))

plt.scatter(x=df['Work load Average/day '],y=df['Absenteeism time in hours'])

r= df.groupby('Disciplinary failure')

r['Absenteeism time in hours'].count()
r['Absenteeism time in hours'].mean()
maximum=df['Service time'].max()

minimum=df['Service time'].min()

bins=list(np.linspace(minimum-1,19,num=6))

print(bins)
df1=df.assign(st=pd.cut(df['Service time'], bins=bins))

print(df1['st'].value_counts())
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

r= df1.groupby('st')

arr=r['Absenteeism time in hours'].count()

print(arr)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='green',label='Count')

plt.xlabel('Service Time(years)')

plt.ylabel('Frequency')

z=list(arr.index)

plt.xticks(np.arange(len(z)),z)

plt.title('Service Time Variation(Count)')

plt.legend()

barlabels(bars)

plt.show()



plt.figure(figsize=(10,10))

plt.subplot(2,1,2)

arr1=r['Absenteeism time in hours'].mean()

print(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr1,color='black',label='Mean')

plt.xlabel('Service Time(years)')

plt.ylabel('Absent Time')

z=list(arr.index)

plt.xticks(np.arange(len(z)),z)

plt.title('Service Time Variation(Mean)')

plt.legend()

barlabels(bars)

plt.show()
r= df.groupby('Education')

r['Absenteeism time in hours'].count()
arr1=r['Absenteeism time in hours'].mean()

arr1=np.array(arr1)

bars=plt.bar(np.arange(len(arr1)),height=arr1,label='mean',color='#58A45E')

plt.ylabel('Absent Time')

plt.title('Education Variation')

plt.xticks(np.arange(len(arr1)),['high school', 'graduate', 'postgraduate', 'master and doctor'],rotation=15)

plt.legend()

barlabels(bars)

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(y=df['Hit target'],x=df['Service time'])
sns.lmplot(x='Age',y='Service time',data=df)
sns.lmplot(x='Hit target',y='Work load Average/day ',data=df)
bins=list([0,18.4,24.9,29.9,39.9])

print(bins)
df1=df.assign(bmi=pd.cut(df['Body mass index'], bins=bins))

print(df1['bmi'].value_counts())
plt.figure(figsize=(5,7))

r= df1.groupby('bmi')

arr=r['Absenteeism time in hours'].count()

arr1=r['Absenteeism time in hours'].mean()

print(arr)

print(arr1)

bars=plt.bar(x=np.arange(len(arr)),height=arr,color='crimson',label='Count')

bars1=plt.bar(x=np.arange(len(arr)),height=arr1,color='yellow',label='Mean')

plt.ylabel('Values')

plt.xticks(np.arange(len(arr)),['Underweight','Healthy Weight','OverWeight','Obese'],rotation=30)

plt.title('BMI variation')

plt.legend(loc='upper left')

barlabels(bars)

#barlabels(bars1,'white')

plt.show()


z=df[df['Body mass index']>=29.9]
print(df['Transportation expense'].mean())

print(z['Transportation expense'].mean())
print(df['Distance from Residence to Work'].mean())

print(z['Distance from Residence to Work'].mean())
z1=df2[df2['Body mass index']>=29.9]

print(df2['avg'].mean())

print(z1['avg'].mean())
print(z['Reason for absence'].value_counts())
z1=df[df['Seasons']==1]

z2=df[df['Seasons']==2]

z3=df[df['Seasons']==3]

z4=df[df['Seasons']==4]
print(df['Transportation expense'].mean())

print(z1['Transportation expense'].mean())

print(z2['Transportation expense'].mean())

print(z3['Transportation expense'].mean())

print(z4['Transportation expense'].mean())
print(z2['Age'].mean())

print(z2['Transportation expense'].mean())