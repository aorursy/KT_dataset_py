# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataPath='../input/'
#read the train csv file using the first row as the col names and using the ID for index
original_df=pd.read_csv(dataPath+'train.csv',index_col='AnimalID')
#describing the dataframe
original_df.describe()
#split dog data and cat data
dog_df=original_df.loc[original_df['AnimalType']=='Dog',:].fillna(value='No value')
cat_df=original_df.loc[original_df['AnimalType']=='Cat',:].fillna(value='No value')
#dogs' most popular names
count_by_name=dog_df['Name'].value_counts().head(20)
dog_name_list=tuple(count_by_name.index)

#cats' most popular names
count_by_name=cat_df['Name'].value_counts().head(20)
cat_name_list=tuple(count_by_name.index)
#see the influence of the names#
#choose the most appeared names and print how it infulences
feature='Name'
feature_dog_df=dog_df.loc[dog_df['Name'].isin(dog_name_list),[feature,'OutcomeType']]
feature_cat_df=cat_df.loc[cat_df['Name'].isin(cat_name_list),[feature,'OutcomeType']]

unique_outcome=original_df['OutcomeType'].unique()
#Doing the statistics
fraction_dog=np.zeros([20,unique_outcome.size])
fraction_cat=np.zeros([20,unique_outcome.size])

df_sum_dog=pd.DataFrame(data=fraction_dog,index=dog_name_list,columns=unique_outcome)
df_sum_cat=pd.DataFrame(data=fraction_cat,index=cat_name_list,columns=unique_outcome)
for ID in feature_dog_df.index:
    name=feature_dog_df.ix[ID,'Name']
    outcome=feature_dog_df.ix[ID,'OutcomeType']
    df_sum_dog.ix[name,outcome]=df_sum_dog.ix[name,outcome]+1
    
for ID in feature_cat_df.index:
    name=feature_cat_df.ix[ID,'Name']
    outcome=feature_cat_df.ix[ID,'OutcomeType']
    df_sum_cat.ix[name,outcome]=df_sum_cat.ix[name,outcome]+1
    
#turn to fractions
df_fraction_dog=df_sum_dog.div(df_sum_dog.sum(axis=1),axis=0)
df_fraction_cat=df_sum_cat.div(df_sum_cat.sum(axis=1),axis=0)

#sortting by survive rate
df_fraction_dog['Survived']=df_fraction_dog.Adoption+df_fraction_dog.Return_to_owner+df_fraction_dog.Transfer
df_fraction_cat['Survived']=df_fraction_cat.Adoption+df_fraction_cat.Return_to_owner+df_fraction_cat.Transfer

df_fraction_cat=df_fraction_cat.sort_values(by='Survived')
df_fraction_dog=df_fraction_dog.sort_values(by='Survived')

#plotting name(most popular 20) v.s. outcome

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Cat''s name v.s outcome')
plt.xlabel('Name')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(20),df_fraction_cat.ix[:,'Return_to_owner'],color='brown')
cat_bottom1=df_fraction_cat.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(20),df_fraction_cat.ix[:,'Adoption'],bottom=cat_bottom1,color='orange')
cat_bottom2=cat_bottom1+df_fraction_cat.ix[:,'Adoption']
bar3=plt.bar(np.arange(20),df_fraction_cat.ix[:,'Transfer'],bottom=cat_bottom2,color='green')
cat_bottom3=cat_bottom2+df_fraction_cat.ix[:,'Transfer']
bar4=plt.bar(np.arange(20),df_fraction_cat.ix[:,'Euthanasia'],bottom=cat_bottom3,color='white')
cat_bottom4=cat_bottom3+df_fraction_cat.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(20),df_fraction_cat.ix[:,'Died'],bottom=cat_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(20),df_fraction_cat.sort_values(by='Survived').index,rotation=40,fontsize=8)

plt.subplot(1,2,2)
plt.title('dog''s name v.s outcome')
plt.xlabel('Name')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(20),df_fraction_dog.ix[:,'Return_to_owner'],color='brown')
dog_bottom1=df_fraction_dog.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(20),df_fraction_dog.ix[:,'Adoption'],bottom=dog_bottom1,color='orange')
dog_bottom2=dog_bottom1+df_fraction_dog.ix[:,'Adoption']
bar3=plt.bar(np.arange(20),df_fraction_dog.ix[:,'Transfer'],bottom=dog_bottom2,color='green')
dog_bottom3=dog_bottom2+df_fraction_dog.ix[:,'Transfer']
bar4=plt.bar(np.arange(20),df_fraction_dog.ix[:,'Euthanasia'],bottom=dog_bottom3,color='white')
dog_bottom4=dog_bottom3+df_fraction_dog.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(20),df_fraction_dog.ix[:,'Died'],bottom=dog_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(20),df_fraction_dog.sort_values(by='Survived').index,rotation=40,fontsize=8)
plt.show()
#get an approximate view on the influence of month of the datetime
dog_df['Month']=dog_df['DateTime'].apply(lambda x : x.split(' ')[0].split('-')[1])
cat_df['Month']=cat_df['DateTime'].apply(lambda x : x.split(' ')[0].split('-')[1])
#preparing the plotting data
unique_outcome=original_df['OutcomeType'].unique()
Month_list=sorted(dog_df['Month'].unique())

fraction_dog=np.zeros([len(Month_list),unique_outcome.size])
fraction_cat=np.zeros([len(Month_list),unique_outcome.size])

df_sum_dog=pd.DataFrame(data=fraction_dog,index=Month_list,columns=unique_outcome)
df_sum_cat=pd.DataFrame(data=fraction_cat,index=Month_list,columns=unique_outcome)

for ID in dog_df.index:
    Month=dog_df.ix[ID,'Month']
    outcome=dog_df.ix[ID,'OutcomeType']
    df_sum_dog.ix[Month,outcome]=df_sum_dog.ix[Month,outcome]+1
    
for ID in cat_df.index:
    Month=cat_df.ix[ID,'Month']
    outcome=cat_df.ix[ID,'OutcomeType']
    df_sum_cat.ix[Month,outcome]=df_sum_cat.ix[Month,outcome]+1
    
df_fraction_dog=df_sum_dog.div(df_sum_dog.sum(axis=1),axis=0)
df_fraction_cat=df_sum_cat.div(df_sum_cat.sum(axis=1),axis=0)
#plotting month(from datetime) v.s. outcome
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Cat''s month v.s outcome')
plt.xlabel('Month')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(12),df_fraction_cat.ix[:,'Return_to_owner'],color='brown')
cat_bottom1=df_fraction_cat.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(12),df_fraction_cat.ix[:,'Adoption'],bottom=cat_bottom1,color='orange')
cat_bottom2=cat_bottom1+df_fraction_cat.ix[:,'Adoption']
bar3=plt.bar(np.arange(12),df_fraction_cat.ix[:,'Transfer'],bottom=cat_bottom2,color='green')
cat_bottom3=cat_bottom2+df_fraction_cat.ix[:,'Transfer']
bar4=plt.bar(np.arange(12),df_fraction_cat.ix[:,'Euthanasia'],bottom=cat_bottom3,color='white')
cat_bottom4=cat_bottom3+df_fraction_cat.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(12),df_fraction_cat.ix[:,'Died'],bottom=cat_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(12),df_fraction_cat.index)

plt.subplot(1,2,2)
plt.title('dog''s month v.s outcome')
plt.xlabel('Month')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(12),df_fraction_dog.ix[:,'Return_to_owner'],color='brown')
dog_bottom1=df_fraction_dog.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(12),df_fraction_dog.ix[:,'Adoption'],bottom=dog_bottom1,color='orange')
dog_bottom2=dog_bottom1+df_fraction_dog.ix[:,'Adoption']
bar3=plt.bar(np.arange(12),df_fraction_dog.ix[:,'Transfer'],bottom=dog_bottom2,color='green')
dog_bottom3=dog_bottom2+df_fraction_dog.ix[:,'Transfer']
bar4=plt.bar(np.arange(12),df_fraction_dog.ix[:,'Euthanasia'],bottom=dog_bottom3,color='white')
dog_bottom4=dog_bottom3+df_fraction_dog.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(12),df_fraction_dog.ix[:,'Died'],bottom=dog_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(12),df_fraction_dog.index)
plt.show()
#Begin to deal with the sex column
#preparing for the data
#filling the n/a
dog_df.ix[dog_df['SexuponOutcome']=='No value','SexuponOutcome']='Unknown'
cat_df.ix[cat_df['SexuponOutcome']=='No value','SexuponOutcome']='Unknown'


unique_sex=dog_df.SexuponOutcome.unique()
unique_outcome=dog_df.OutcomeType.unique()
sum_df=pd.DataFrame(np.zeros([unique_sex.size,unique_outcome.size]),index=unique_sex,columns=unique_outcome)

for AnimalID in dog_df.index:
    Outcome=dog_df.ix[AnimalID,'OutcomeType']
    Sex=dog_df.ix[AnimalID,'SexuponOutcome']
    sum_df.ix[Sex,Outcome]+=1
    
fraction_dog=sum_df.div(sum_df.sum(axis=1),axis=0)

for AnimalID in cat_df.index:
    Outcome=cat_df.ix[AnimalID,'OutcomeType']
    Sex=cat_df.ix[AnimalID,'SexuponOutcome']
    sum_df.ix[Sex,Outcome]+=1
    
fraction_cat=sum_df.div(sum_df.sum(axis=1),axis=0)

fraction_dog['Survived']=fraction_dog.Adoption+fraction_dog.Return_to_owner+fraction_dog.Transfer
fraction_cat['Survived']=fraction_cat.Adoption+fraction_cat.Return_to_owner+fraction_cat.Transfer
#plottng sex v.s. outcome
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Cat''s sex v.s outcome')
plt.xlabel('sex type')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(5),fraction_cat.ix[:,'Return_to_owner'],color='brown')
cat_bottom1=fraction_cat.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(5),fraction_cat.ix[:,'Adoption'],bottom=cat_bottom1,color='orange')
cat_bottom2=cat_bottom1+fraction_cat.ix[:,'Adoption']
bar3=plt.bar(np.arange(5),fraction_cat.ix[:,'Transfer'],bottom=cat_bottom2,color='green')
cat_bottom3=cat_bottom2+fraction_cat.ix[:,'Transfer']
bar4=plt.bar(np.arange(5),fraction_cat.ix[:,'Euthanasia'],bottom=cat_bottom3,color='white')
cat_bottom4=cat_bottom3+fraction_cat.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(5),fraction_cat.ix[:,'Died'],bottom=cat_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(5),fraction_cat.index,rotation=20)

plt.subplot(1,2,2)
plt.title('dog''s sex v.s outcome')
plt.xlabel('sex type')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(5),fraction_dog.ix[:,'Return_to_owner'],color='brown')
dog_bottom1=fraction_dog.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(5),fraction_dog.ix[:,'Adoption'],bottom=dog_bottom1,color='orange')
dog_bottom2=dog_bottom1+fraction_dog.ix[:,'Adoption']
bar3=plt.bar(np.arange(5),fraction_dog.ix[:,'Transfer'],bottom=dog_bottom2,color='green')
dog_bottom3=dog_bottom2+fraction_dog.ix[:,'Transfer']
bar4=plt.bar(np.arange(5),fraction_dog.ix[:,'Euthanasia'],bottom=dog_bottom3,color='white')
dog_bottom4=dog_bottom3+fraction_dog.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(5),fraction_dog.ix[:,'Died'],bottom=dog_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(5),fraction_dog.index,rotation=20)

plt.show()
#define functions for age comparing and converting
import re

def AgeCompare(x,y):
    if x=='No value':
        xValue=0
    else:
        xValue=int(x.split(" ")[0])
    
    if y=='No value':
        yValue=0
    else:
        yValue=int(y.split(" ")[0])
    
    if  re.match('[\w]*[\s]week[\w]*',x):
        xDay=xValue*7
    elif re.match('[\w]*[\s]month[\w]*',x):
        xDay=xValue*30
    elif re.match('[\w]*[\s]year[\w]*',x):
        xDay=xValue*365
    else:
        xDay=xValue
    
    if  re.match('[\w]*[\s]week[\w]*',y):
        yDay=yValue*7
    elif re.match('[\w]*[\s]month[\w]*',y):
        yDay=yValue*30
    elif re.match('[\w]*[\s]year[\w]*',y):
        yDay=yValue*365
    else:
        yDay=yValue
     
    if xDay < yDay:                       
        return -1                       
    elif xDay > yDay:                       
        return 1                       
    else:                       
        return 0
    
def AgeConvert(x):
    if x=='No value':
        xValue=0
    else:
        xValue=int(x.split(" ")[0])
        
    if  re.match('[\w]*[\s]week[\w]*',x):
        xDay=xValue*7
    elif re.match('[\w]*[\s]month[\w]*',x):
        xDay=xValue*30
    elif re.match('[\w]*[\s]year[\w]*',x):
        xDay=xValue*365
    else:
        xDay=xValue
        
    return xDay
#mid-value for dogs below 1 year
dog_below_1=dog_df.ix[dog_df['AgeuponOutcome'].apply(lambda x : AgeCompare(x,'1 year')<0),'AgeuponOutcome']
dog_below_1=sorted(dog_below_1,key=lambda k : AgeConvert(k))
print("mid-value for dogs below 1 year is "+dog_below_1[int(len(dog_below_1)/2)])
#mid-value for cats below 1 year
cat_below_1=cat_df.ix[cat_df['AgeuponOutcome'].apply(lambda x : AgeCompare(x,'1 year')<0),'AgeuponOutcome']
cat_below_1=sorted(cat_below_1,key=lambda k : AgeConvert(k))
print("mid-value for cats below 1 year is "+cat_below_1[int(len(cat_below_1)/2)])
#preparing the data
unique_age=sorted(original_df.fillna('No value').AgeuponOutcome.unique(),key=lambda k : AgeConvert(k))
unique_outcome=original_df.fillna('No value')['OutcomeType'].unique()
sum_df=pd.DataFrame(np.zeros([len(unique_age),unique_outcome.size]),index=unique_age,columns=unique_outcome)

for AnimalID in dog_df.index:
    Outcome=dog_df.ix[AnimalID,'OutcomeType']
    Age=dog_df.ix[AnimalID,'AgeuponOutcome']
    sum_df.ix[Age,Outcome]+=1
    
fraction_dog=sum_df.div(sum_df.sum(axis=1),axis=0)

for AnimalID in cat_df.index:
    Outcome=cat_df.ix[AnimalID,'OutcomeType']
    Age=cat_df.ix[AnimalID,'AgeuponOutcome']
    sum_df.ix[Age,Outcome]+=1
    
fraction_cat=sum_df.div(sum_df.sum(axis=1),axis=0)
#Cat's age v.s outcome
plt.figure(figsize=(16,4))
plt.title('Cat''s age v.s outcome')
plt.xlabel('Age')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(len(unique_age)),fraction_cat.ix[:,'Return_to_owner'],color='brown')
cat_bottom1=fraction_cat.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(len(unique_age)),fraction_cat.ix[:,'Adoption'],bottom=cat_bottom1,color='orange')
cat_bottom2=cat_bottom1+fraction_cat.ix[:,'Adoption']
bar3=plt.bar(np.arange(len(unique_age)),fraction_cat.ix[:,'Transfer'],bottom=cat_bottom2,color='green')
cat_bottom3=cat_bottom2+fraction_cat.ix[:,'Transfer']
bar4=plt.bar(np.arange(len(unique_age)),fraction_cat.ix[:,'Euthanasia'],bottom=cat_bottom3,color='white')
cat_bottom4=cat_bottom3+fraction_cat.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(len(unique_age)),fraction_cat.ix[:,'Died'],bottom=cat_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(len(unique_age)),fraction_cat.index,rotation=40)

plt.show()
#Dog's age v.s outcome
plt.figure(figsize=(16,4))
plt.title('Dog''s age v.s outcome')
plt.xlabel('Age')
plt.ylabel('outcome fraction')
bar1=plt.bar(np.arange(len(unique_age)),fraction_dog.ix[:,'Return_to_owner'],color='brown')
dog_bottom1=fraction_dog.ix[:,'Return_to_owner']
bar2=plt.bar(np.arange(len(unique_age)),fraction_dog.ix[:,'Adoption'],bottom=dog_bottom1,color='orange')
dog_bottom2=dog_bottom1+fraction_dog.ix[:,'Adoption']
bar3=plt.bar(np.arange(len(unique_age)),fraction_dog.ix[:,'Transfer'],bottom=dog_bottom2,color='green')
dog_bottom3=dog_bottom2+fraction_dog.ix[:,'Transfer']
bar4=plt.bar(np.arange(len(unique_age)),fraction_dog.ix[:,'Euthanasia'],bottom=dog_bottom3,color='white')
dog_bottom4=dog_bottom3+fraction_dog.ix[:,'Euthanasia']
bar5=plt.bar(np.arange(len(unique_age)),fraction_dog.ix[:,'Died'],bottom=dog_bottom4,color='blue')
plt.legend([bar1,bar2,bar3,bar4,bar5],['Return_to_owner','Adoption','Transfer','Euthanasia','Died'])
plt.xticks(np.arange(len(unique_age)),fraction_dog.index,rotation=20)

plt.show()