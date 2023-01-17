import pandas as pd
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
import glob as glob # This was used for reading the directory from python file handling , however we were able to get the data without it 

names1880 = pd.read_csv('../input/1-name-data/yob1880.txt',names=['name', 'sex', 'births'])

names1880.set_index('births')
a= (names1880['name']=='Anna')
print(a)

#iloc works as a rows:columns
names1880.iloc[[1],0:3] 


#To see the details from the data frame
names1880[a]
names1880.head()
grouped_multiple_column = names1880.groupby(['name','sex'])['sex'].count()
grouped_multiple_column
names1880.value_counts('sex')

# combine datasets for all years
years = range(1880, 2020)

pieces = []
columns = ['Name', 'Sex', 'Births']


for year in years:
    path = '../input/1-name-data/yob' + str(year)+ str('.txt') 
    frame = pd.read_csv(path, names=columns)
    frame['Year'] = year
    pieces.append(frame)

    
df = pd.concat(pieces, ignore_index=True)
df
df.head(15)
plt.figure(figsize=(5,5))
sns.heatmap(df.isnull())
df_viz = df[df['Year']== 1990]
#figsize=(10,10)
#df_viz.plot(kind='bar',x= 'Year',y='Births',color='blue')
total_birth_counts = df['Births'].sum()
print (f"Total Pouplation is increased {total_birth_counts}")



grouped_multiple_column = df.groupby(['Year'])['Births'].sum()
grouped_multiple_column
fig, ax = plt.subplots(figsize=(20,7))
df.groupby(['Year'])['Births'].sum().plot(ax=ax,kind='bar' )

#Dividing the data in female and male
df_f = df[df['Sex']=='F']
df_m = df[df['Sex']=='M']
df_m.head()
df_f.head()

#set index to reduce the computation time 
df_f.set_index('Name', inplace = True)
df_m.set_index('Name', inplace = True)
#print(df_f['Births'].sum())
fig, ax = plt.subplots(figsize=(20,7))
plt.ylim(0,3000000)
(df_f.groupby(['Year'])['Births'].sum()).plot(ax=ax,kind='bar')



fig, ax = plt.subplots(figsize=(20,7))
plt.ylim(0,5000000)
(df_m.groupby(['Year'])['Births'].sum()).plot(ax=ax,kind='bar' )

#Popular name rank in females
df_f.sort_values(by='Births',ascending=False).head(20)

#Popular name rank in males
df_m.sort_values(by='Births',ascending=False).head(20)
#year wise popular names Male
#fig, ax = plt.subplots(figsize=(20,7))
#ax.set_xlabel('Year')
#ax.set_ylabel('Bith Count')
#ax.set_xticklabels('Name')
#plt.ylim(0,150000)
#(df_m.sort_values(by='Births',ascending=False).head(20)).plot(ax=ax,kind='bar' )


#fig, ax = plt.subplots(figsize=(20,7))
#plt.ylim(0,5000000)
#(df_m.groupby(['Year'])['Births'].sum()).plot(ax=ax,kind='bar' )

#extensive CPU task 
#Top 100 pupolar names 

#Top_100_popular_male _name  = df_m.sort_values(by='Births',ascending=False).sum().head(100)
#extensive CPU task 
#Top 100 pupolar names 

#Top_100_popular_female _name  = df_f.sort_values(by='Births',ascending=False).sum().head(100)
# girls to boy ratio 

grouped_multiple_column = (df.groupby(['Year','Sex'])['Births'].sum())
grouped_multiple_column




grouped_multiple_column = df.groupby(['Year','Sex'])['Births'].sum().unstack(level=1)
grouped_multiple_column
grouped_multiple_column['ratio'] = grouped_multiple_column['F'] / grouped_multiple_column['M'] 
grouped_multiple_column
#Most famous name over decade

#Names that faded over time
#current popular names
#Analyse baby names by sorting out all birth counts.











