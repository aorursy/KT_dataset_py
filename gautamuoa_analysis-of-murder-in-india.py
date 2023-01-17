#import the neccesary libraries
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read the dataset
victim = pd.read_csv('../input/VICTIM_OF_MURDER_0.csv')

#rename columns
victim.rename(columns={'STATE/UT':'State','YEAR':'Year',
                       'GENDER':'Gender',
                       'Upto 10 years':'Age < 10',
                       '10-15 years':'10 < Age < 15',
                       '15-18 years':'15 < Age < 18',
                       '18-30 years':'18 < Age < 30',
                       '30-50 years':'30 < Age < 50',
                       'Above 50 years':'Age > 50'
                       },inplace=True)

#get information about missing datapoints
print('Missing Details of the Dataset')
print('_'*50)
print(victim.isnull().sum())
print('_'*50)
#method for getting count value on top of the graphs
def assignVerticalAxis(df,g,value=1000):
    # Get current axis on current figure  .g.axes is array which has attribute size 
    
    for i in range(0,g.axes.size):
        ax = g.fig.get_axes()[i]
        displayVerticalCount(df,ax,value)
        
def displayVerticalCount(df,ax,value):
    # ylim max value to be set
    y_max = df.max() + value 
    ax.set_ylim(top=y_max)
    
    # Iterate through the list of axes' patches
    for p in ax.patches:
        #checks if there index has 0 count        
        if(math.isnan(p.get_height())):
            ax.text(p.get_x() + p.get_width()/2, 0, 0, 
                fontsize=12, color='Black', ha='center', va='bottom',rotation=90)
            continue
        else:            
            ax.text(p.get_x() + p.get_width()/2, p.get_height(), int(p.get_height()), 
                fontsize=12, color='Black', ha='center', va='bottom',rotation=90)
#display murder in states of India over the year

for i in range(len(victim['Year'].value_counts().index)):
    temp = victim.drop(victim[victim['Gender'] == 'Male'].index)
    temp = temp.drop(temp[temp['Gender'] == 'Female'].index)
    temp = temp.drop(temp[temp['Year'] != temp['Year'].value_counts().index[i]].index)
    temp = temp.sort_values(by=['Total'], ascending=False)
    title = 'Murders in Different States for the year ' + str(temp['Year'].values[0])
    g = sns.factorplot(x='State',y='Total',data=temp,kind='bar',
                       palette=sns.color_palette('Reds_r',34),  
                       size=5,aspect=2).set(
                               title=title)
    assignVerticalAxis(temp['Total'],g)
    plt.xticks(rotation=90)
   
temp = victim.drop(victim[victim['State'] != 'Bihar'].index)

sns.set(context='notebook',style='whitegrid',font_scale=1.3)

aa = temp.drop(temp[temp['Gender'] == 'Female'].index)
aa = aa.drop(aa[aa['Gender'] == 'Male'].index)

g = sns.factorplot(x='Year',y='Total',data=aa,               
               size=5,aspect=2).set(title='Murders in Bihar over the years')
assignVerticalAxis(aa['Total'],g,500)

temp = victim.drop(victim[victim['State'] != 'Bihar'].index)
temp = temp.drop(temp[temp['Gender'] == 'Total'].index)

g = sns.factorplot(x='Year',y='Total',data=temp,kind='bar',hue='Gender',               
               size=5,aspect=2).set(title='Murders of Male and Female Population of Bihar over the years')
assignVerticalAxis(temp['Total'],g)

g = sns.factorplot(x='Year',y='Age < 10',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with Age < 10 years',ylabel='Count')
assignVerticalAxis(temp['Age < 10'],g,10)

g = sns.factorplot(x='Year',y='10 < Age < 15',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with 10 < Age < 15 years',ylabel='Count')
assignVerticalAxis(temp['10 < Age < 15'],g,10)

g = sns.factorplot(x='Year',y='15 < Age < 18',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with 15 < Age < 18 years',ylabel='Count')
assignVerticalAxis(temp['15 < Age < 18'],g,50)

g = sns.factorplot(x='Year',y='18 < Age < 30',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with 18 < Age < 30',ylabel='Count')
assignVerticalAxis(temp['18 < Age < 30'],g,500)

g = sns.factorplot(x='Year',y='30 < Age < 50',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with 30 < Age < 50 years',ylabel='Count')
assignVerticalAxis(temp['30 < Age < 50'],g,500)

g = sns.factorplot(x='Year',y='Age > 50',kind='bar',data=temp,hue='Gender',               
               size=5,aspect=2).set(title='Murder of Male and Female Population of Bihar with Age > 50 years',ylabel='Count')
assignVerticalAxis(temp['Age > 50'],g,50)

