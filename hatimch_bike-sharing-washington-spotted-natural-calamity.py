%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics
#df=pd.read_csv('Desktop/day.csv')
df=pd.read_csv('../input/bike-sharing-dataset/day.csv')
df.head()
correlation = df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(correlation, vmax=0.9, square=True)
plt.figure(figsize=(11,5))
sns.barplot('yr','registered',hue='season', data=df,palette='Paired', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented on Registered basis')
plt.title('Number of bikes rented per season')
plt.figure(figsize=(11,5))
sns.barplot('yr','casual',hue='season', data=df,palette='pastel', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented on casual basis')
plt.title('Number of bikes rented per season')
plt.figure(figsize=(11,5))
sns.barplot('yr','cnt',hue='mnth', data=df,palette='Set2', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented ')
plt.title('Number of bikes rented per year in different months')
plt.figure(figsize=(17,10))
ax = sns.violinplot(x="yr", y="cnt", hue="mnth",data=df, palette="Set2", split=False)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented total count ')
plt.title('Number of bikes rented per year in different months on violin plot')
plt.figure(figsize=(11,5))
sns.barplot('yr','cnt',hue='weathersit', data=df,palette='Set2', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented ')
plt.title('Number of bikes rented per year in different months')
plt.figure(figsize=(11,5))
ax =  sns.stripplot("yr", "cnt", "season", data=df,palette="Set2", size=50, marker="D",edgecolor="gray", alpha=.25)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented ')
plt.title('Distribution of total number of bikes rented based on the season')
plt.figure(figsize=(11,5))
ax =  sns.stripplot("yr", "cnt", "weathersit", data=df,palette="Set2", size=50, marker="D",edgecolor="gray", alpha=.25)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented ')
plt.title('Distribution of total number of bikes rented based on the weathersit')
#dividing the frame into year frame of two different year
df_2011=df[:365] #data frame of year 2011
#plotting the graph again
new_2011=df_2011.groupby('mnth').mean()['cnt'].values
print(new_2011)



df_2012=df[366:] #data frame of year 2012
#plotting the graph again
new_2012=df_2012.groupby('mnth').mean()['cnt'].values
print(new_2012)






plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],new_2012 ,label="year 2012")
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],new_2011,color='red',label="year 2011")
plt.xlabel('month')
plt.ylabel('Total number of bikes rented ')
plt.title('LIne chart comparision of rented biikes of year 2011 and 2012')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
september_2012=df[609:639] #data frame of september 2012
october_2012=df[639:670] #data frame of october 2012
november_2012=df[670:700] #data frame of november 2012
december_2012=df[700:] #data frame of december 2012


#print(october_2012)
day_30=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
day_31=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

plt.plot(day_30,september_2012['cnt'] ,label="september,2012")

plt.plot(day_31,october_2012['cnt'] ,label="october,2012",color="red")

plt.plot(day_30,november_2012['cnt'] ,label="november,2012",color="green")

plt.plot(day_31,december_2012['cnt'] ,label="december,2012",color="black")


plt.xlabel('day')
plt.ylabel('Total number of bikes rented ')
plt.title('LIne chart comparision of rented biikes of month september to december year 2012 ')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()