
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/lifestyle-and-wellbeing-data/Wellbeing_and_lifestyle_data.csv')

data=data[data.DAILY_STRESS !='1/1/00']
data['DAILY_STRESS']=pd.to_numeric(data['DAILY_STRESS'])
data.columns=[x[0]+x[1:].lower()for x in data.columns]
data
# A pie chart to have an overview of the percentage of respondents for the different number of  hours of daily stress (0-5)
stress_data=(data['Daily_stress'].value_counts()*100)/(data.shape[0])


explode = (0, 0, 0.2, 0, 0.25, 0) 
plt.pie(stress_data, labels=stress_data.index,explode=explode, autopct='%1.1f%%',shadow=False, startangle=90,textprops={'fontsize': 15})
plt.title('Percentage or repspondents per number of hours of daily stress ',bbox={'facecolor':'0.9', 'pad':5})
plt.axis('equal')


def plotsns(data,param,x_lab,y_lab):
    '''takes in the data frame, the parameter to plot and the labels of the axis and return the plot'''
    sns.set()
    fig=data.set_index(param).T.plot(kind='bar', stacked=True,colormap=ListedColormap(sns.color_palette("GnBu", 10)),figsize=(10,10))
    sns.set_style("whitegrid", {'axes.grid' : False})
    fig.set_ylabel(y_lab, fontsize=18)
    fig.set_xlabel(x_lab, fontsize=18)
    fig.tick_params(rotation= 45,labelsize=20)

    
def percentage(data_p):
    '''takes in the data frame and return a data frame with all the values converted to their percentage per column'''
    summ=data_p.sum(axis=0)
    data_p=data_p*100/summ
    return data_p
# prepare a numeric data frame (data_num) containing the attributes that will be plotted in function of the age range (data_age )

data_age=pd.get_dummies(data['Age'])

data_num=pd.DataFrame(data.drop(['Timestamp','Gender','Age'],axis=1))
# To compare the 22 attributes for the 4 age ranges,a for loop is used to generate a plot for each one of the numeric attributes  in 
# function of the age ranges.

data_perc_per_age=[]

for i,col in enumerate(data_num):
    col=str(col)
    data_age_i=pd.concat([data_num[col],data_age],axis=1).groupby(col).sum()
    data_age_i.index.name=col
    data_age_i=percentage(data_age_i)
    
    data_perc_per_age.append(data_age_i)
    
for df in (data_perc_per_age):
    plotsns(df,df.index, 'Age','Percentage')

# Calculate for each one of the age ranges the percentage of respondents for the different numbers of daily hours of  stress
data_age=pd.concat([data['Daily_stress'],pd.get_dummies(data['Age'])],axis=1).groupby('Daily_stress').sum()
data_age.index.name='Daily_stress'
data_age=percentage(data_age)

data_age
data_gender=pd.concat([data['Daily_stress'],pd.get_dummies(data['Gender'])],axis=1).groupby('Daily_stress').sum()
data_gender=percentage(data_gender)

plotsns(data_gender,data_gender.index,'Gender','Percentage')

corrmat = data.corr() 
map_corr= sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1)
plt.setp(map_corr.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
map_corr.ax_heatmap.set_xticklabels(map_corr.ax_heatmap.get_xmajorticklabels(), fontsize = 16)
map_corr.ax_heatmap.set_yticklabels(map_corr.ax_heatmap.get_ymajorticklabels(), fontsize = 16)

map_corr
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
# Generate a data frame  with the correlation between the attributes and select a data frame (corr_stress) having only the correclation to
#daily stress

corr_stress=pd.DataFrame(data.corr(method='pearson').loc['Daily_stress',:])

corr_stress.sort_values(by='Daily_stress', inplace=True)
corr_stress.drop(['Daily_stress'],inplace=True)

sns.set()
f, ax = plt.subplots(figsize=(15,5))
plt.bar(corr_stress.index,corr_stress['Daily_stress'])
plt.xticks(rotation=45,fontsize=20, horizontalalignment="right")
