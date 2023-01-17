import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
honey = pd.read_csv('../input/honeyproduction.csv')
honey.dtypes
honey.info()
honey.describe().round()
honey.head(10)
plt.figure(figsize=(40,15))
sns.barplot(x=honey['state'],y = honey['totalprod'])
plt.title('Statewise Total Honey production in USA',fontsize =30)
plt.xlabel('States',fontsize=30)
plt.ylabel('Total Production of Honey in USA',fontsize=30)
honey_byYear = honey[['totalprod','year']].groupby('year').sum()
honey_byYear.reset_index(level=0, inplace=True)
plt.figure(figsize=(30,10))
plt.plot(honey_byYear['year'],honey_byYear['totalprod'])
plt.title('Total Production Of Honey in USA (lbs.) Vs Year',fontsize=30)
plt.xlabel('Year',fontsize = 20)
plt.ylabel('Total Production of Honey',fontsize = 20)
honey_byState = honey[['totalprod','state']].groupby('state').sum()
honey_byState.reset_index(level=0, inplace=True)
honey_byState.sort_values(by='totalprod',ascending=False,inplace=True)
plt.figure(figsize=(20,10))
sns.barplot(x=honey_byState['state'],y=honey_byState['totalprod'])
plt.title('Statewise total honey production',fontsize = 20)
plt.xlabel('State',fontsize = 20)
plt.ylabel('Total Production(lbs)',fontsize = 20)
def percentage_change(dataframe,column,y1,y2):
    '''Creates 2 dataframes with rows only having the given year values. Then these 2 frames are merged to a single frame
    with respect to the states and returned'''
    honey_change_yearA = dataframe[['state',column,'year']].loc[dataframe['year']==y1].sort_values('state')
    honey_change_yearB = dataframe[['state',column,'year']].loc[dataframe['year']==y2].sort_values('state')
    honey_yearAyearB = pd.merge(honey_change_yearA,honey_change_yearB,on=['state','state']).drop('year_x',axis=1).drop('year_y',axis=1)    
    honey_yearAyearB['percentage_change'] = (honey_yearAyearB.iloc[:,2]-honey_yearAyearB.iloc[:,1])/honey_yearAyearB.iloc[:,1] * 100
    return honey_yearAyearB

def percentage_plot(plot_parameterList,fig_size=(10,15)):
    '''Creates 'n' Bargrph subplots.
    The plot_parameterList is a list of lists that has the following arguments
    0 -> Dataframe
    1 -> Quantity for the bar graphs over which the percentage change has been meeasured using percentage_change method.
    2 -> Start year of the time range
    3 -> End year of the time range
    
    And additional parameter to change the figure size is included.
    Best for maximum of 3x2 plots'''
    sns.set(rc={'figure.figsize':fig_size})
    sns.set(font_scale=0.9)
    import math
    total_plots = len(plot_parameterList)
    plotRows = math.ceil(total_plots/2)
    if total_plots > 1:
        fig, ax = plt.subplots(plotRows,2)
        ax = ax.flatten()
        for x in range(total_plots):
                sns.barplot(x = plot_parameterList[x][0]['percentage_change'], y = plot_parameterList[x][0]['state'], ax = ax[x])
                ax[x].title.set_text('PercentageChange : ' + plot_parameterList[x][1] +' years>'+str(plot_parameterList[x][2])+'-'+str(plot_parameterList[x][3]))
                ax[x].set_xlabel('Percentage Change')
                ax[x].set_ylabel('State')        
    else:
        sns.barplot(x = plot_parameterList[0][0]['percentage_change'], y = plot_parameterList[0][0]['state'])
        plt.title('PercentageChange : ' + plot_parameterList[0][1] +' years>'+str(plot_parameterList[0][2])+'-'+str(plot_parameterList[0][3]))
        plt.xlabel('Percentage Change')
        plt.ylabel('State')

column_name,start_year,end_year = ['totalprod',2004,2005]
column_name,start_year1,end_year1 = ['totalprod',2005,2006]
column_name,start_year2,end_year2 = ['totalprod',2006,2007]
column_name,start_year3,end_year3 = ['totalprod',2007,2008]
honey_yearAyearB_production = percentage_change(honey,column_name,start_year,end_year)
honey_yearAyearB_production1 = percentage_change(honey,column_name,start_year1,end_year1)
honey_yearAyearB_production2 = percentage_change(honey,column_name,start_year2,end_year2)
honey_yearAyearB_production3 = percentage_change(honey,column_name,start_year3,end_year3)
percentage_plot([\
                 [honey_yearAyearB_production,column_name,start_year,end_year],\
                 [honey_yearAyearB_production1,column_name,start_year1,end_year1],\
                 [honey_yearAyearB_production2,column_name,start_year2,end_year2],\
                 [honey_yearAyearB_production3,column_name,start_year3,end_year3]\
                ],(10,15))
def scatter_plot(plot_parameterList,fig_size):
    sns.set(rc={'figure.figsize':fig_size})
    sns.set(font_scale=0.9)
    import math
    total_plots = len(plot_parameterList)
    #plotRows = math.ceil(total_plots/2)
    #fig, ax = plt.subplots(plotRows,2)
    #ax = ax.flatten()
    
    for x in range(total_plots):
            sns.lmplot(x = 'yieldpercol', y = 'totalprod',data = plot_parameterList[x][0],fit_reg=False,hue='year')
            plt.title('Year wise -Mean Totalproduction and Num.Colonies scatter plot('+plot_parameterList[x][1]+')')
honey_colprod_singleState = honey[['totalprod','yieldpercol','year','state']].loc[honey['state']=='AL'].drop('state',axis=1)
honey_colprod_USA = honey[['totalprod','yieldpercol','year']].groupby(['year']).mean()
honey_colprod_USA.reset_index(inplace=True)
scatter_plot([[honey_colprod_USA,"USA"],[honey_colprod_singleState,'State-AL']],(10,15))
# For Additional reference
#g = sns.FacetGrid(honey_colprod_USA, col="state",col_wrap = 5)
#g = (g.map(plt.scatter, 'year','numcol').set_axis_labels('xyx','abx'))
honey_priceDemand = honey.groupby('year').mean()
honey_priceDemand.reset_index(level=0,inplace=True)
#honey.loc[honey['state'] == 'ND']
plt.figure(figsize=(20,10))
plt.plot(honey_priceDemand['year'],honey_priceDemand['totalprod'],c='b',marker='o',markersize=12)
plt.plot(honey_priceDemand['year'],honey_priceDemand['prodvalue'],c='g',marker='X',markersize=12)
plt.legend(ncol=2,loc=2,fontsize = 15)
plt.title('Total Production and Production Value of Honey over the years',fontsize = 15)
plt.xlabel('years',fontsize = 15)
plt.ylabel('quantity',fontsize = 15)
plt.figure(figsize=(20,10))
sns.barplot(x='year',y='priceperlb',data=honey_priceDemand)
plt.title('Price of Honey over the years',fontsize = 15)
plt.xlabel('years',fontsize = 15)
plt.ylabel('Price per Pound',fontsize = 15)