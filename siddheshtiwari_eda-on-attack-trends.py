# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # hhta processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import calendar
%matplotlib inline
# Input data files are available the "../input/" directory.
# For example, runnihg this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
globalTerror = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1',index_col=0)
globalTerror.head()
#Remove rows with day and month as 0 as their count is very low
print(globalTerror.loc[globalTerror['imonth'] == 0,'imonth'].value_counts())
print(globalTerror.loc[globalTerror['iday'] == 0,'iday'].value_counts())
globalTerror = globalTerror[~((globalTerror['iday']==0) | (globalTerror['imonth']==0))]

#Create Date,Day and Month
globalTerror.loc[:,'Date'] =  pd.to_datetime(globalTerror['iday'].map(str)+'-'+globalTerror['imonth'].map(str)+'-'+globalTerror['iyear'].map(str),format='%d-%m-%Y')
globalTerror.loc[:,'DayOfWeek'] = globalTerror.Date.dt.weekday_name
globalTerror.loc[:,'MonthName'] = [calendar.month_name[i] for i in globalTerror.Date.dt.month]#globalTerror_SUB1.loc[:,'MonthName'] = globalTerror_SUB1.Date.dt.month.apply(lambda x:calendar.month_name[x])
globalTerror.head()

missing_counts = (globalTerror.isnull().sum()/len(globalTerror)).reset_index().\
                    rename(columns = {'index':'Columns',0:'Missing Percent'}).\
                    sort_values(by ='Missing Percent',ascending=False)
missing_counts = missing_counts.loc[missing_counts['Missing Percent']>0,:]
plt.subplots(figsize=(15,20))
sns.barplot(data=missing_counts,x='Missing Percent',y='Columns')
globalTerror.shape
pd.options.display.max_columns = 30
globalTerror_SUB1  = globalTerror[['iyear', 'MonthName','DayOfWeek', 'approxdate', 'extended', 'resolution', 'country_txt', 
                                   'region_txt','provstate', 'city', 'latitude', 'longitude', 'specificity', 'vicinity', 'location', 'summary', 'crit1',
                                   'crit2', 'crit3', 'doubtterr', 'alternative', 'alternative_txt', 'multiple','related']]


month_type = pd.api.types.CategoricalDtype(categories=['January','February','March','April','May','June','July','August','September','October','November','December'],ordered=True)
month_type_index = pd.CategoricalIndex(['January','February','March','April','May','June','July','August','September','October','November','December'],
                    categories = ['January','February','March','April','May','June','July','August','September','October','November','December'],
                    ordered=True)
day_type = pd.api.types.CategoricalDtype(categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],ordered=True)

globalTerror_SUB1.loc[:,'MonthName'] = globalTerror_SUB1.MonthName.astype(month_type)
globalTerror_SUB1.loc[:,'DayOfWeek'] = globalTerror_SUB1.DayOfWeek.astype(day_type)

#Remove Columns with mostly missing values and assign missing value identifier in other columns
globalTerror_SUB1 = globalTerror_SUB1.drop(columns=['resolution','approxdate','location','vicinity','alternative'])
globalTerror_SUB1.loc[:,'provstate'] = globalTerror_SUB1.provstate.fillna(value='Unknown')
globalTerror_SUB1.loc[:,'city'] = globalTerror_SUB1.city.fillna(value='Unknown')
globalTerror_SUB1.loc[globalTerror_SUB1.city == 'unknown','city']='Unknown'
globalTerror_SUB1.loc[:,'summary'] = globalTerror_SUB1.summary.fillna(value='NotAvailable')
globalTerror_SUB1.loc[:,'alternative_txt'] = globalTerror_SUB1.alternative_txt.fillna(value='NoTerrorDoubt')
globalTerror_SUB1.loc[:,'related'] = globalTerror_SUB1.related.fillna(value='StandAloneEvent')
globalTerror_SUB1.loc[:,'specificity'] = globalTerror_SUB1.specificity.fillna(value=globalTerror_SUB1.specificity.value_counts().idxmax())
#globalTerror_SUB1.isnull().sum().reset_index()
globalTerror_SUB1.head()
# Function to get heat map for crosstabs
def get_crostab_heat_hist(data,col1,col2,filter_query = None,likelihood = False,sum_stat = True):
    if filter_query != None:
        data = data.query(filter_query)

    if likelihood == True:
        crosstab = (pd.crosstab(data.loc[:,col1],data.loc[:,col2]))
        crosstab = crosstab.divide(crosstab.sum(axis=1),axis=0)
    else:
        crosstab = pd.crosstab(data.loc[:,col1],data.loc[:,col2])
    flatten_d = crosstab.values.ravel()
    if sum_stat == True:
        print("Mean: %s, Median: %s, Std Dev: %s ,Max: %s, Min: %s"%( np.mean(flatten_d),np.median(flatten_d),np.std(flatten_d),np.max(flatten_d),np.min(flatten_d)))
    cmap = sns.cm.rocket_r
    
    plt.subplots(figsize = (30,10))
    plt.subplot(1,2,1)
    if likelihood == True:
        sns.heatmap(crosstab,annot=True,cmap=cmap)#,fmt='g')
        plt.title("%s vs %s"%(col1,col2))
    else:
        sns.heatmap(crosstab,annot=True,cmap=cmap,fmt='d')
        plt.title("%s vs %s"%(col1,col2))
    plt.subplot(1,2,2)
    sns.distplot(flatten_d)
    plt.title("Distribution of Values")
    
    if likelihood == True:
        plt.suptitle("Cross Tab with Likelihood Of Attack Occurence with Ref to: %s, Filter: %s"%(col1,filter_query), fontsize = 20)
    else:
        plt.suptitle("Cross Tab With Frequency, Filter: %s"%(filter_query), fontsize = 20)
    
def get_count_plot(data,col,title,hue=None,filter_query = None,order = False,show=True):
    if filter_query != None:
        data = data.query(filter_query)
    plt.subplots(figsize = (15,10))
    if order == True:
        sns.countplot(col,data = data,hue = hue,order=data[col].value_counts().index)
    else:
        sns.countplot(col,data = data,hue = hue)
    plt.xticks(rotation = 90, fontsize = 15)
    plt.xlabel(col,fontsize = 15)
    title = title+"-"+str(filter_query)
    plt.title(title,fontsize = 15)
    if show == True:
        plt.show()


sns.set_style("white")
filter = None
get_count_plot(globalTerror_SUB1,"iyear",title="Terrorist Attack by Year",filter_query=filter)
# Attacks have increased significantly since 2013
get_count_plot(globalTerror_SUB1,"MonthName",title="Terrorist Attack by Month",filter_query=filter,order = True)
get_count_plot(globalTerror_SUB1,"DayOfWeek",title="Terrorist Attack by Week Day",filter_query=filter,order = True)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = None,order = True)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = "iyear >= 1970 & iyear < 1980 & doubtterr == 0",order = True,show=False)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = "iyear >= 1980 & iyear < 1990 & doubtterr == 0",order = True,show=False)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = "iyear >= 1990 & iyear < 2000 & doubtterr == 0",order = True,show=False)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = "iyear >= 2000 & iyear < 2010 & doubtterr == 0",order = True,show=False)
get_count_plot(globalTerror_SUB1.loc[(globalTerror_SUB1['country_txt'].\
                                     isin(globalTerror_SUB1.country_txt.\
                                          value_counts().iloc[:30].index)),:]\
                                    #& (globalTerror_SUB1['iyear'] >= 2010),:]\
               ,'country_txt',title="Terrorist Attack by Country",filter_query = "iyear >= 2010 & doubtterr == 0",order = True,show=False)
plt.tight_layout()
#Countries with high attacks
#70s - UK,US
#80s - Latin America
#90s - Latin America,India,Uk,Turkey
#00s - Iraq,India,Pak,Afgan
#10s - Iraq,India,Pak,Afgan

#plt.savefig("output.png",dpi =300)

#X = plt.imread("output.png")[:,:,:3]
#plt.subplots(figsize = (20,20))
#plt.imshow(X)
globalTerror_SUB1.query("iyear >= 1970 & iyear<1980 & doubtterr == 0 & country_txt == 'United States'").summary.values
filter = None
get_crostab_heat_hist(data = globalTerror_SUB1,col1='iyear',col2 = 'MonthName',likelihood=False,filter_query=filter)
get_crostab_heat_hist(data = globalTerror_SUB1,col1='iyear',col2 = 'MonthName',likelihood=True,filter_query=filter)

for i in globalTerror_SUB1.region_txt.unique():
    filter = "region_txt == '%s'"%(i)
    get_crostab_heat_hist(data = globalTerror_SUB1,col1='iyear',col2 = 'MonthName',likelihood=False,filter_query=filter,sum_stat=False)

filter = None
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=False,filter_query=filter)
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=True,filter_query=filter)
# bi modal distrbution is indicatin there are two populationn here which we can see in heatmap, population with higher likehood of attac Mon-Thus, 
#Population with low liklihood Fri-Sat, but recently this difference seems to go away these trends differ by country and year
filter = "iyear < 2013"# & country_txt == 'India'"
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=False,filter_query=filter)
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=True,filter_query=filter)

filter = "iyear >= 2013"#& country_txt == 'India'"
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=False,filter_query=filter)
get_crostab_heat_hist(data = globalTerror_SUB1,col1='MonthName',col2 = 'DayOfWeek',likelihood=True,filter_query=filter)


#Comment on charts




