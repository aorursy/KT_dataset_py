import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LinearRegression
#Let's read the data
df = pd.read_csv('../input/inc_occ_gender.csv', na_values = 'Na')
df.head()
df.shape
df.count()
df.isnull().sum()
#Need to drop all NaN values since they will contribute nothing to the analysis
df = df.dropna().reset_index(drop=True) #Drop = True term drops the old index column and replaces it with a new one
df.head()
df.isnull().sum()
#No more null values!
#We should also have a different size data frame, obviously
df.shape
#Now I'm going to make a dataframe with the specific job sectors, the ones in all caps

sectors = []

for i in range(df.count()['Occupation']):
    x = df['Occupation'][i]
    if x.isupper():
        sectors.append(x)
   
sectors
#Creating our new data frame with the sectors list and importing all data associated with each sector

data = []

for i in range(df.count()['Occupation']):
    if df['Occupation'][i] in sectors:
        data.append(df.loc[i])

dfsectors = pd.DataFrame(data, columns = df.columns)
dfsectors.reset_index(drop = True)
#Data Visualization: Plots of Male and Female Median Weekly Earnings

m_weekly_plot = sns.barplot(x = "M_weekly", y = "Occupation", data = dfsectors.sort_values("M_weekly", ascending = False))
m_weekly_plot.set(xlim = (0,2000), xlabel = "Male Median Weekly Earnings in USD", ylabel = "Occupation" )


f_weekly_plot = sns.barplot(x = "F_weekly", y = "Occupation", data = dfsectors.sort_values("F_weekly", ascending = False))

f_weekly_plot.set(xlim = (0,2000), xlabel = "Female Median Weekly Earnings in USD", ylabel = "Occupation" )
dfsectors['F/M_weekly'] = dfsectors['F_weekly']/dfsectors['M_weekly']
dfsectors.reset_index(drop = True)
ratio_plot = sns.barplot(x = "F/M_weekly", y = "Occupation", data = dfsectors.sort_values("F/M_weekly", ascending = False))
ratio_plot.set(xlim = (0,1), xlabel = "Ratio of Female to Male Median Weekly Earnings in USD", ylabel = "Occupation")
#So in the most general sense, there seems to be a pay gap, even though number of working hours was not factored into the data.

#Next I would like to look at specific jobs where women make up the overwhelming majority of workers, source is BLS again.

#I will search for 9 occupations from the BLS where women are 90% of the workforce. I'm hoping for the occupations to be in the data and for them to be labled as they are on the BLS website.
for i in range(df.count()['Occupation']):
    if df['Occupation'][i] == 'Registered nurses':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Secretaries and administrative assistants':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Nursing, psychiatric, and home health aides':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Receptionists and information clerks':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Maids and housekeeping cleaners':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Bookkeeping, accounting, and auditing clerks':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Teacher assistants':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Preschool and kindergarten teachers':
        print(i, df['Occupation'][i])
    elif df['Occupation'][i] == 'Licensed practical and lisenced vocational nurses':
        print(i, df['Occupation'][i])
# it appears we don't have all that we searched for. We can just use the 6 we found.
womendata = [df.loc[65], df.loc[70], df.loc[86], df.loc[106], df.loc[109], df.loc[116]]

dfwomen = pd.DataFrame(womendata, columns = df.columns)
dfwomen.reset_index(drop = True)


dfwomen['F/M_weekly'] = df['F_weekly']/df['M_weekly']
dfwomen.reset_index(drop = True)
F_ratio_plot = sns.barplot(x = "F/M_weekly", y = "Occupation", data = dfwomen.sort_values("F/M_weekly", ascending = False))
F_ratio_plot.set(xlim = (0,1.3), xlabel = "Ratio of Female to Male Median Weekly Earnings in USD", ylabel = "Female Dominated Occupations (>90%)")
#Let's add a F/M column to our original data set to see possibly where else women may make more

df['F/M_weekly'] = df['F_weekly']/df['M_weekly']
df.head()
for i in range(df.count()['F/M_weekly']):
    if df['F/M_weekly'][i] >=1:
        print(i, df['Occupation'][i])
#Creating a new dataframe to contain the information presented above
Womendata = [df.loc[19], df.loc[73], df.loc[106], df.loc[117], df.loc[119]]

dfWomen = pd.DataFrame(Womendata, columns = df.columns)
dfWomen.reset_index(drop = True)
#Adding a column to get the ratio of female to male workers in said fields

dfWomen['F/M_workers'] = dfWomen['F_workers']/dfWomen['M_workers']
dfWomen.reset_index(drop = True)
#The really interesting one above is Police officers, where women only make up about 13% of the workforce yet outearn men

#Another thing we can do is to see what specific jobs have the largest wage gaps. Let's also ad a F/M workers column to df

df['F/M_workers'] = df['F_workers']/df['M_workers']
df.head()
largegapdata=[]

for i in range(df.count()['F/M_weekly']):
    if df['F/M_weekly'][i] <= .75 and df['Occupation'][i] not in sectors:
        largegapdata.append(df.loc[i])
dflarge = pd.DataFrame(largegapdata, columns = df.columns)
dflarge.reset_index(drop = True)
dflarge.head()
large_ratio_plot = sns.barplot(x = "F/M_weekly", y = "Occupation", data = dflarge.sort_values("F/M_weekly", ascending = False))
large_ratio_plot.set(xlim = (0,1), xlabel = "Ratio of Female to Male Median Weekly Earnings in USD", ylabel = "Occupation")
#The last thing I want to do is to see what industries females outnumber males but yet are outearned by males

df.head()
totalgapdata = []

for i in range(df.count()['F/M_workers']):
    if df['F/M_workers'][i] > 1 and df['F/M_weekly'][i] < 1 and df['Occupation'][i] not in sectors:
        totalgapdata.append(df.loc[i])
        
dftotalgap = pd.DataFrame(totalgapdata, columns = df.columns)
dftotalgap.reset_index(drop = True)
dftotalgap.shape
#There are 54 professions where women outnumber men but yet are outearned by men
