#this is a practice for data analysis

#tell me if there's any issue



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting the graphs

import os



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/crime.csv",encoding="ISO-8859-1") #reading crime.csv in df

codes_df=pd.read_csv("../input/offense_codes.csv",encoding="ISO-8859-1") #reading offense_codes.csv in codes_df
df.info() #infor of crimes.csv file
# function to find PMF (Probability Mass Function)

# ref:https://stackoverflow.com/questions/25273415/how-to-plot-a-pmf-of-a-sample

def pmf(dat):

    return dat.value_counts().sort_index()/len(dat)

# function to find CDF (cumulative distribution function)

# ref: I could'nt find any python method to find cdf so i created my own tell me if there's some issue

def pmf_to_cdf(pmf):

    cdf=np.array([pmf.iloc[0]])

    for i in range(1,len(pmf)):

        cdf=np.append(cdf,cdf[i-1]+pmf.iloc[i])

    cdf=pd.DataFrame(data=cdf,index=pmf.index)

    return cdf
year_pmf=pmf(df["YEAR"]) #pmf of years
#pmf plot of years or simply percentage of crimes in each year

fig=plt.figure(figsize=(5,5))

plt.pie(year_pmf,labels=year_pmf.index,autopct='%1.0f%%')

plt.suptitle('PMF % of Crimes from 2015-2018')
year_cdf=pmf_to_cdf(year_pmf) ##cdf of years
#cdf plot of years or simply increase in percentage of crimes till each year

fig=plt.figure(figsize=(5,5))

plt.plot(year_cdf)

plt.suptitle('CDF % of Crimes from 2015-2018')

plt.xlabel('Years')

plt.ylabel('CDF')

plt.xticks(np.arange(2015,2019))
hours_pmf=pmf(df["HOUR"]) #pmf of hours
#pmf plot of hour or simply percentage of crimes in each hour

fig=plt.figure(figsize=(5,5))

plt.scatter(hours_pmf.index,hours_pmf.values)

plt.suptitle('PMF % of Crimes in HOURS')

plt.xlabel('Hours')

plt.ylabel('PMF')

plt.xticks(np.arange(0,24,2))
hours_cdf=pmf_to_cdf(hours_pmf) #cdf of hours
#cdf plot of years or simply increase in percentage of crimes till each hour

fig=plt.figure(figsize=(5,5))

plt.plot(hours_cdf,label="CDF")

plt.suptitle('CDF % of CRIMES IN HOURS')

plt.xlabel('Hours')

plt.ylabel('% of Crimes')

plt.xticks(np.arange(0,24,2))

plt.show()
##pmf plot of each crime in each hour

group_df=df.groupby(by="OFFENSE_CODE")

for x in group_df.groups:

    fig=plt.figure(figsize=(10,5))

    temp=group_df.get_group(x)["HOUR"]

    if(len(temp)==1):

        continue

    temp_pmf=pmf(temp)

    plt.scatter(temp_pmf.index,temp_pmf.values) 

    plt.suptitle('PMF % of '+str(codes_df[codes_df["CODE"]==x]["NAME"])+'-Hours')

    plt.xlabel('Hours')

    plt.ylabel('% of Crimes')

    #plt.legend()

    plt.xticks(np.arange(0,24))

    plt.show()

    
##cdf plot of each crime till each hour

group_df=df.groupby(by="OFFENSE_CODE")

for x in group_df.groups:

    fig=plt.figure(figsize=(10,5))

    temp=group_df.get_group(x)["HOUR"]

    if(len(temp)==1):

        continue

    temp_pmf=pmf(temp)

    temp_cdf=pmf_to_cdf(temp_pmf)

    plt.plot(temp_cdf)

    plt.suptitle('CDF % of '+str(codes_df[codes_df["CODE"]==x]["NAME"])+'-Hours CDF Summary')

    plt.xlabel('Hours')

    plt.ylabel('% of Crimes')

    #plt.legend()

    plt.xticks(np.arange(0,24))

    plt.show()

    