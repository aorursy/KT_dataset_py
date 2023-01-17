# !pip install arxivscraper



# import arxivscraper

# import pandas as pd



# scraper_cond = arxivscraper.Scraper(category='physics:cond-mat', date_from='2012-01-01', date_until='2020-07-01',timeout=10000)

# output_cond = scraper_cond.scrape()

# dfcond = pd.DataFrame(output_cond,columns=cols)



# scraper_astro = arxivscraper.Scraper(category='physics:astro-ph', date_from='2012-01-01', date_until='2020-07-01',timeout=10000)

# output_astro = scraper_astro.scrape()

# dfastro = pd.DataFrame(output_astro,columns=cols)



# cols = ('categories', 'created', 'authors')



# dfcond = pd.DataFrame(output_cond,columns=cols)

# dfcond.to_csv('Data/arxiv_cond_2012_2020.csv',index=False)

# dfastro = pd.DataFrame(output_astro,columns=cols)

# dfastro.to_csv('Data/arxiv_astro_2012_2020.csv',index=False)
!pip install gender_guesser
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import gender_guesser.detector as gender



%matplotlib inline
dfcond = pd.read_csv('../input/arxiv-scrape/arxiv_cond_2012_2020.csv',converters={'authors': eval})

dfastro = pd.read_csv('../input/arxiv-scrape/arxiv_astro_2012_2020.csv',converters={'authors': eval})
dfcond['first_author']=dfcond['authors'].apply(lambda x:x[0].split(' ')[0])

dfastro['first_author']=dfastro['authors'].apply(lambda x:x[0].split(' ')[0])
# First we split the author list and returning the first name of the first and last authors in a new columns

# We then drop the authors column for clarity

dfcond['first_author']=dfcond['authors'].apply(lambda x:x[0].split(' ')[0]) 

dfcond['last_author']=dfcond['authors'].apply(lambda x:x[-1].split(' ')[0])

dfcond.drop('authors',inplace=True,axis=1)



#Drop all the rows for which the author's name is shorter than 3 characters 

#This is done to include punctuation in the initials since the split of the author list is done using a single space

#This passage can be improves as it removes from the dataset also authors with a two-letter first name

dfcond.drop(dfcond[dfcond['first_author'].map(len)<3].index, inplace=True)



#We transform the created columns into a datetime column 

#We also drop all the entries that were created before 2012 but entered the data set because modified after 2012

dfcond['created'] = pd.to_datetime(dfcond['created'],format='%Y-%m-%d')

dfcond.drop(dfcond[dfcond['created'].dt.year<2012].index,inplace=True)



#Finally we reset the dataframe index

dfcond.reset_index(drop=True, inplace=True)
# We capitalize the names as this is required for the gender guesser method to work correctly

dfcond['first_author'] = dfcond['first_author'].apply(lambda x:x.capitalize())

dfcond['last_author'] = dfcond['last_author'].apply(lambda x:x.capitalize())



# We instantiate a gender detector object and run it on the first and last author columns separately

detect = gender.Detector()

dfcond['gender_first']=dfcond['first_author'].map(detect.get_gender)

dfcond['gender_last']=dfcond['last_author'].map(detect.get_gender)



# We then split the dataframe in two to better handle separately the two data sets

# This is useful to more easily handle cases where the gender of the first author is clearly identifyed 

# but not that of the last author, or viceversa. This approach can likely be improved

# We also rename the columns for simplicity 



dfcondfirst = dfcond.loc[:,['created','first_author','gender_first']]

dfcondfirst.rename(columns={'created':'date','first_author':'author','gender_first':'gender'},inplace=True)

dfcondlast = dfcond.loc[:,['created','last_author','gender_last']]

dfcondlast.rename(columns={'created':'date','last_author':'author','gender_last':'gender'},inplace=True)



# Finally we map the results of the gender guesser so that 'mostly male' and 'mostly female' 

# are transformed to 'male' and 'female' respectively. The gender guesser also return 'andy' when the name

# has equal likelihood of being male or female and 'uknown' when no record of the name is found

# We drop all the rows associated to names with these last two tags



dfcondfirst['gender'] = dfcondfirst['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',

                                                       'female':'female'})

dfcondfirst.dropna(inplace=True)



dfcondlast['gender'] = dfcondlast['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',

                                                       'female':'female'})

dfcondlast.dropna(inplace=True)
dfcondlast.info()
plt.figure(figsize = (10,6))



sns.countplot(x=dfcondfirst[dfcondfirst['date'].dt.year<2020]['date'].dt.year,

              data = dfcondfirst[dfcondfirst['date'].dt.year<2020],hue='gender');



plt.xlabel('Year', fontsize=15);

plt.ylabel('Publication Count', fontsize=15);

plt.xticks(size=13);

plt.yticks(size=13);

plt.legend(fontsize=15);
plt.figure(figsize = (10,6))



sns.countplot(x=dfcondlast[dfcondlast['date'].dt.year<2020]['date'].dt.year,

              data = dfcondlast[dfcondlast['date'].dt.year<2020],hue='gender');



plt.xlabel('Year', fontsize=15);

plt.ylabel('Publication Count', fontsize=15);

plt.xticks(size=13);

plt.yticks(size=13);

plt.legend(fontsize=15);
fig, axes = plt.subplots(2,3, figsize=(20,12))

d = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'}



for key,month in d.items():



        sns.countplot(dfcondfirst[dfcondfirst['date'].dt.month==key]['date'].dt.year,

              data = dfcondfirst[dfcondfirst['date'].dt.month==key], hue='gender',ax=axes[int(np.floor((key-1)/3)),

                                                                                          int((key-1)-np.floor((key-1)/3)*3)])

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_title('arXiv publication for the month of %s' % month, size = 15);

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_xlabel('Year', size = 13);

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_ylabel('Publications',size=13);
# We create a small function to create a dataframe for the evolution of the gender publication ratio  

# we sample the data by month



def GenderRatio(df):

    

    ratio = []

    years = df['date'].dt.year.unique()

    months = df['date'].dt.month.unique()



    for year in years:



        try:

            for month in months:



                [a,b] = df[(df['date'].dt.year==year)

                             & (df['date'].dt.month==month)]['gender'].value_counts()

                ratio.append(b/(a+b))

        except:

            break

        

    time = pd.date_range(start=str(months[0])+'/'+str(years[0]),end=str(month)+'/'+str(year),freq='m');



    return pd.DataFrame(ratio,time,columns=['Percentage Female Author'])
# Use function to create the dataframe



GndRtCondFirst = GenderRatio(dfcondfirst)
# Visualization of the female/male first author ratio between 2012 and 2020 



fig = plt.figure(figsize = (10,8))



sns.set_style('darkgrid')

sns.lineplot(data=GndRtCondFirst,lw=2,legend=False)



plt.title('Gender ratio in first authorship',size=25)

plt.xlabel('Time',size=18);

plt.ylabel('Percentage of female first authors',size=18);

plt.xticks(fontsize=15);

plt.yticks(fontsize=15);
# We look at some descriptive statistics to better understand the variability of the data



STD = GndRtCondFirst.groupby(GndRtCondFirst.index.year).std()

AVE = GndRtCondFirst.groupby(GndRtCondFirst.index.year).mean()

AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)

STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)



STAT = pd.concat((AVE,STD),axis=1)

STAT
# We use the function previously created to repeat the analysis for the last author



GndRtCondLast = GenderRatio(dfcondlast)
fig = plt.figure(figsize = (10,8))



sns.set_style('darkgrid')

chart = sns.lineplot(data=GndRtCondLast,lw=2,legend=False)



plt.title('Gender ratio in last authorship',size=25)

plt.xlabel('Time',size=18);

plt.ylabel('Percentage of female last authors',size=18);

plt.xticks(fontsize=15);

plt.yticks(fontsize=15);
# We look at some descriptive statistics to better understand the variability of the data



STD = GndRtCondLast.groupby(GndRtCondLast.index.year).std()

AVE = GndRtCondLast.groupby(GndRtCondLast.index.year).mean()

AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)

STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)



STAT = pd.concat((AVE,STD),axis=1)

STAT
dfastro['first_author']=dfastro['authors'].apply(lambda x:x[0].split(' ')[0]) 

dfastro['last_author']=dfastro['authors'].apply(lambda x:x[-1].split(' ')[0])

dfastro.drop('authors',inplace=True,axis=1)



dfastro.drop(dfastro[dfastro['first_author'].map(len)<3].index, inplace=True)



dfastro['created'] = pd.to_datetime(dfastro['created'],format='%Y-%m-%d')

dfastro.drop(dfastro[dfastro['created'].dt.year<2012].index,inplace=True)



dfastro.reset_index(drop=True, inplace=True)
dfastro['first_author'] = dfastro['first_author'].apply(lambda x:x.capitalize())

dfastro['last_author'] = dfastro['last_author'].apply(lambda x:x.capitalize())



detect = gender.Detector()

dfastro['gender_first']=dfastro['first_author'].map(detect.get_gender)

dfastro['gender_last']=dfastro['last_author'].map(detect.get_gender)



dfastrofirst = dfastro.loc[:,['created','first_author','gender_first']]

dfastrofirst.rename(columns={'created':'date','first_author':'author','gender_first':'gender'},inplace=True)

dfastrolast = dfastro.loc[:,['created','last_author','gender_last']]

dfastrolast.rename(columns={'created':'date','last_author':'author','gender_last':'gender'},inplace=True)



dfastrofirst['gender'] = dfastrofirst['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',

                                                       'female':'female'})

dfastrofirst.dropna(inplace=True)



dfastrolast['gender'] = dfastrolast['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',

                                                       'female':'female'})

dfastrolast.dropna(inplace=True)
dfastrolast.info()
fig = plt.figure(figsize = (10,6))



sns.countplot(x=dfastrofirst[dfastrofirst['date'].dt.year<2020]['date'].dt.year,

              data = dfastrofirst[dfastrofirst['date'].dt.year<2020],hue='gender');



plt.xlabel('Year', fontsize=15);

plt.ylabel('Publication Count', fontsize=15);

plt.xticks(size=13);

plt.yticks(size=13);

plt.legend(fontsize=15);
fig = plt.figure(figsize = (10,6))



sns.countplot(x=dfastrolast[dfastrolast['date'].dt.year<2020]['date'].dt.year,

              data = dfastrolast[dfastrolast['date'].dt.year<2020],hue='gender');



plt.xlabel('Year', fontsize=15);

plt.ylabel('Publication Count', fontsize=15);

plt.xticks(size=13);

plt.yticks(size=13);

plt.legend(fontsize=15);
fig, axes = plt.subplots(2,3, figsize=(20,12),sharey=True)

d = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'}



for key,month in d.items():



        sns.countplot(dfastrofirst[dfastrofirst['date'].dt.month==key]['date'].dt.year,

              data = dfastrofirst[dfastrofirst['date'].dt.month==key], 

                      hue='gender',hue_order = ['male','female'], ax=axes[int(np.floor((key-1)/3)),

                                                                            int((key-1)-np.floor((key-1)/3)*3)])

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_title('arXiv publication for the month of %s' % month, size = 15);

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_xlabel('Year', size = 13);

        axes[int(np.floor((key-1)/3)),

             int((key-1)-np.floor((key-1)/3)*3)].set_ylabel('Publications',size=13);
# Use function to create the dataframe



GndRtAstroFirst = GenderRatio(dfastrofirst)
# Visualization of the female/male first author ratio between 2012 and 2020 



fig = plt.figure(figsize = (10,8))



sns.set_style('darkgrid')

sns.lineplot(data=GndRtAstroFirst,lw=2,legend=False)



plt.title('Gender ratio in first authorship',size=25)

plt.xlabel('Time',size=18);

plt.ylabel('Percentage of female first authors',size=18);

plt.xticks(fontsize=15);

plt.yticks(fontsize=15);
# We look at some descriptive statistics to better understand the variability of the data



STD = GndRtAstroFirst.groupby(GndRtAstroFirst.index.year).std()

AVE = GndRtAstroFirst.groupby(GndRtAstroFirst.index.year).mean()

AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)

STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)



STAT = pd.concat((AVE,STD),axis=1)

STAT
# Last author



GndRtAstroLast = GenderRatio(dfastrolast)
# Visualization of the female/male Last author ratio between 2012 and 2020 



fig = plt.figure(figsize = (10,8))



sns.set_style('darkgrid')

sns.lineplot(data=GndRtAstroLast,lw=2,legend=False)



plt.title('Gender ratio in last authorship',size=25)

plt.xlabel('Time',size=18);

plt.ylabel('Percentage of female last authors',size=18);

plt.xticks(fontsize=15);

plt.yticks(fontsize=15);
# We look at some descriptive statistics to better understand the variability of the data



STD = GndRtAstroLast.groupby(GndRtAstroLast.index.year).std()

AVE = GndRtAstroLast.groupby(GndRtAstroLast.index.year).mean()

AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)

STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)



STAT = pd.concat((AVE,STD),axis=1)

STAT