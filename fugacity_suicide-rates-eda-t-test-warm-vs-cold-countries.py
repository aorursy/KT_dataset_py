# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# for Box-Cox Transformation

from scipy import stats

from scipy.stats import pearsonr



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# for Welch t-test

import scipy.stats



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')



total_cells = np.product(df.shape)

missing_cells = df.isnull().sum().sum()



percent_missing_cells=(missing_cells/total_cells) * 100



# percent of data that is missing

print ('Total # of Cells: %d' %total_cells)

print('Total # of Missing Cells: %d' %missing_cells)

print('Percent Missing Cells: %d' %percent_missing_cells)



print('\nColumns with missing_values_count: ')

df.isnull().sum()
# Dropping HDI column based on Missing Value analysis

df.drop(columns=['HDI for year'],inplace=True)



# Converting GFD_per_capita to numeric

df['gdp_per_capita ($)']=pd.to_numeric(df['gdp_per_capita ($)'])



# Dropping 'years' from Age feature for easier read

df['age'].replace(regex=True,inplace=True,to_replace=r'years',value=r'')



# Removing word Generation from generation feature

df['generation']=df['generation'].apply(lambda x: x.replace('Generation',''))


plt.figure(figsize=(22,36))

plt.subplot(221)

sns.countplot(y='country',data=df, alpha=0.7, order=reversed(df['country'].value_counts().index))

plt.axvline(x=df['country'].value_counts().mean(), color='k')

plt.gca().xaxis.tick_bottom()

plt.title('Data count by Country \nFrom 1987 to 2016')



plt.figure(figsize=(14,5))

plt.subplot(221)

sns.boxplot(x=df['country'].value_counts())

plt.gca().xaxis.tick_bottom()

plt.title('Data count distribution by Country')



plt.subplot(222)

df['country'].value_counts().plot.kde()

plt.gca().xaxis.tick_bottom()

plt.title('Data count density distribution by Country')
plt.figure(figsize=(20,10))

#fig.subplots_adjust(hspace=0.4, wspace=1)



plt.subplot(231)

sns.countplot(y='generation',data=df, alpha=0.7, order=reversed(df['generation'].value_counts().index))

plt.axvline(x=df['generation'].value_counts().mean(), color='k')

plt.gca().xaxis.tick_bottom()

plt.title('Data count by Generation')



plt.subplot(232)

sns.countplot(y='age',data=df)

plt.gca().xaxis.tick_bottom()

plt.title('Data count by Age')



plt.subplot(233)

sns.countplot(y='sex',data=df)

plt.gca().xaxis.tick_bottom()

plt.title('Data count by Sex')
by_country_suicide= df.groupby('country').mean().sort_values('suicides/100k pop', ascending=False).reset_index()





plt.figure(figsize=(22,36))

plt.subplot(221)

sns.barplot(x='suicides/100k pop',y='country',data=by_country_suicide)

plt.axvline(x=by_country_suicide['suicides/100k pop'].mean(), color='k')

plt.gca().xaxis.tick_bottom()

plt.title('Suicides per 100k of population by Country \nFrom 1987 to 2016')



plt.tight_layout()
plt.figure(figsize=(14,5))

plt.subplot(221)

sns.boxplot(df['suicides/100k pop'])



plt.subplot(222)

sns.boxplot(df['gdp_per_capita ($)'])
def subset_by_iqr(df, column, whisker_width=1.5):



    # Calculate Q1, Q2 and IQR

    q1 = df[column].quantile(0.25)                 

    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    # Apply filter with respect to IQR, including optional whiskers

    

    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)

    return df.loc[filter]                                                     





df_filtered_suicide = subset_by_iqr(df, 'suicides/100k pop', whisker_width=1.5)

df_filtered_gpd = subset_by_iqr(df, 'gdp_per_capita ($)', whisker_width=1.5)



df_filtered_suicide = df_filtered_suicide.rename(columns ={'suicides/100k pop' : 'filtered_suicides/100k pop'})

df_filtered_gpd = df_filtered_gpd.rename(columns ={'gdp_per_capita ($)' : 'filtered_gdp_per_capita ($)'})

plt.figure(figsize=(14,5))

plt.subplot(221)

sns.boxplot(df['suicides/100k pop'], color = 'red')

plt.gca().xaxis.tick_bottom()



plt.subplot(222)

sns.boxplot(df['gdp_per_capita ($)'],  color = 'red')

plt.gca().xaxis.tick_bottom()



plt.subplot(223)

sns.boxplot(df_filtered_suicide['filtered_suicides/100k pop'],color = 'g')

plt.gca().xaxis.tick_bottom()



plt.subplot(224)

sns.boxplot(df_filtered_gpd['filtered_gdp_per_capita ($)'], color = 'g')

plt.gca().xaxis.tick_bottom()





plt.tight_layout()
df_filtered = df_filtered_suicide.merge(df_filtered_gpd)



print ('Original data: ',df.shape[0])

print ('Filtered data: ',df_filtered.shape[0])



print ('Percent outliers by suicides/100k pop and GDP: ', 100 - df_filtered.shape[0]*100/df.shape[0])





# Plotting Suicides/100k of population and GDP_per_capita as a function of years



suicide_year_gdp = df_filtered[['suicides/100k pop','year','gdp_per_capita ($)']].groupby(['year']).mean().reset_index()



fig, ax1 = plt.subplots(figsize=(12,6))



# Plot the suicides over the years.

ln1 = ax1.plot(suicide_year_gdp['year'], suicide_year_gdp['suicides/100k pop'], 'ro--', label='Suicides')

plt.gca().xaxis.tick_bottom()

# Adding GDP plot to the same plot, but on different scale.

ax2 = ax1.twinx()

ln2 = ax2.plot(suicide_year_gdp['year'], suicide_year_gdp['gdp_per_capita ($)'], 'o--', label='GDP')



# Joining legends.

lns = ln1 + ln2

labels = [l.get_label() for l in lns]

ax1.legend(lns, labels, loc=2)



# Setting labels

ax1.set_ylabel('Suicides per 100k population')

ax2.set_ylabel('GDP per Capita($)')

ax1.set_xlabel('Time(Years)')

plt.title ('Evolution of Suicide and GDP')



plt.show()



suicide_year_gdp.corr()
df_filtered_2015_2016 = df_filtered[df_filtered['year']==2016]





by_country_suicide= df_filtered_2015_2016.groupby('country').mean().sort_values('suicides/100k pop', ascending=False).reset_index()





plt.figure(figsize=(14,8))

plt.subplot(221)

sns.barplot(x='suicides/100k pop',y='country',data=by_country_suicide)

plt.axvline(x=by_country_suicide['suicides/100k pop'].mean(), color='k')

plt.gca().xaxis.tick_bottom()

plt.title('Suicides per 100k of population by Country in 2016')



plt.tight_layout()
by_Sex = df.groupby(['sex']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()

by_Sex_Time = df.groupby(['sex','year']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()

by_Sex_Age = df.groupby(['sex','age']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()

by_Sex_Gen = df.groupby(['sex','generation']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()

plt.figure(figsize=(16,10))



plt.subplot(231)

sns.barplot(x='sex',y='suicides/100k pop', data=by_Sex,alpha=0.7,  ci='sd')

plt.gca().xaxis.tick_bottom()

plt.title('Average Suicides by Sex \nFrom 1987 to 2016')



plt.subplot(232)

sns.barplot(x='age',y='suicides/100k pop', hue='sex', data=by_Sex_Age,alpha=0.7)

plt.gca().xaxis.tick_bottom()

plt.title('Average Suicides Age & Sex \nFrom 1987 to 2016')



plt.subplot(233)

sns.barplot(x='generation',y='suicides/100k pop', hue='sex', data=by_Sex_Gen, alpha=0.7, palette=('husl'))

plt.gca().xaxis.tick_bottom()

plt.title('Average Suicides by Sex & Generation \nFrom 1987 to 2016')



plt.tight_layout()
by_Gen_Time = df.groupby(['generation','year']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()

by_Age_Time = df.groupby(['age','year']).mean().sort_values('suicides/100k pop', ascending=True).reset_index()
plt.figure(figsize=(22,8))

fig.subplots_adjust(hspace=.5)





plt.subplot(131)

sns.lineplot(x='year',y='suicides/100k pop', hue='sex', data=by_Sex_Time,alpha=0.7)

plt.gca().xaxis.tick_bottom()

plt.title('Evolution of average sucide/100k of population by Sex')



plt.subplot(132)

sns.lineplot(x='year',y='suicides/100k pop', hue='age', data=by_Age_Time, alpha=0.7)

plt.gca().xaxis.tick_bottom()

plt.title('Evolution of average sucide/100k of population by Age')



plt.subplot(133)

sns.lineplot(x='year',y='suicides/100k pop', hue='generation', data=by_Gen_Time, alpha=0.7)

plt.gca().xaxis.tick_bottom()

plt.title('Evolution of average sucide/100k of population by Generation')



# Load world-capitals-gps

continents=pd.read_csv("../input/world-capitals-gps/concap.csv")   # world GPS

suicide = df
continents['country'] = continents['CountryName'].apply(lambda x: x.lower())

suicide['country'] = suicide['country'].apply(lambda x: x.lower())



suicide_country = suicide.groupby(['country']).mean().reset_index()



# Merging Suicidce and GPS dataframes based on country using left join method

combined_df = pd.merge(suicide_country,

                            continents[['country','ContinentName']]

                            ,on='country',how='left')



combined_df[combined_df.isna().any(axis=1)]
# Lets impute countries have not been assigned with a continent name

combined_df.reset_index()



combined_df.loc[17,'ContinentName'] = 'Africa'

combined_df.loc[73,'ContinentName'] = 'Asia'

combined_df.loc[75,'ContinentName'] = 'Europe'

combined_df.loc[78,'ContinentName'] = 'North America'
# Lets look have many countries are in each continent group

sns.countplot(y='ContinentName',data =combined_df, alpha=0.7, order=reversed(combined_df['ContinentName'].value_counts().index))

asian_countries =         combined_df.loc[combined_df['ContinentName']=='Asia']

european_countries =      combined_df.loc[combined_df['ContinentName']=='Europe']

south_america_countries = combined_df.loc[combined_df['ContinentName']=='South America']

north_america_countries = combined_df.loc[combined_df['ContinentName']=='North America']

africa_countries =        combined_df.loc[combined_df['ContinentName']=='Africa']
# Looping 10 T-test



t_test_result=[]

def t_test_loop(group_1,group_2,sample_size):

    

    group1_continent = group_1['ContinentName'].iloc[0]

    group2_continent = group_2['ContinentName'].iloc[0]

    print ('%s vs %s (sample size=%s) :  \n'%(group1_continent,group2_continent,sample_size))

    for i in range(11):

        

        random_group1_countries = group_1.sample(sample_size)

        random_group2_countries = group_2.sample(sample_size)

        

        group1_suicide = random_group1_countries['suicides/100k pop'].tolist()

        group2_suicide = random_group2_countries['suicides/100k pop'].tolist()

    

        p_val = scipy.stats.ttest_ind(group1_suicide,group2_suicide,equal_var=False).pvalue

        t_test_result.append(round(p_val,2))

        print ('Welch t-Test results:',scipy.stats.ttest_ind(group1_suicide,group2_suicide,equal_var=False))

        

    

  

    return  print ('\nP-value results of series of t-Tests: ',t_test_result)

    

t_test_loop(north_america_countries,south_america_countries,9)
def t_test(group_1,group_2,sample_size):

    

    group1_continent = group_1['ContinentName'].iloc[0]

    group2_continent = group_2['ContinentName'].iloc[0]

    

    random_group1_countries = group_1.sample(sample_size)

    random_group2_countries = group_2.sample(sample_size)

    

    group1_suicide = random_group1_countries['suicides/100k pop'].tolist()

    group2_suicide = random_group2_countries['suicides/100k pop'].tolist()



    p_val = scipy.stats.ttest_ind(group1_suicide,group2_suicide,equal_var=False).pvalue

    

    group_1_mean = round(random_group1_countries['suicides/100k pop'].mean(),2)

    group_1_std = round(random_group1_countries['suicides/100k pop'].std(),2)

    group_2_mean = round(random_group2_countries['suicides/100k pop'].mean(),2)

    group_2_std = round(random_group2_countries['suicides/100k pop'].std(),2)

    

    print ('%s vs %s (sample size=%s) :  \n'%(group1_continent,group2_continent,sample_size))

    

    print ('1st group countries:',random_group1_countries['country'].tolist())

    print ('2nd group countries:',random_group2_countries['country'].tolist())

    

    print ('\n1st group average suicide/100k of population:  %s +- %s '%(group_1_mean,group_1_std))

    print ('2st group average suicide/100k of population:  %s +- %s '%(group_2_mean,group_2_std))



    print ('\nWelch t-Test results:',scipy.stats.ttest_ind(group1_suicide,group2_suicide,equal_var=False))

    

    if p_val<0.05:

        print ('\np values = {} ==> reject Ho'.format(round(p_val,2)))

    else:

        print ('\np values = {} ==> can NOT reject Ho'.format(round(p_val,2)))

    

t_test(asian_countries,european_countries,10)