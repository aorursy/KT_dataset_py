# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.float_format = '{:,.3f}'.format

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import variation ,bartlett, ttest_ind,f_oneway, normaltest,kruskal

from sklearn.cluster import KMeans 

pd.options.display.float_format = '{:,.1f}'.format

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#User defined functions

def print_Number_of_Missing_Values(df,col):

    '''

    The function is to calculate and print number of missing values and rates of the given dataframe and column.

    df=dataframe

    col=coloumn

    '''

    number_of_missing_values=df[col].isnull().sum()

    number_of_values=df[col].shape[0]

    ratio_of_missing_values=number_of_missing_values/number_of_values

    print('There are {:,.0f}-{:.0%} missing values in the population variable out of {:,.0f}.'.format(number_of_missing_values,

                                                                                     ratio_of_missing_values,number_of_values))
dataset=pd.read_csv("../input/who_suicide_statistics.csv")

dataset.head()
dataset.isnull().any()
dataset['suicides_no'].fillna(value=0,

              inplace=True)
#first calculate number of missing values

print_Number_of_Missing_Values(df=dataset,col='population')
# Let's create a new column in our data set that symbolize that if the population column is missing or not.

missing_row_filter=dataset['population'].isnull()

dataset['is_population_value_missing']=False

dataset['is_population_value_missing'][missing_row_filter]=True
temp_table=dataset.groupby('country').agg({'is_population_value_missing':['sum','count']}).reset_index()

Countries_with_missing_values_filter=temp_table['is_population_value_missing']['sum']>0

temp_table[Countries_with_missing_values_filter]
dataset.groupby(['country']).describe()
#Country-population plot

country_year_population_data=dataset.groupby(['country','year']).agg({'population':'sum'})

country_year_population_data.reset_index(inplace=True)



plt.figure(figsize=(25,35))

plt.title('Country-Population Plot')

sns.boxplot(x='population',

           y='country',

           data=country_year_population_data,);

del country_year_population_data
#country-suicides number plot

country_year_suicides_data=dataset.groupby(['country','year']).agg({'suicides_no':'sum'})

country_year_suicides_data.reset_index(inplace=True)



plt.figure(figsize=(25,25))

plt.title('Country-Suicide Number Distribution Plot')

sns.boxplot(y='country',

            x='suicides_no',

            data=country_year_suicides_data);

del country_year_suicides_data
#country-sucicide rates plot

country_year_suicide_ratio_data=dataset.groupby(['country','year']).agg({'suicides_no':'sum','population':'sum'})

country_year_suicide_ratio_data.reset_index(inplace=True)

country_year_suicide_ratio_data['suicide_ratio']=country_year_suicide_ratio_data['suicides_no']/country_year_suicide_ratio_data['population']



plt.figure(figsize=(25,25))

plt.title('Country-Suicide Ration Distribution')

sns.boxplot(y='country',

            x='suicide_ratio',

            data=country_year_suicide_ratio_data);
#Which countries have the highest suicide rates ?

country_suicide_ratio_data=dataset.dropna().groupby(['country']).agg({'suicides_no':'sum','population':'sum'})

country_suicide_ratio_data['suicide_ratio']=country_suicide_ratio_data['suicides_no']/country_suicide_ratio_data['population']

top_5_countries_with_the_highest_suicide_rates=country_suicide_ratio_data.sort_values(by='suicide_ratio',

                                       ascending=False).head().reset_index()



plt.figure(figsize=(10,5))

sns.barplot(x='suicide_ratio',

            y='country',

            data=top_5_countries_with_the_highest_suicide_rates).set_title('Top 5 Countries with Highest Suicide Rates')

del top_5_countries_with_the_highest_suicide_rates
#Which countries have the most volitile suicede rates ?

country_year_suicide_ratio_data_cv=country_year_suicide_ratio_data.groupby('country').agg({'suicide_ratio':variation})

#create a list of countries with most volatile suicide rates

countries_with_most_volatile_suicide_rates=list(country_year_suicide_ratio_data_cv.sort_values('suicide_ratio',ascending=False).index)[:5]

#create a filter to list these countries suicide rates

filter_temp=np.isin(country_year_suicide_ratio_data['country'],

                    countries_with_most_volatile_suicide_rates)



#plot the data

plt.figure(figsize=(15,5))

plt.title('Top 5 countries with most volatile suicide rates')

sns.boxplot(y='country',

           x='suicide_ratio',

           data=country_year_suicide_ratio_data[filter_temp]);
#Can I group these countries due to their suicide rates?



# lets cluster the countries into 3 different groups due to their suicide rates.

country_suicide_ratio_data_clustering=pd.DataFrame(country_suicide_ratio_data['suicide_ratio'].dropna())

X=np.array(country_suicide_ratio_data_clustering['suicide_ratio']).reshape(-1,1)



k_means=KMeans(n_clusters=3)

k_means.fit(X)#.values.reshape(-1,1))

The_saddest_cluster=k_means.cluster_centers_.argmax() #find the cluster with highest suicide rates

country_suicide_ratio_data_clustering['Clusters']=k_means.predict(X)



country_suicide_ratio_data_clustering=country_suicide_ratio_data_clustering.reset_index().sort_values(by='suicide_ratio')#preparing the data for plotting

print('The top saddest(with highest suicide rates) countries are :')

print(country_suicide_ratio_data_clustering[country_suicide_ratio_data_clustering['Clusters']==The_saddest_cluster]['country'].values)



plt.figure(figsize=(25,25)) #setting the plot size

sns.barplot(x='suicide_ratio', 

            y='country',

            hue='Clusters',

           data=country_suicide_ratio_data_clustering);

           
# Here are three plots of suicide numbers by years data. 

# In the first plot:

##I can see a drop in 1983 and 1984. It's not a natural change in the trend.

## Periaod between 1998 and 2003 has the most suicide numbers. And after these years, suicide numbers start droping down.

#The second and the third plot are made for checking that there is any misleading trend in the first plot.

## Despite the decline in the total number of suicide in 2013, Average number of suicide by counties are rising.



#first create a temp dataframe

data_temp=dataset.groupby('year').agg({'suicides_no':'sum',

                             'country':pd.Series.nunique})

data_temp['suicides_no_country_ratio']=data_temp['suicides_no']/data_temp['country']



#Now create plots

fig=plt.figure(figsize=(20,18))

ax1=fig.add_subplot(3,1,1)

data_temp['suicides_no'].plot(kind='bar',

                              title='Total Number of Suicides by Years',

                              ax=ax1,

                             sharex=True)



ax2=fig.add_subplot(3,1,2)

data_temp['country'].plot(kind='bar',

                          title='Number of unique countries in each year',

                          ax=ax2,

                          sharex=True)



ax3=fig.add_subplot(3,1,3)

data_temp['suicides_no_country_ratio'].plot(kind='bar',

                                            title='Average Number of Suicides by Country',

                                            ax=ax3,

                                            sharex=True);
#Create a dataset to analyze gender and suicide relationship

sex_suicide_rates_data=dataset[dataset['is_population_value_missing']==False][['country','sex','suicides_no','population']]

sex_suicide_rates_data['suicide_ratio']=sex_suicide_rates_data['suicides_no']/sex_suicide_rates_data['population']

sns.boxplot(y='sex',x='suicide_ratio',data=sex_suicide_rates_data).set_title('suicide ratio by sex');

#looks like suicide ratio in males is higher.

#lets do a statistical test

filter_male=sex_suicide_rates_data['sex']=='male' 

male_data=sex_suicide_rates_data[filter_male]['suicide_ratio']



filter_female=filter_male=sex_suicide_rates_data['sex']=='female'

female_data=sex_suicide_rates_data[filter_female]['suicide_ratio']



#Variance equality test

bartlett_statistics, bartlett_p_value=bartlett(male_data,female_data)

print('Variance equality test results:')

if bartlett_p_value>=0.05:

    print('We can assume that the variances of {} and {} data are equal to each other. We can do the t-test with this assumption'.format('male','female'))

else :

    print('That the variances of {} and {} data are NOT equal to each other. We can do the t-test with inequal variance assumption'.format('male','female'))

print('-'*50)



#Mean equality test

print('Mean equality test results:')

t_test_statistic, t_test_p_value=ttest_ind(male_data,female_data,equal_var=False)





### insert mean difference and confidence level resutls here

if t_test_p_value>=0.05:

    print('{} and {} data come from same population. We can do the t-test with equal variance assumption'.format('male','female'))

else :

    print('{} and {} data come from different population. We can do the t-test with inequal variance assumption'.format('male','female'))
data_age_suicideRatio=dataset[dataset['is_population_value_missing']==False].drop(columns=['sex','is_population_value_missing'])

data_age_suicideRatio.groupby(['country','year','age'],as_index=False).agg({'suicides_no':'sum',

                                            'population':'sum'})

data_age_suicideRatio['ratio']=data_age_suicideRatio['suicides_no']/data_age_suicideRatio['population']



#Normality test

for age_group in data_age_suicideRatio['age'].unique():

    age_filter=data_age_suicideRatio['age']==age_group

    Normality_test_statistic,Normalitiy_test_p_value=normaltest(data_age_suicideRatio[age_filter]['ratio'])

    if Normalitiy_test_p_value>=0.05:

        print('{} data has normal distribution.'.format(age_group))

    else :

        print('{} data has other kind of distribution except normal.'.format(age_group))

 



#Variance test: Due to the normality test results, Kruskall test is used for variance equality test.



krs_statistics, krs_p_value=kruskal(data_age_suicideRatio[data_age_suicideRatio['age']=='15-24 years']['ratio'],

        data_age_suicideRatio[data_age_suicideRatio['age']=='25-34 years']['ratio'],

        data_age_suicideRatio[data_age_suicideRatio['age']=='35-54 years']['ratio'],

        data_age_suicideRatio[data_age_suicideRatio['age']=='5-14 years']['ratio'],

        data_age_suicideRatio[data_age_suicideRatio['age']=='55-74 years']['ratio'],

        data_age_suicideRatio[data_age_suicideRatio['age']=='75+ years']['ratio']

       )

### insert mean difference and confidence level resutls here

if krs_p_value>=0.05:

    print('Suicide ratios are diffenrent between age groups. Some groups have different suicide ratios than the others')

else :

    print('Suicide ratios are Not diffenrent between age groups. All age groups have same suicide ratios.')



sns.boxplot(y='age',

            x='ratio',

            data=data_age_suicideRatio).set_title('Suicide ratios by age groups');