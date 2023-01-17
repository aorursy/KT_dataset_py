# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
data.head()
data.info()
data.describe().T
t=data.isnull().sum()/len(data)



print(t.sort_values(ascending=False))
data.drop(columns=['HDI for year'],inplace=True)
data.drop(columns=['country-year'],inplace=True)
data.head()
def unique_values(x):

    tf=data[x].unique()

    print('The column name:',x)

    print('The unique column names\n',tf)

    print('The total number of unique values',len(tf))

    print('\n')
t=data.columns

for i in range(len(t)):

    unique_values(t[i])
import seaborn as sns

import matplotlib.pyplot as plt
sucide_year=pd.DataFrame(data['suicides_no'].groupby([data['year']]).sum())

sucide_year=sucide_year.reset_index()

plt.figure(figsize=(15,8))

plt.title('Total no.of sucides per year')

sns.lineplot(x='year',y='suicides_no',data=sucide_year)

plt.ylabel('count of sucides')

sucide_sex=pd.DataFrame(data['suicides_no'].groupby([data['sex'],data['age']]).sum())

sucide_sex=sucide_sex.reset_index()

plt.figure(figsize=(15,8))

plt.title("sucides happend bascing on age group and gender")

sns.barplot(x='sex',y='suicides_no',hue='age',data=sucide_sex,estimator=sum)

plt.ylabel('count of sucides')

male_population=data.loc[data.loc[:,'sex']=='male']

female_population=data.loc[data.loc[:,'sex']=='female']
plt.figure(figsize=(15,7))

plt.title('The suicides rate as per year')

sns.lineplot(x='year',y='suicides_no',hue='sex',data=male_population,palette='Set1')

sns.lineplot(x='year',y='suicides_no',hue='sex',data=female_population,palette='Set2')

plt.ylabel('count of suicides')

plt.show()
suicide_country_age =pd.DataFrame(data['suicides_no'].groupby([data['country'],data['age']]).sum())

suicide_country_age=suicide_country_age.reset_index().sort_values(by='suicides_no',ascending=False)

most_count_c=suicide_country_age.head(20)

plt.figure(figsize=(15,8))

plt.title('Most suicides with different age groups')

sns.barplot(x='country',y='suicides_no',hue='age',data=most_count_c)

plt.ylabel('count of suicides')

plt.show()

suicide_country_age.tail(50)
most_count_c=suicide_country_age.tail(50)

plt.figure(figsize=(35,8))

plt.title('Least suicides with different age groups')

sns.set(font_scale=1)

sns.barplot(x='country',y='suicides_no',hue='age',data=most_count_c)

plt.ylabel('count of suicides')

plt.show()

suicides_generation=data['suicides_no'].groupby([data['generation'],data['year'],data['sex']]).sum()



suicides_generation=suicides_generation.reset_index()

plt.figure(figsize=(20,9))

plt.title('Total number of suicides per year ')

sns.lineplot(x='year',y='suicides_no',hue='generation',data=suicides_generation)

plt.ylabel('count of suicides')

plt.show()
sucide__sex=pd.DataFrame(data['suicides/100k pop'].groupby([data['sex'],data['age']]).sum())

sucide__sex=sucide__sex.reset_index()

plt.figure(figsize=(15,8))

plt.title("sucides/100k happend bascing on age group and gender")

sns.barplot(x='sex',y='suicides/100k pop',hue='age',data=sucide__sex,estimator=sum)

plt.ylabel('count of sucides')

sucide__year=pd.DataFrame(data['suicides/100k pop'].groupby([data['year']]).sum())

sucide__year=sucide__year.reset_index()

plt.figure(figsize=(15,8))

plt.title('Total no.of sucides per year')

sns.lineplot(x='year',y='suicides/100k pop',data=sucide__year)

plt.ylabel('count of sucides')

plt.figure(figsize=(15,7))

plt.title('The suicides rate as per year')

sns.lineplot(x='year',y='suicides/100k pop',hue='sex',data=male_population,palette='Set1')

sns.lineplot(x='year',y='suicides/100k pop',hue='sex',data=female_population,palette='Set2')

plt.ylabel('count of suicides')

plt.show()
suicide__country_age =pd.DataFrame(data['suicides/100k pop'].groupby([data['country'],data['age']]).sum())

suicide__country_age=suicide__country_age.reset_index().sort_values(by='suicides/100k pop',ascending=False)

most_count_c=suicide__country_age.head(20)

plt.figure(figsize=(25,8))

plt.title('Most suicides/100k population with different age groups')

sns.barplot(x='country',y='suicides/100k pop',hue='age',data=most_count_c)

plt.ylabel('count of suicides')

plt.show()

most_count_c=suicide__country_age.tail(50)

plt.figure(figsize=(55,15))

plt.title('Least suicides/100k population with different age groups')

sns.set(font_scale=0.8)

sns.barplot(x='country',y='suicides/100k pop',hue='age',data=most_count_c)

plt.ylabel('count of suicides')

plt.show()

suicides__generation=data['suicides/100k pop'].groupby([data['generation'],data['year'],data['sex']]).sum()



suicides__generation=suicides__generation.reset_index()

plt.figure(figsize=(20,9))

plt.title('Total number of suicides/100k population per year ')

sns.lineplot(x='year',y='suicides/100k pop',hue='generation',data=suicides__generation)

plt.ylabel('count of suicides')

plt.show()