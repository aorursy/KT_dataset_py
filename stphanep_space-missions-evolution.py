!pip install seaborn --upgrade
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')



# Extract the country

df['country'] = df['Location'].str.split(', ').str[-1]



countries_dict = {

    'Russia' : 'Russian Federation',

    'New Mexico' : 'USA',

    "Yellow Sea": 'China',

    "Shahrud Missile Test Site": "Iran",

    "Pacific Missile Range Facility": 'USA',

    "Barents Sea": 'Russian Federation',

    "Gran Canaria": 'USA',

    "Pacific Ocean": 'USA',

    "Kenya" : 'Italia'

}

df['country'] = df['country'].replace(countries_dict)



# Retrieve the year

df['Launch date']=pd.to_datetime(df['Datum'])

df['Launch date']=df['Launch date'].astype(str)





df['Launch date']=df['Launch date'].str.split(' ',expand=True)[0]

df['Launch date']=pd.to_datetime(df['Launch date'])



df['Year']=df['Launch date'].dt.year



# Categorize private or public

private_company = ['SpaceX', 'ULA', 'Northrop', 'Rocket Lab', 'Virgin Orbit', 'MHI', 'Arianespace', 'Blue Origin',

                   'Exos', 'ILS', 'i-Space', 'OneSpace', 'Landspace', 'Eurockot', 'Land Launch', 'Kosmotras',

                   'Sea Launch', 'Boeing', 'SRC', 'Lockheed', 'Starsem', 'General Dynamics', 'Martin Marietta',

                   'Douglas', 'AMBA']



dataset_private = df[df['Company Name'].isin(private_company)]

dataset_public = df[~df['Company Name'].isin(private_company)]

df['PrivateCompany'] = df['Company Name'].isin(private_company)



# Remove useless features

df.drop(['Unnamed: 0.1','Unnamed: 0'], axis = 1, inplace = True) 



# Group failure together

fail_dict = {

    'Partial Failure':'Failure',

    'Prelaunch Failure':'Failure'

}

df['Status Mission'].replace(fail_dict, inplace=True)

plt.figure(figsize=(18,8))

ax = sns.countplot(x=df['Year'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
# Evolution of country participation

df_countries=df.groupby('country').count().sort_values('Location', ascending=False)

eight_first_country=df_countries.head(8).index

df_countries = df.copy()

df_countries['country'] = df['country'].apply(lambda a : a if a in eight_first_country else 'Others')



fig = plt.figure(figsize=(15,9))

ax1 = fig.add_subplot(111)

ax1.plot(sns.histplot(df_countries, x='Year', hue='country', multiple="stack").plot())

plt.show()
# Calculate percentage

df_countries_count = df_countries.groupby(['Year', 'country'])['Datum'].count().unstack('country')

df_countries_count['total_year'] = df_countries_count.sum(axis=1)

df_countries_count = df_countries_count.div(df_countries_count['total_year'], axis=0).mul(100)

df_countries_count.drop('total_year', axis=1,inplace=True)



fig = plt.figure(figsize=(26,16))

# Initialize the figure

plt.style.use('seaborn-darkgrid')



# create a color palette

palette = plt.get_cmap('Set1')



# multiple line plot

num=0

for column in df_countries_count:

    num+=1



    # Find the right spot on the plot

    plt.subplot(3,3, num)



    # plot every groups, but discreet

    for v in df_countries_count:

        plt.plot(df_countries_count.index, df_countries_count[v], marker='', color='grey', linewidth=0.6, alpha=0.3)



    # Plot the lineplot

    plt.plot(df_countries_count.index, df_countries_count[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)



    # Same limits for everybody!

    plt.xlim(1956,2020)

    plt.ylim(-2,85)



    # Not ticks everywhere

    if num in range(7) :

        plt.tick_params(labelbottom='off')

    if num not in [1,4,7] :

        plt.tick_params(labelleft='off')



    # Add title

    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )



# general title

plt.suptitle("Percentage of launches per country", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

plt.show()


count_private = df.groupby(['Year',]).agg(

    nb_launch=pd.NamedAgg('PrivateCompany', aggfunc='count'),

    nb_private=pd.NamedAgg('PrivateCompany',  sum)

)



percent = 100*count_private['nb_private']/count_private['nb_launch']



fig= plt.figure(figsize=(16,9))

ax1 = fig.add_subplot(111)

ax1.plot(sns.histplot(df, x='Year', hue='PrivateCompany',multiple="stack").plot())

ax2 = ax1.twinx()

ax2.plot(sns.lineplot(data=percent.rolling(window=3, center=True).mean(), color='red').plot())

ax2.set_ylabel('Percent')

plt.title('Evolution per year of the number of launch by category (public/private) with a three years rolling percentage')

plt.show()



# Explication du pic des années 60

print('Explanation of the peak during the 60\'')

print(dataset_private[(dataset_private['Year']>1965) & (dataset_private['Year']<1970)]['Company Name'].value_counts()[:4])



# Explication du pic des années 60

print("Explanation of the five last years decrease")

print(dataset_public[(dataset_public['Year']>2015) & (dataset_public['Year']<=2020)]['Company Name'].value_counts()[:4])
company_count = df['Company Name'].value_counts()



#company_count = company_count.loc(company_count['values'] >= 9.5)

f, ax = plt.subplots(figsize=(25, 7))

ax.set(yscale="log")

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

sns.barplot(x=company_count.index, y=company_count.values, alpha=0.8)

ax2 = ax.twinx()

ax2.plot(sns.lineplot(x=company_count.index, y=company_count.values, color='red').plot(),label='Normal scale', color='red')

plt.legend()

plt.show()



count_per_private = df[df['PrivateCompany']==True].groupby(['Company Name']).agg('count')['country'].rename('Launch per private company')

print('The ten first private companies are :', count_per_private.sort_values(ascending=False).head(10),'\n')

print(count_per_private.describe(),'\n')



count_per_public = df[df['PrivateCompany']==False].groupby(['Company Name']).agg('count')['country'].rename('Launch per public company')

print(count_per_public.describe(),'\n')



company_existence = df.groupby(['Company Name']).agg(

    start=pd.NamedAgg('Year', min),

    end=pd.NamedAgg('Year', max)

)



nb_active_company = []

for year in range(df['Year'].min(), df['Year'].max()):

    nb = company_existence[(company_existence['start']<=year) & (company_existence['end']>=year)].count()[0]

    nb_active_company.append([year, nb])

nb_active_company = pd.DataFrame(nb_active_company,columns=['Year', 'Count'])



private_country = df[['country', 'PrivateCompany', 'Company Name']].drop_duplicates(ignore_index=True)

private_country=private_country.groupby('country').sum()['PrivateCompany'].sort_values(ascending=False).head(9)

br = sns.barplot(x=private_country.index,y=private_country.values, alpha=0.8)

br.set_xticklabels(br.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Number of private company per country')

plt.show()



private_launch_country = df[df['PrivateCompany']==True].groupby('country').count()['Year'].rename('Count').sort_values(axis=0, ascending=False)

br = sns.barplot(x=private_launch_country.index,y=private_launch_country.values, alpha=0.8)

br.set_xticklabels(br.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Number of launches operate by private company per country')

plt.show()


for c in ['USA', 'France', 'Japan']:

    count_private_us = df[df['country']==c].groupby(['Year',]).agg(

        nb_launch=pd.NamedAgg('PrivateCompany', aggfunc='count'),

        nb_private=pd.NamedAgg('PrivateCompany',  sum)

    )



    percent_us = 100*count_private_us['nb_private']/count_private_us['nb_launch']



    fig= plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(sns.histplot(df[df['country']==c], x='Year', hue='PrivateCompany',multiple="stack").plot())

    ax2 = ax1.twinx()

    ax2.plot(sns.lineplot(data=percent_us.rolling(window=3, center=True).mean(), color='red').plot())

    ax2.set_ylabel('Percent')

    plt.title(f'Evolution of the number of mission (public/private) for {c} with a three years rolling percentage')

    plt.show()

    print(f"Private companies names for {c} : {df[(df['country']==c) &(df['PrivateCompany']==True)]['Company Name'].unique()}")

company_existence = df.groupby(['Company Name']).agg(

    start=pd.NamedAgg('Year', min),

    end=pd.NamedAgg('Year', max)

)



nb_active_company = []

for year in range(df['Year'].min(), df['Year'].max()):

    nb = company_existence[(company_existence['start']<=year) & (company_existence['end']>=year)].count()[0]

    nb_active_company.append([year, nb])

nb_active_company = pd.DataFrame(nb_active_company,columns=['Year', 'Count'])







country_participation = df.groupby(['country']).agg(

    start=pd.NamedAgg('Year', min),

    end=pd.NamedAgg('Year', max)

)



nb_active_country = []

for year in range(df['Year'].min(), df['Year'].max()):

    nb = country_participation[(country_participation['start']<=year) & (country_participation['end']>=year)].count()[0]

    nb_active_country.append([year, nb])

nb_active_country = pd.DataFrame(nb_active_country,columns=['Year', 'Count'])



fig = plt.figure()

ax1 = fig.add_subplot()

ax1.plot(sns.lineplot(x='Year', y='Count', data=nb_active_company).plot())

ax1.set_ylabel('Company')

ax1.set_label('aaaa')



ax2 = ax1.twinx()

ax2.plot(sns.lineplot(x='Year', y='Count', data=nb_active_country,color='green').plot(), color='green', label='Country')

plt.title('Number of active country and company')

ax2.set_ylabel('Country')

plt.legend()

plt.show()
success_evol = df.groupby(['Year', 'Status Mission']).count()['country'].rename('Count').unstack()

success_evol['sum']=success_evol.sum(axis=1)

success_evol = success_evol.div(success_evol['sum'], axis=0).mul(100)

success_evol.drop('sum', axis=1,inplace=True)



mean_failure = success_evol['Failure'].sum()/(success_evol['Failure'].sum()+success_evol['Success'].sum())



success_evol.plot(kind='bar',stacked=True)



plt.legend()

plt.show()



print(f'The world mean failure rate is {mean_failure.round(3)*100}%','\n')



#%%

for country in ['China', 'India', 'Iran']:

    success_evol_country = df[df['country']==country].groupby(['Year', 'Status Mission']).count()['country'].rename('Count').unstack()

    success_evol_country['sum']=success_evol_country.sum(axis=1)

    success_evol_country = success_evol_country.div(success_evol_country['sum'], axis=0).mul(100)

    success_evol_country.drop('sum', axis=1,inplace=True)

    mean_failure_country = success_evol_country['Failure'].sum()/(success_evol_country['Failure'].sum()+success_evol_country['Success'].sum())

    

    success_evol_country_world = success_evol[success_evol.index >= success_evol_country.index.min()]

    mean_failure_world_country = success_evol_country_world['Failure'].sum()/(success_evol_country_world['Failure'].sum()+success_evol_country_world['Success'].sum())

    

    success_evol_usa = df[df['country']=='USA'].groupby(['Year', 'Status Mission']).count()['country'].rename('Count').unstack()

    success_evol_country_usa = success_evol_usa[success_evol_usa.index >= success_evol_country.index.min()]

    mean_failure_usa_country = success_evol_country_usa['Failure'].sum()/(success_evol_country_usa['Failure'].sum()+success_evol_country_usa['Success'].sum())



    print(f'{country} launched its first mission in {success_evol_country.index.min()} '

          f'since this date it has a failure rate of {mean_failure_country.round(3)*100}% '

          f'and for the same period the world failure rate is {(mean_failure_world_country.round(3)*100).round(1)}% '

         f'and for the USA failure rate is {(mean_failure_usa_country.round(3)*100).round(1)}% ')

statut_evol=df.groupby(['Year', 'Status Rocket']).count()['country'].rename('count').unstack()

statut_evol.plot(kind='bar',stacked=True)

plt.locator_params(nbins=20)

plt.show()



oldest = df.iloc[df[df['Status Rocket'] == 'StatusActive']['Year'].idxmin(axis=0)]

print(f'The oldest and still active satellite has been launched in {oldest["Year"]} '

      f'by {oldest["country"]} from {oldest["Location"]}')
dict_display = []

for country in df_countries['country'].unique():

    statut_evol=df_countries[df_countries['country']==country].groupby(['Year', 'Status Rocket']).count()['country'].rename('count').unstack()

    statut_evol.plot(kind='bar',stacked=True,title='Status evolution for '+country)

    plt.show()

    

    sum = statut_evol.sum(axis=0)

    if len(sum)==2:

        dict_display.append([country, (sum[0]/(sum[0]+sum[1])).round(3)*100, sum[0]])

        

   
for a in dict_display:        

    print(f'{a[0]} has {a[2]} missions still active, that represented {a[1].round(1)}% of its total missions')
fig= plt.figure()

statut_evol=df_countries[df_countries['Status Rocket']=='StatusActive'].groupby(['country']).count()['Datum'].rename('count')

statut_evol.sort_values(ascending=False, inplace=True)



ax1=statut_evol.plot(kind='bar')



sum = statut_evol.sum(axis=0)

statut_evol = statut_evol*100/sum

ax2=ax1.twinx()

ax2.plot(statut_evol, color='red', label='World percentage')

plt.legend()

plt.show()



print(f"Russian and Kazakhstan has send "

      f"{float(len(df[df['country'].isin(['Russian Federation', 'Kazakhstan'])].index)/len(df.index)).__round__(3)*100}% of all missions "

      f"but has only {statut_evol[statut_evol.index.isin(['Russian Federation', 'Kazakhstan'])].sum().round(1)}% of the active missions")
