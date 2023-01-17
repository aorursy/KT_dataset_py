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
import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import textwrap
suicide_data = pd.read_csv('../input/suicide-dataset/suicide_dataset.csv')

suicide_data.head()
suicide_data.isnull().sum()
Nan_columns = suicide_data.iloc[:,4:22]

suicide_data.drop(Nan_columns, axis=1,inplace=True)

suicide_data.head(5)
mean_value = round(suicide_data['Suicide Rate'].mean(),1)

suicide_data['Suicide Rate'].replace(np.nan, mean_value, inplace=True)

suicide_data.head(5)
suicide_data = suicide_data.dropna()

suicide_data
suicide_data2000 = suicide_data.loc[suicide_data.Year.isin(['2000'])]

suicide_data2000.head(5)
su_data2000 = suicide_data2000.groupby('Country')['Suicide Rate'].sum().reset_index()

Su_data2000 = su_data2000.sort_values('Suicide Rate',ascending=False).head(8)

Su_data2000.reset_index(inplace=True)



su_data2000min = suicide_data2000.groupby('Country')['Suicide Rate'].sum().reset_index()

Su_data2000min = su_data2000min.sort_values('Suicide Rate',ascending=True).head(8)

Su_data2000min.reset_index(inplace=True)



print(Su_data2000, '\n')

print(Su_data2000min)
max_width = 15

suicide = [Su_data2000,Su_data2000min]

suicide_title = ['Top 8', 'Bottom 8']

fig, ax = plt.subplots(2,1, figsize = (22,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = suicide[i], x = 'Country', y = 'Suicide Rate')

    sns.barplot(ax = ax[i], data = suicide[i], x = 'Country', y = 'Suicide Rate')

    ax[i].legend()

    ax[i].set_title(suicide_title[i]+' Countries with the highest Suicide Rate', fontsize = 20)

    ax[i].set_ylabel('Country', fontsize = 20)

    ax[i].set_xlabel('Suicide Count', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 200, step = 50))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
suicide_datac = suicide_data.groupby('Country')['Suicide Rate'].sum().reset_index()

Suicide_datac = suicide_datac.sort_values('Suicide Rate',ascending=False).head(12)

Suicide_datac
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Suicide_datac,x = 'Country',y = 'Suicide Rate', ax = ax)

ax.set_ylabel('Suicide Count')

ax.set_title('Top 12 Countries with the highest Suicide Rate')

for index,Suicide_datac in enumerate(Suicide_datac['Suicide Rate'].astype(int)):

       ax.text(x=index-0.1 , y =Suicide_datac+2 , s=f"{Suicide_datac}" , fontdict=dict(fontsize=8))

plt.show()
sex_suicide = suicide_data2000.groupby('Sex')['Suicide Rate'].sum().reset_index()

Sex_suicide = sex_suicide.sort_values('Suicide Rate',ascending=False)

Sex_suicide
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Sex_suicide,x = 'Sex',y = 'Suicide Rate', ax = ax)

ax.set_ylabel('Suicide Count')

ax.set_title('Gender with the highest Suicide Rate in 2000')

for index,Sex_suicide in enumerate(Sex_suicide['Suicide Rate'].astype(int)):

       ax.text(x=index-0.1 , y =Sex_suicide+1 , s=f"{Sex_suicide}" , fontdict=dict(fontsize=10))

plt.show()
gender_russia = suicide_data2000.loc[suicide_data2000.Country.isin(['Russian Federation'])]

gender_russia.head()
import plotly.express as px

fig = px.pie(gender_russia, values=gender_russia['Suicide Rate'], names=gender_russia['Sex'])

fig.update_layout(title = 'Gender with high Suicidal Rate')

fig.show()
year_suicide = suicide_data.groupby('Year')['Suicide Rate'].sum().reset_index()

year_suicide
plt.figure(figsize=(10,5))

chart = sns.barplot(

    data=year_suicide,

    x='Year',

    y='Suicide Rate',

    palette='Set1'

)

chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation=65, 

    horizontalalignment='right',

    fontweight='light',

 

)

chart.axes.yaxis.label.set_text("Suicide Count")
import plotly.express as px

fig = px.line(y=year_suicide['Suicide Rate'], x=year_suicide['Year'], labels={'y':'Suicide Count', 'x':'Year'})

fig.update_layout(title = 'Relation between the Year and Suicide Rate')

fig.show()
suicide_data.Country.unique()
suicide_Afr = suicide_data.loc[suicide_data.Country.isin(['Egypt','Nigeria','Ghana','South Africa','Tunisia','Uganda','Liberia','Sudan','Equatorial Guinea','Cameroon'])]

suicide_Afr
su_dataAfric = suicide_Afr.groupby('Country')['Suicide Rate'].sum().reset_index()

Su_dataAfric = su_dataAfric.sort_values('Suicide Rate',ascending=False).head(10)

Su_dataAfric
import plotly.express as px



fig = px.choropleth(locations= Su_dataAfric['Country'], 

                    locationmode="country names", 

                    color= Su_dataAfric['Suicide Rate'],

                    labels={'color':'Suicide Rate', 'locations':'Country'},

                    scope="africa") 





fig.update_layout(

    

    title_text = 'Top 10 Countries in Africa with High Suicide Rate',

    geo_scope='africa'

)

fig.show()
suicide_Asia = suicide_data.loc[suicide_data.Country.isin(['India','Viet Nam','Republic of Korea','Japan','China','Malaysia','Thailand','Indonesia','Singapore','Malaysia'])]

suicide_Asia
su_dataAsia = suicide_Asia.groupby('Country')['Suicide Rate'].sum().reset_index()

Su_dataAsia = su_dataAsia.sort_values('Suicide Rate',ascending=False).head(10)

Su_dataAsia
fig = px.choropleth(locations= Su_dataAsia['Country'], 

                    locationmode="country names", 

                    color= Su_dataAsia['Suicide Rate'],

                    labels={'color':'Suicide Rate', 'locations':'Country'},

                    scope="asia") 





fig.update_layout(

    

    title_text = 'Top 10 Countries in Asia with High Suicide Rate',

    geo_scope='asia'

)

fig.show()