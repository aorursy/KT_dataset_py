
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
color = sns.color_palette()

from subprocess import check_output


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

print(check_output(["ls", "../input"]).decode("utf8"))
kiva_loans= pd.read_csv("../input/kiva_loans.csv");
kiva_loans.head()

kiva_loans.shape
kiva_mpi_region_locations=pd.read_csv("../input/kiva_mpi_region_locations.csv");
kiva_mpi_region_locations.head()
loan_theme_ids=pd.read_csv("../input/loan_theme_ids.csv");
loan_theme_ids.head()
loan_theme_ids.shape
loan_themes_by_region=pd.read_csv("../input/loan_themes_by_region.csv");
loan_themes_by_region.head()
loan_themes_by_region.shape
country_series = kiva_loans['country'].value_counts().head(50)

country_series
type(country_series)
plt.figure(figsize=(20,10))


country_series = kiva_loans['country'].value_counts().head(30)

sns.barplot(country_series.values, country_series.index)
for i, v in enumerate(country_series.values):
    plt.text(15,i,v,color='k',fontsize=19)
plt.xticks(rotation='vertical')
plt.xlabel('Country Name')
plt.ylabel('Number of loans were given')
plt.title("Top countries in which more loans were given")
plt.show()
rwanda=kiva_loans.loc[kiva_loans['country'] == 'Rwanda']

rwanda['funded_amount'].count()
rwanda['funded_amount'].sum()
kenya=kiva_loans.loc[kiva_loans['country'] == 'Kenya']

kenya['funded_amount'].sum()
# print("Top sectors in which more loans were given : ", len(kiva_loans_data["sector"].unique()))
# print(kiva_loans_data["sector"].value_counts().head(10))
plt.figure(figsize=(15,8))
sector_name = kiva_loans['sector'].value_counts()
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xticks(rotation='vertical')
plt.xlabel('Sector Name')
plt.ylabel('Number of loans were given')
plt.title("Top sectors in which more loans were given")
plt.show()
rwanda.head()
plt.figure(figsize=(15,8))
sector_name = rwanda['sector'].value_counts()
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xticks(rotation='vertical')
plt.xlabel('Sector Name')
plt.ylabel('Number of loans were given')
plt.title("Top sectors in which more loans were given in RWANDA")
plt.show()
rwanda.iloc[:,[4,5]].head(10)
rwanda[['sector','use','funded_amount','posted_time','funded_time','borrower_genders']].head(10)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls

loan_use_in_rwanda = kiva_loans['use'][kiva_loans['country'] == 'Rwanda']
percentages = round(loan_use_in_rwanda.value_counts() / len(loan_use_in_rwanda) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top 13 loan uses in Rwanda',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)
gender_list = []
for gender in kiva_loans["borrower_genders"].values:
    if str(gender) != "nan":
        gender_list.extend( [lst.strip() for lst in gender.split(",")] )
temp_data = pd.Series(gender_list).value_counts()

labels = (np.array(temp_data.index))
sizes = (np.array((temp_data / temp_data.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Borrowers by Gender world wide')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")
gender_list = []
for gender in rwanda["borrower_genders"]:
    if str(gender) != "nan":
        gender_list.extend( [lst.strip() for lst in gender.split(",")] )
temp_data = pd.Series(gender_list).value_counts()

labels = (np.array(temp_data.index))
sizes = (np.array((temp_data / temp_data.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Borrowers by Gender in Rwanda')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")
east_africa  =["Rwanda","Kenya","Uganda","Tanzania","Burundi"]

east_africa_df = kiva_loans.loc[kiva_loans['country'].isin(east_africa)]
east_africa_df.head()
ea_country_series = east_africa_df['country'].value_counts()

sns.barplot(ea_country_series.values, ea_country_series.index)
for i, v in enumerate(ea_country_series.values):
    plt.text(15,i,v,color='k',fontsize=19)
plt.xticks(rotation='vertical')
plt.xlabel('Country Name')
plt.ylabel('Number of loans given in East Africa')
plt.title("East African countries")
plt.show()