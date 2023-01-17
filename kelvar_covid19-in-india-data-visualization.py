import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Path of the file to read

indiaCovid19_filepath = "../input/coronavirus-cases-in-india/Covid cases in India.csv"



# Reading the file

indiaCovid19_data = pd.read_csv(indiaCovid19_filepath, index_col="S. No.")
indiaCovid19_data.columns
indiaCovid19_data.dtypes
indiaCovid19_data.describe()
indiaCovid19_data.head()
plt.figure(figsize=(10,6))

plt.title("Number of confirmed cases by state in India")



sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Total Confirmed cases'])



# Add label for vertical axis

plt.ylabel("Number of cases")



# Rotating xlabels so that we can read the text displayed

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light' 

)
# What is the exact number of cases in Maharashtra ?

indiaCovid19_data[indiaCovid19_data['Name of State / UT'] == 'Maharashtra']
plt.figure(figsize=(10,6))

plt.title("Number of Cured/Discharged/Migrated cases by state in India")



sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Cured/Discharged/Migrated'])



# Add label for vertical axis

plt.ylabel("Number of cases")



# Rotating xlabels so that we can read the text displayed

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light' 

)
indiaCovid19_data[indiaCovid19_data['Name of State / UT'] == 'Kerala']
plt.figure(figsize=(10,6))

plt.title("Number of deaths cases by state in India")



sns.barplot(x=indiaCovid19_data['Name of State / UT'], y=indiaCovid19_data['Deaths'])



# Add label for vertical axis

plt.ylabel("Number of cases")



# Rotating xlabels so that we can read the text displayed

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light' 

)
indiaCovid19_data[indiaCovid19_data['Name of State / UT'].isin(['Kerala', 'Delhi', 'Maharashtra', 'Tamil Nadu'])]
sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'])
sns.regplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'])
sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'])
sns.regplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'])
sns.scatterplot(x=indiaCovid19_data['Total Confirmed cases'], 

                y=indiaCovid19_data['Cured/Discharged/Migrated'], 

                hue=indiaCovid19_data['Deaths'])
sns.jointplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Cured/Discharged/Migrated'], kind="kde")
sns.jointplot(x=indiaCovid19_data['Total Confirmed cases'], y=indiaCovid19_data['Deaths'], kind="kde")
sns.jointplot(x=indiaCovid19_data['Deaths'], y=indiaCovid19_data['Cured/Discharged/Migrated'], kind="kde")