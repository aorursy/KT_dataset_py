# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file_path= "/kaggle/input/top-covid19-countries-and-health-demographic-trend/top_countries_latest_metric_summary.csv"

data = pd.read_csv(file_path, index_col="Country")



df = pd.read_csv(file_path, index_col="Country")

plt.figure(figsize=(30,10))

metric_key = 'Tobacco consumption'



groupedvalues=df.groupby('Country').sum().reset_index()

groupedvalues=groupedvalues.sort_values([metric_key]).reset_index()

# plt.title("Tobacco consumption(% of population aged 15+ who are daily smokers)")

g=sns.barplot(x='Country',y=metric_key,data=groupedvalues)

g.axes.set_title("Tobacco consumption(% of population aged 15+ who are daily smokers)",fontsize=50)

g.set_xlabel("Country",fontsize=30)

g.set_ylabel("Tobacco consumption",fontsize=40)

g.tick_params(axis="x", labelsize=25, width=2, labelrotation=90)

for index, row in groupedvalues.iterrows():

    g.text(row.name,row[metric_key], round(row[metric_key],2), color='black', fontsize=30, ha="center")



plt.show()
file_path= "/kaggle/input/top-covid19-countries-and-health-demographic-trend/top_countries_latest_metric_summary.csv"

df = pd.read_csv(file_path, index_col="Country")

plt.figure(figsize=(30,10))



groupedvalues=df.groupby('Country').sum().reset_index()

groupedvalues=groupedvalues.sort_values(['Chronic respiratory diseases']).reset_index()

# print(groupedvalues)

title = "Chronic respiratory diseases(DALYs (Disability-Adjusted Life Years), rate per 100k)"

# plt.title("Chronic respiratory diseases(DALYs (Disability-Adjusted Life Years), rate per 100k)")

g=sns.barplot(x='Country',y='Chronic respiratory diseases',data=groupedvalues, palette="Blues")

g.axes.set_title(title,fontsize=50)

g.set_xlabel("Country",fontsize=40)

g.set_ylabel("Chronic respiratory diseases",fontsize=30)

g.tick_params(axis="x", labelsize=25, width=2, labelrotation=90)

for index, row in groupedvalues.iterrows():

    g.text(row.name,row['Chronic respiratory diseases'], round(row['Chronic respiratory diseases'],2), color='black',fontsize=30, ha="center")



plt.show()
file_path= "/kaggle/input/top-covid19-countries-and-health-demographic-trend/top_countries_latest_metric_summary.csv"

data = pd.read_csv(file_path, index_col="Country")



df = pd.read_csv(file_path, index_col="Country")

plt.figure(figsize=(30,10))

metric_key = 'Air pollution'



groupedvalues=df.groupby('Country').sum().reset_index()

groupedvalues=groupedvalues.sort_values([metric_key]).reset_index()

# plt.title()

title = "Air pollution(DALYs (Disability-Adjusted Life Years), rate per 100k)"

g=sns.barplot(x='Country',y=metric_key,data=groupedvalues, palette="Blues")

g.axes.set_title(title,fontsize=50)

g.set_xlabel("Country",fontsize=30)

g.set_ylabel("Air pollution",fontsize=40)

g.tick_params(axis="x", labelsize=25, width=2, labelrotation=90)

for index, row in groupedvalues.iterrows():

    g.text(row.name,row[metric_key], round(row[metric_key],2), color='black', fontsize=30, ha="center")



plt.show()