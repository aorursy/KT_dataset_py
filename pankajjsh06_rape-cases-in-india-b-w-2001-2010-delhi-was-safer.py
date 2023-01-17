import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")
vic_df = pd.read_csv("../input/20_Victims_of_rape.csv")
state_df =  vic_df[['Area_Name','Victims_of_Rape_Total','Subgroup']]

state_sort = state_df.groupby(['Area_Name','Subgroup'],as_index=False).sum().sort_values('Victims_of_Rape_Total', ascending=False)

state_sort

plt.figure(figsize=(12,8))

sns.barplot(x="Area_Name", y="Victims_of_Rape_Total", hue="Subgroup" ,data=state_sort,palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.title('Total Rape Cases in overall states', fontsize=15)

plt.tight_layout()

plt.show()
state_df =  vic_df[['Year','Area_Name','Victims_of_Rape_Total','Subgroup']]

state_sort = state_df.groupby(['Year','Area_Name'],as_index=False).sum().sort_values('Victims_of_Rape_Total', ascending=False)

plt.figure(figsize=(12,8))

sns.catplot(x="Year", y="Victims_of_Rape_Total", kind="bar", data=state_sort, col="Area_Name", height=5)

plt.xticks(rotation=45,ha='right')

plt.title('Total Rape Cases in overall states', fontsize=15)

plt.tight_layout()

plt.show()
state_df =  vic_df[['Area_Name','Victims_of_Rape_Total']]

state_sort = state_df.groupby(['Area_Name'],as_index=False).sum().sort_values('Victims_of_Rape_Total', ascending=False)

state_sort

plt.figure(figsize=(12,8))

sns.barplot(x="Area_Name", y="Victims_of_Rape_Total", data=state_sort,palette='deep')

plt.xticks(rotation=45,ha='right')

plt.title('Total Rape Cases in overall states', fontsize=15)

plt.tight_layout()

plt.show()
Delhi_df = vic_df[vic_df['Area_Name']=='Delhi']

Delhi_df =  Delhi_df[['Year','Victims_of_Rape_Total']]

Delhi_sort = Delhi_df.groupby(['Year'],as_index=False).sum().sort_values('Victims_of_Rape_Total', ascending=False)

plt.figure(figsize=(12,6))

sns.barplot(x="Year", y="Victims_of_Rape_Total", data=Delhi_sort,palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.title('Total Rape Cases in Delhi(2001-2010)', fontsize=15)

plt.tight_layout()

plt.show()

Delhi_df = vic_df[vic_df['Area_Name']=='Delhi']

Delhi_df =  Delhi_df[['Year','Victims_of_Rape_Total','Subgroup']]

Delhi_df

Delhi_sort = Delhi_df.groupby(['Year','Subgroup'],as_index=False).sum().sort_values('Victims_of_Rape_Total', ascending=False)

plt.figure(figsize=(12,6))

sns.barplot(x="Year", y="Victims_of_Rape_Total", data=Delhi_sort, hue="Subgroup", palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.title('Total Rape Cases in Delhi(2001-2010)', fontsize=15)

plt.tight_layout()

plt.show()
