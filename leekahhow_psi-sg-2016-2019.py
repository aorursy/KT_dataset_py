import pandas as pd

psi_df = pd.read_csv("../input/singapore-psi-pm25-20162019/psi_df_2016_2019.csv")

psi_df.head()
# select relevant columns for median and mean calculation

x = ['national', 'south', 'north', 'east', 'central'] 

list_of_df = [psi_df]

for psi_df in list_of_df:

  psi_df['mean']=psi_df[x].mean(axis=1, skipna=True)

  psi_df['median']=psi_df[x].median(axis=1, skipna=True)

psi_df.head(100)