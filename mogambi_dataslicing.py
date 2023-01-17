import pandas as pd
url="../input/wiki_movie_plots_deduped.csv"
data=pd.read_csv("../input/wiki_movie_plots_deduped.csv")
#dataframes1
'''
print(data.iloc[0])# first row of data frame
#dataframe2
print(data.iloc[[0,3,6],[0,2,3]])# 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
data.head()
#dataframe3'''
data.set_index('Title')
print(data.loc('Kansas Saloon Smashers'))
#dataframe4
data.set_index('Title','Director')
print(data.loc(['American','Unknown']))
#dataframe5
data.set_index('Title','Director')
print(data.loc(['American','Unknown'],['Cast','Genre']))
















data.head(5)