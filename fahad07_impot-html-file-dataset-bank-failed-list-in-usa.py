import pandas as pd
df = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
df
df[0]
df[0]['City'].value_counts()
df[0]['City'].value_counts().head(10).plot(kind='bar',figsize=[20,10])

# when we have to find out the particular aplhabetical letter startswith with 
cond = df[0]['City'].str.startswith('W')==True
df[0]['City'][cond].value_counts()
df[0]['Bank Name']
df[0]['Bank Name']
df[0]['Bank Name']
Cond == (df[0][df[0]['ST']=='IL'][['City','Bank Name','Closing Date']])
df[0][df[0]['City']=='Chicago']
cond = ['City','Bank Name','Closing Date']
df[0][df[0]['City']=='Chicago'][cond]
