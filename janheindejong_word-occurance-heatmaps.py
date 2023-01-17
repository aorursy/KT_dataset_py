# read data 
import os
files = os.listdir('../input')
print(files)
# import pandas and read data 
import pandas as pd 
df = pd.read_csv('../input/' + files[0], encoding="ISO-8859-1")
df.describe()
df.head()
# Check occurances per year
df.groupby('Year').size().describe()
# Check formatting of lyrics
df['Lyrics'].loc[0]
# Add wordcount column 
df['Wordcount'] = df['Lyrics'].str.split().str.len()
# Convert to lowercase
df['Lyrics'] = df['Lyrics'].str.lower()
# Drop nan
df = df.dropna()
# Import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Define words of interest
subs_s = 'weed|heroine|cocaine|drugs|marihuana|splif|smack'
alco_s = 'beer|wine|whiskey|alcohol|booze|wodka|moonshine'
reli_s = 'god|jesus|lord|heaven|bible|christ|jerusalem|church'
sex_s = 'sex|suck|penis|pussy|ass|cunt|dick'
# Relative wordcount for certain topics per year
subs = df.groupby('Year').apply(lambda x: x['Lyrics'].str.count(subs_s).sum() / x['Wordcount'].sum())
alco = df.groupby('Year').apply(lambda x: x['Lyrics'].str.count(alco_s).sum() / x['Wordcount'].sum())
reli = df.groupby('Year').apply(lambda x: x['Lyrics'].str.count(reli_s).sum() / x['Wordcount'].sum())
sex_ = df.groupby('Year').apply(lambda x: x['Lyrics'].str.count(sex_s).sum() / x['Wordcount'].sum())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(subs, 'g', label='Drugs')
ax.plot(alco, 'b', label='Alcohol')
ax.plot(reli, 'y', label='Religion')
ax.plot(sex_, 'r', label='Sex')
ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
legend = ax.legend()
# Count relative number of songs containing reference to a certain topic 
subs = df.groupby('Year').apply(lambda x: x['Lyrics'].str.contains(subs_s).sum() / x.shape[0])
alco = df.groupby('Year').apply(lambda x: x['Lyrics'].str.contains(alco_s).sum() / x.shape[0])
reli = df.groupby('Year').apply(lambda x: x['Lyrics'].str.contains(reli_s).sum() / x.shape[0])
sex_ = df.groupby('Year').apply(lambda x: x['Lyrics'].str.contains(sex_s).sum() / x.shape[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(subs, 'g', label='Drugs')
ax.plot(alco, 'b', label='Alcohol')
ax.plot(reli, 'y', label='Religion')
ax.plot(sex_, 'r', label='Sex')
ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
legend = ax.legend()
# Create heatmaps
keys_long = [subs_s, alco_s, reli_s, sex_s]
index = []
data_oc, data_wc = [], []
for keys_short in keys_long: 
    keys = keys_short.split('|')
    for key in keys: 
        key = ' ' + key + ' '
        index.append(key)
        data_oc.append(df.groupby('Year').apply(lambda x: x['Lyrics'].str.contains(key).sum() / x.shape[0] * 100))
        data_wc.append(df.groupby('Year').apply(lambda x: x['Lyrics'].str.count(key).sum() / x['Wordcount'].sum()  * 100))
heatmap_df_oc = pd.DataFrame(data_oc, index=index, columns=df.groupby('Year').groups.keys())
heatmap_df_wc = pd.DataFrame(data_wc, index=index, columns=df.groupby('Year').groups.keys())
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax1 = sns.heatmap(heatmap_df_oc, ax=ax1, cmap='viridis', cbar_kws = {'format': '%.2f%%'})
ax1.set_title('Percentage of songs containing a word')
ax2 = fig.add_subplot(122)
ax2.set_title('Percentage of overall word occurance in a year')
ax2 = sns.heatmap(heatmap_df_wc, ax=ax2, cmap='viridis', cbar_kws = {'format': '%.2f%%'})
# Print a selection of artists containing the word ass
print(list(df.loc[df['Lyrics'].str.contains(' ass ')]['Artist'].sort_values().unique()[:30]))
# Print a selection of artists containing the word god
print(list(df.loc[df['Lyrics'].str.contains(' god ')]['Artist'].sort_values().unique()[:30]))
