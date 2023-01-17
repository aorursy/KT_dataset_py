from part1_cleaning import *

df1, df2, df3 = get_clean_data()
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()



hl_compounds = [] # headline compounds

ds_compounds = [] # description/preview compounds



for value in df1["Headlines"].values:

    hl_compounds.append(analyzer.polarity_scores(value)['compound'])

for value in df1["Description"].values:

    ds_compounds.append(analyzer.polarity_scores(value)['compound'])

    

print(hl_compounds[0:10])

print(ds_compounds[0:10])
df1['vs_hl_compounds'] = hl_compounds

df1['vs_ds_compounds'] = ds_compounds

df1
import matplotlib.pyplot as plt

plt.scatter(df1['vs_hl_compounds'].values, df1['vs_ds_compounds'].values)

plt.show()
# storing data

vader_df1 = df1

%store vader_df1
# loading data

%store -r df2
analyzer = SentimentIntensityAnalyzer()



hl_compounds = [] # headline compounds

ds_compounds = [] # description/preview compounds



for value in df2["Headlines"].values:

    hl_compounds.append(analyzer.polarity_scores(value)['compound'])

for value in df2["Description"].values:

    ds_compounds.append(analyzer.polarity_scores(value)['compound'])

    

print(hl_compounds[0:10])

print(ds_compounds[0:10])
df2['vs_hl_compounds'] = hl_compounds

df2['vs_ds_compounds'] = ds_compounds

df2
plt.scatter(df2['vs_hl_compounds'].values, df2['vs_ds_compounds'].values)

plt.show()
# storing data

vader_df2 = df2

%store vader_df2
# loading data

%store -r df3
analyzer = SentimentIntensityAnalyzer()



# since guardian dataset does not include description/preview

hl_compounds = [] # headline compounds



for value in df3["Headlines"].values:

    hl_compounds.append(analyzer.polarity_scores(value)['compound'])

    

print(hl_compounds[0:10])
df3['vs_hl_compounds'] = hl_compounds

df3
plt.hist(df3['vs_hl_compounds'], bins = 50)

plt.show()
# storing data

vader_df3 = df3

%store vader_df3