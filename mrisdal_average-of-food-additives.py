import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Read the data
df = pd.read_csv("../input/FoodFacts.csv", encoding="utf8", low_memory=False);

# Convert country names to lowercase
df.countries = df.countries.str.lower()

# Fix some of the names with multiple entries
df.loc[df['countries'] == 'en:fr','countries'] = 'france'
df.loc[df['countries'] == 'en:es','countries'] = 'spain'
df.loc[df['countries'] == 'en:gb','countries'] ='united kingdom'
df.loc[df['countries'] == 'en:uk','countries'] ='united kingdom'
df.loc[df['countries'] == 'españa','countries'] ='spain'
df.loc[df['countries'] == 'us','countries'] = 'united states'
df.loc[df['countries'] == 'en:us','countries'] ='united states'
df.loc[df['countries'] == 'usa','countries'] = 'united states'
df.loc[df['countries'] == 'en:cn','countries'] = 'canada'
df.loc[df['countries'] == 'en:au','countries'] = 'australia'
df.loc[df['countries'] == 'en:de','countries'] ='germany'
df.loc[df['countries'] == 'deutschland','countries'] ='germany'
# Pick some countries
countries = ['france','united kingdom','spain','germany','united states','australia','canada']

# Subset the data
sub_df = df[df.countries.isin(countries)]
sub_df = sub_df[sub_df.additives_n.notnull()]

#print(sub_df["countries"].value_counts())

# Get mean # of additives for each country
df_groupedby = sub_df.groupby(['countries']).mean().additives_n.reset_index()

# Convert to numpy array
df_np = np.array(df_groupedby)

# Sort the data descending by # of additives
df_np = df_np[df_np[:,1].argsort()[::-1]]

# Ready the plot
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
y_pos = np.arange(len(df_np[:,0]))
x_pos = df_np[:,1]
x_ticks = df_np[:,0]

# Make a barplot
plt.bar(y_pos, x_pos, align='center', color='#6cbddf')
plt.title('Average number of additives per product by country')
plt.xticks(y_pos, x_ticks)
plt.ylabel('Average number of additives') 
plt.show()