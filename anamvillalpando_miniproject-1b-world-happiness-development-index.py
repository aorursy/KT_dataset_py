# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import matplotlib.pyplot as plt # Import the library that handles coloring
import matplotlib.patches as pch # Import to add figures
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans # Clustering algorithm
from  matplotlib.ticker import MaxNLocator # Ticker formatter.

# Get our Human Develoment Index dataset.
hdi_data = "/kaggle/input/hdi2019/HDI.csv"
# Get our World Happiness Ladder Score.
whls_data = "/kaggle/input/world-happiness-ranking/Happiness.csv"
# Read my dataset and make if a dataframe called "hdi_df". Use the row one(second row) as headers.
hdi_df = pd.read_csv(hdi_data, header=1)
# Show the result
hdi_df.head()
# Drop all the columns that have 'all' values as NaN.
hdi_df.dropna(axis=1, how='all', inplace=True)
# Show the result
hdi_df.head()
# Drop all the columns but 'HDI Rank (2018)', 'Country' and '2018'.
hdi_df = hdi_df[['HDI Rank (2018)','Country', '2018']]
# Rename the 'HDI Rank (2018)' column to just 'Rank' to make it easier to handle, and rename the '2018' column to 'HDI'.
hdi_df.rename(columns={'HDI Rank (2018)':'Rank', '2018':'HDI'}, inplace=True)
# Show the result.
hdi_df.head()
# First, remove the rows where 'Rank' is NaN,
hdi_df.dropna(axis=0, subset=['Rank'], inplace=True)
# Convert the 'Rank' to numberic to that we can sort the dataframe based on it as numbers. Any parsing error should be coerced(converted to NaN)..
hdi_df['Rank'] = pd.to_numeric(hdi_df['Rank'], errors='coerce')
# Sort our values using the 'Rank' column and save it 'inplace'.
hdi_df.sort_values(by='Rank', inplace=True)
# Show the tail to see if the conversion left any NaNs.
hdi_df.tail()
# Save the regions to a new dataframe. The regions are the rows that contain null. 
regions_df = hdi_df[hdi_df['Rank'].isnull()]
# Print the regions dataframe to make sure we only got regions and not countries.
print(regions_df)
# Remove the regions from our hdi dataframe by grabing the notnull ones.
hdi_df = hdi_df[hdi_df['Rank'].notnull()]
# Cast the 'Rank' column to integer.
hdi_df['Rank'] = hdi_df['Rank'].astype(int)
# Change the Index to be the 'Coutry'
hdi_df.set_index('Country',inplace=True)
# Show the result.
hdi_df.head()
# Read my dataset and make if a dataframe called "whls_df".
whls_df = pd.read_csv(whls_data)
# Show the result.
whls_df.head()
# Drop all the columns but 'Ladder score' and 'Country name'.
whls_df = whls_df[['Ladder score','Country name']]
# Rename the 'Ladder score' column to 'WHLS', 'Country name' to just 'Country'
whls_df.rename(columns={'Ladder score':'WHLS', 'Country name':'Country'}, inplace=True)
# Show the result.
whls_df.head()
# Sort our values using the 'WHLS' column in descending order and save it inplace.
whls_df.sort_values('WHLS', ascending=False, inplace=True)
# Calculate the 'Rank'
whls_df['Rank'] = range(1,len(whls_df)+1)
#
# Show the result.
whls_df.head()
# Get the column list.
cols = list(whls_df.columns)
# Get the column indexes.
a, b, c = cols.index('WHLS'), cols.index('Country'), cols.index('Rank')
# Rearrenge the columns in our cols list.
cols[c], cols[b], cols[a] = cols[a], cols[b], cols[c]
# Overwrite the dataframe to save the changes.
whls_df = whls_df[cols]
# Change the Index to be the 'Coutry'
whls_df.set_index('Country',inplace=True)
# Show the result.
whls_df.head()
# Show the 10 happiest contries in the world.
whls_df.head(10)
# Sort our values using the 'Rank' column in descending order to get the least happy top-down.
whls_df_temp = whls_df.sort_values('Rank', ascending=False)
# Show the top 10 which will now be the least happy contries in the world.
whls_df_temp.head(10)
# Show the 10 most developed countries in the world.
hdi_df.head(10)
# Sort our values using the 'Rank' column in descending order to get the least developed top-down.
hdi_df_temp = hdi_df.sort_values('Rank', ascending=False)
# Show the top 10 which will now be the least developed contries in the world.
hdi_df_temp.head(10)
# Merge the dataframes of both the HDI and the WHLS using an outer join on the index.
outer_country_df = hdi_df.merge(whls_df, right_index=True, left_index=True, how='outer',
          suffixes=('_hdi', '_whls'))
# Check the result.
print(outer_country_df)

# Get the missing Countries(index) which would end up have the Ranks(either) as Null after the merge.
missing_in_hdi = outer_country_df[outer_country_df['Rank_hdi'].isnull()].index
missing_in_whls = outer_country_df[outer_country_df['Rank_whls'].isnull()].index
# Convert the sequences resulting from it to a list.
missing_in_hdi = missing_in_hdi.tolist()
missing_in_whls = missing_in_whls.tolist()

# Print the results.
print("Missing in HDI:")
print(missing_in_hdi)
print("\nMissing in WHLS:")
print(missing_in_whls)
# Make our lists of names to be change on each of the studies.
hdi_renamed_indexes = {'Czechia': 'Czech Republic',
                       'Hong Kong, China (SAR)': 'Hong Kong',
                       'Korea (Republic of)': 'South Korea',
                       'Bolivia (Plurinational State of)': 'Bolivia',
                       'Viet Nam': 'Vietnam',
                       'Russian Federation': 'Russia',
                       'Iran (Islamic Republic of)': 'Iran',
                       'North Macedonia': 'Macedonia',
                       'Congo': 'Congo (Brazzaville)',
                       'Eswatini (Kingdom of)': 'Swaziland',
                       "Lao People's Democratic Republic": 'Laos',
                       'Congo (Democratic Republic of the)': 'Congo (Kinshasa)',
                       "Côte d'Ivoire": 'Ivory Coast',
                       'Venezuela (Bolivarian Republic of)': 'Venezuela',
                       'Palestine, State of': 'Palestine',
                       'Tanzania (United Republic of)': 'Tanzania',
                       'Moldova (Republic of)': 'Moldova',
                       'Micronesia (Federated States of)': 'Micronesia',
                       'Syrian Arab Republic': 'Syria',
                       'Brunei Darussalam': 'Brunei',
                       'Timor-Leste': 'East Timor'}
whls_renamed_indexes = {"Palestinian Territories":'Palestine',
                        'Hong Kong S.A.R. of China':'Hong Kong',
                        'Taiwan Province of China':'Taiwan'}
# Rename the indexes using the lists we created.
hdi_df.rename(index=hdi_renamed_indexes, inplace=True)
whls_df.rename(index=whls_renamed_indexes, inplace=True)
# Print the results.
print(hdi_df)
print(whls_df)
# Re-Merge the dataframes of both the HDI and the WHLS using an outer join on the index now that the names are fixed.
outer_country_df = hdi_df.merge(whls_df, right_index=True, left_index=True, how='outer',
          suffixes=('_hdi', '_whls'))

# Get the missing Countries(index) which would end up have the Ranks(either) as Null after the merge.
missing_in_hdi = outer_country_df[outer_country_df['Rank_hdi'].isnull()].index
missing_in_whls = outer_country_df[outer_country_df['Rank_whls'].isnull()].index
# Convert the sequences resulting from it to a list.
missing_in_hdi = missing_in_hdi.tolist()
missing_in_whls = missing_in_whls.tolist()

# Print the results.
print("Missing in HDI:")
print(missing_in_hdi)
print("\nMissing in WHLS:")
print(missing_in_whls)
# Remove the list of missing countries in WHLS from HDI.
hdi_df.drop(missing_in_whls, inplace=True)
# Sort our values using the 'HDI' column in descending order and save it inplace.
hdi_df.sort_values('HDI', ascending=False, inplace=True)
# Re-calculate the 'Rank'.
hdi_df['Rank'] = range(1,len(hdi_df)+1)
# Print the result for cofirmation
print(hdi_df)

# Remove the list of missing countries in HDI from WHLS.
whls_df.drop(missing_in_hdi, inplace=True)
# Sort our values using the 'WHLS' column in descending order and save it inplace.
whls_df.sort_values('WHLS', ascending=False, inplace=True)
# Re-calculate the 'Rank'.
whls_df['Rank'] = range(1,len(whls_df)+1)
# Print the result for cofirmation
print(whls_df)
# Inner merge using the both right and left index as matching value. We then add suffixes to the Ranks of the studies for differentiation.
inner_country_df = hdi_df.merge(whls_df, right_index=True, left_index=True, suffixes=('_hdi', '_whls'))
# Extract the countries that are in both the top 20 of HDI and top 20 of WHLS. We make sure we create a copy of the view so that we don't alter the joined dataframe because it will be used later.
both_top_df = inner_country_df[(inner_country_df['Rank_hdi'] <= 20) & (inner_country_df['Rank_whls'] <= 20)].copy()
# Drop the HDI and WHSL columns because we already used them to re-calculate the Ranks so we don't need them for these questions anymore.
both_top_df.drop(columns=['HDI', 'WHLS'], inplace=True)
# Sort our values using the index(Country) in ascending order(alphabetical) and save it inplace.
both_top_df.sort_index(inplace=True)
# Print the result to verify.
print(both_top_df)

# Now create a bar plot using 'ggplot' style to be able to see the countries that match in the top 20 visually.
with plt.style.context("ggplot"):
    # Plot a bar chart, rotating the ticks and increasing the figure size.
    both_top_df.plot(kind='bar', rot=90, figsize=(15, 15))
    # Make sure we force the Y Axis to shot the values as integers.
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # We also going to include all the ticks from the min (0) and the max(20).
    plt.yticks(np.arange(min(both_top_df['Rank_hdi']), max(both_top_df['Rank_hdi'])+1, 1.0))
    # Finally rename the rank lengeds so that we don't show the undescore.
    plt.legend(["HDI Rank", "WHLS Rank"])
    # Shot the Y Axis label.
    plt.ylabel("Rank")
# Show the plot
plt.show()
# Extract the countries that are in both the bottom 20 of HDI and bottom 20 of WHLS. We make sure we create a copy of the view so that we don't alter the joined dataframe because it will be used later.
both_bot_df = inner_country_df[(inner_country_df['Rank_hdi'] > 130) & (inner_country_df['Rank_whls'] > 130)].copy()
# Drop the HDI and WHSL columns because we already used them to re-calculate the Ranks so we don't need them for these questions anymore.
both_bot_df.drop(columns=['HDI', 'WHLS'], inplace=True)
# Sort our values using the index(Country) in ascending order(alphabetical) and save it inplace.
both_bot_df.sort_index(inplace=True)
# Print the result to verify.
print(both_bot_df)

# Now create a bar plot using 'ggplot' style to be able to see the countries that match in the top 20 visually.
with plt.style.context("ggplot"):
    # Plot a bar chart, rotating the ticks and increasing the figure size.
    both_bot_df.plot(kind='bar', rot=90, figsize=(15, 15))
    # Make sure we force the Y Axis to shot the values as integers.
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # We also going to include all the ticks from the min (131) and the max(150).
    plt.yticks(np.arange(min(both_bot_df['Rank_whls'])-1, max(both_bot_df['Rank_whls'])+1, 1.0))
    # Set the limits to (131,150) so that we only show the last few ranks.
    plt.gca().set_ylim(bottom=131, top=150)
    # Finally rename the rank lengeds so that we don't show the undescore.
    plt.legend(['HDI Rank', 'WHLS Rank'])
    # Shot the Y Axis label.
    plt.ylabel("Rank")
# Show the plot
plt.show()
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(15, 15))
    # Create the scatter using HDI and WHLS as x and y axis.
    plt.scatter(inner_country_df['Rank_hdi'],inner_country_df['Rank_whls']); 
    # This line will represent the spot where the Ranks are the same. The closer the Countries are to it, the more related HDI and WHLS are.
    plt.plot([inner_country_df['Rank_hdi'].min(), inner_country_df['Rank_hdi'].max()],[inner_country_df['Rank_whls'].min(), inner_country_df['Rank_whls'].max()],
             "r--", label='Same Rank', color='Black')
    # Show the legend
    plt.legend()
    # Set the X Axis lavel
    plt.xlabel("HDI Rank")
    # Set the X Axis lavel
    plt.ylabel("WHLS Rank")
    # Invert the axis so that the top ones show up on the top-right corner.
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
# Shot the plot
plt.show()
# Calculate the absolute value of the different between the ranks to see how far appart from each other they are.
inner_country_df['Rank_Diff'] = abs(inner_country_df['Rank_hdi'] - inner_country_df['Rank_whls'])
# Print the result DataFrame to see how it looks.
print(inner_country_df)
# Create a histogram plot using 'ggplot' style to be able to see how the rank differences are distributed.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(15, 15))
    # Plot the histogram using "Rank_Diff" as X, with 14 bins and bar style. Get the return values because we will use them later to set the style.
    n, bins, patches = plt.hist(inner_country_df['Rank_Diff'], bins=14, histtype='bar')
    # Append an extra element(0) to the n array.
    n = np.append(n,[0])
    # Plot a line with points in every (bin,n) to show how the distribution moves.
    plt.plot(bins, n, 'r--', color='Black')
    # Set the X Axis label.
    plt.xlabel("Rank Difference")
    # We also going to include every 5 ticks from 0 to 80.
    plt.xticks(np.arange(0, 80, 5.0))
    # Set the Y Axis label.
    plt.ylabel("Frequency(Number of Countries)")
    # Include every 3 ticks from 0 to 35.
    plt.yticks(np.arange(0, 35, 3.0))
    
    # Get the color map using the 'nipy_spectral' palette for the amount of patches times 2(to get colors from the center[green] to the right[red]) 
    colors = plt.get_cmap('nipy_spectral', len(patches)*2)
    
    # Set the color of the patches iterating through all the patches and setting their color using our palette.
    for i in range(len(patches)):
        # i+length-1 to make sure we start from the middle of the palette to the end.
        patches[i].set_facecolor(colors(i+len(patches)-1))
# Show the plot
plt.show()
# Create kmeans object with 7 clusters
kmeans = KMeans(n_clusters=7)
# Get our scatter plot points
points = inner_country_df[['Rank_hdi','Rank_whls']]
# Fit k-means object to data
kmeans.fit(points)
# Create a scatter plot using 'ggplot' style to be able to see how the rank differences are distributed.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(15, 15))
    # Create out scatter plot using the ranks as Axis and setting the color map to the KMeans cluters.
    plt.scatter(points['Rank_hdi'],points['Rank_whls'], c=kmeans.labels_);
    # This line will represent the spot where the ranks are the same. The closer the Countries are to it, the more related HDI and WHLS are.
    plt.plot([points['Rank_hdi'].min(), points['Rank_hdi'].max()],[points['Rank_whls'].min(), points['Rank_whls'].max()], "r--", label='Same Rank')
    # Make sure we show the center line legend.
    plt.legend()
    # Add the axis labels
    plt.xlabel("HDI Rank")
    plt.ylabel("WHLS Rank")
    
    # Invert the axis to have the top ones on the top-right corner.
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    # Add the country names(index) by annotating them next to each of the points in the plot.
    for i, txt in enumerate(inner_country_df.index.tolist()):
        plt.gca().annotate(txt, (points['Rank_hdi'][i], points['Rank_whls'][i]))
# Show the plot
plt.show()
# Get the countries with HDI Rank lower than 50 and WHLS higher than 25. 
low_hdi_high_whls_df = inner_country_df[(inner_country_df['Rank_hdi'] > 50) & (inner_country_df['Rank_whls'] < 25)].copy()
# Remove the columns that are not necessary.
low_hdi_high_whls_df.drop(columns=['HDI', 'WHLS'], inplace=True)
# Print the result
print("High WHLS/Low HDI:\n")
print(low_hdi_high_whls_df)

# Get the countries with WHLS Rank lower than 50 and HDI higher than 25. 
high_hdi_low_whls_df = inner_country_df[(inner_country_df['Rank_hdi'] < 25) & (inner_country_df['Rank_whls'] > 50)].copy()
# Remove the columns that are not necessary.
high_hdi_low_whls_df.drop(columns=['HDI', 'WHLS'], inplace=True)
# Print the result
print("\nHigh HDI/Low WHLS:\n")
print(high_hdi_low_whls_df)
# Create a scatter plot using 'ggplot' style to be able to see how the rank differences are distributed.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(15, 15))
    # Create out scatter plot using the ranks as Axis and setting the color map to the KMeans cluters.
    plt.scatter(points['Rank_hdi'],points['Rank_whls'], c=kmeans.labels_);
    # This line will represent the spot where the ranks are the same. The closer the Countries are to it, the more related HDI and WHLS are.
    plt.plot([points['Rank_hdi'].min(), points['Rank_hdi'].max()],[points['Rank_whls'].min(), points['Rank_whls'].max()], "r--", label='Same Rank')
    
    # Make sure we show the center line legend.
    plt.legend()
    
    # Add the axis labels
    plt.xlabel("HDI Rank")
    plt.ylabel("WHLS Rank")
        
    # Invert the axis to have the top ones on the top-right corner.
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    # Add the country names(index) by annotating them next to each of the points in the plot.
    for i, txt in enumerate(inner_country_df.index.tolist()):
        plt.gca().annotate(txt, (points['Rank_hdi'][i], points['Rank_whls'][i]))
    
    # Add a rectangle to highlight the entire scatter plot area, this one will be used to identify the top and bottom and their outliers.
    plt.gca().add_patch(pch.Rectangle((-2,-2),155,155,linewidth=2,edgecolor='blue',facecolor='cyan', alpha=0.1))
    
    # Add another rectangle to obscure the center ones. We are interested in the top and bottom outliers, not the center.
    plt.gca().add_patch(pch.Rectangle((25,25),102,102,linewidth=2,edgecolor='red',facecolor='black', alpha=0.9))
    
    # Add 2 more rectangles, one for the top 25/25 and one from the bottom 25/25.
    plt.gca().add_patch(pch.Rectangle((-1,-1),26,26,linewidth=2,edgecolor='Green',facecolor='none'))
    plt.gca().add_patch(pch.Rectangle((152,152),-25,-25,linewidth=2,edgecolor='Blue',facecolor='none'))
    
    # These purple squares highlight the outliers.
    plt.gca().add_patch(pch.Rectangle((22,57),-20,20,linewidth=2,edgecolor='Purple',facecolor='none'))
    plt.gca().add_patch(pch.Rectangle((45,26),20,-20,linewidth=2,edgecolor='Purple',facecolor='none'))
    plt.gca().add_patch(pch.Rectangle((151,95),-20,20,linewidth=2,edgecolor='Purple',facecolor='none'))
    plt.gca().add_patch(pch.Rectangle((77,130),20,20,linewidth=2,edgecolor='Purple',facecolor='none'))
    
# Show the plot
plt.show()