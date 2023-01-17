import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
happy_one_df = pd.read_csv('../input/world-happiness/2015.csv')

happy_two_df = pd.read_csv('../input/world-happiness/2019.csv')
happy_one_df.head(5)
happy_two_df.head(5)
print(happy_one_df.shape, happy_two_df.shape)
print('There are {} countries in the 2015 dataset and {} countries in the 2019 dataset'.format(happy_one_df.shape[0], happy_two_df.shape[0]))
# Before we go ahead, let's check if any value in the dataframes are null

total_missing_one = happy_one_df.isna().sum()

total_missing_two = happy_two_df.isna().sum()

print(total_missing_one, total_missing_two)

# We can see that there are no null values
print(happy_one_df.columns)

print(happy_two_df.columns)

happy_one_df.drop(columns = ['Standard Error', 'Dystopia Residual'], inplace = True)

happy_one_df.rename(columns = {'Economy (GDP per Capita)':'GDP per capita', 'Health (Life Expectancy)':'Healthy life expectancy', 'Family':'Social support', 'Trust (Government Corruption)':'Perceptions of corruption'}, inplace = True)

happy_two_df.rename(columns = {'Country or region':'Country', 'Score':'Happiness Score', 'Overall rank':'Happiness Rank', 'Freedom to make life choices':'Freedom'}, inplace = True)
common = happy_one_df.merge(happy_two_df, on=['Country'])

result1 = happy_one_df[~happy_one_df.Country.isin(common.Country)]

result2 = happy_two_df[~happy_two_df.Country.isin(common.Country)]

print(result1['Country'])

print(result2['Country'])
happy_two_df['Country'].replace({'Trinidad & Tobago': 'Trinidad and Tobago'}, inplace = True)

happy_one_df['Country'].replace({'North Cyprus': 'Northern Cyprus', 'Somaliland region': 'Somalia', 'Macedonia':'North Macedonia', 'Sudan':'South Sudan'}, inplace = True)
print(happy_one_df.Region.unique(), happy_one_df.Region.nunique())
selected_columns = happy_one_df[["Country","Region"]]

new_df = selected_columns.copy()

new_df
common = happy_two_df.merge(new_df, on=['Country'])

happy_three_df = common.copy()

select_countries1 = happy_two_df.loc[happy_two_df['Country'] == 'Namibia']

select_countries1 = select_countries1.append(happy_two_df.loc[happy_two_df['Country'] == 'Gambia'])

select_countries1['Region'] = ['Sub-Saharan Africa', 'Sub-Saharan Africa']

select_countries1

happy_three_df = pd.concat([happy_three_df, select_countries1])

happy_three_df.sort_values('Happiness Score', ascending = False, inplace = True)

happy_three_df.index = np.arange(0, 156)

happy_three_df
happy_one_df.describe()
happy_three_df.describe()
# Setting some initial parameters for the charts

sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (9, 5)

matplotlib.rcParams['figure.facecolor'] = '#00000000'
# Find the correlation using .corr() and then remove unwanted columns and rows.

corr_df = happy_three_df.corr()

corr_df.drop(columns= ['Happiness Rank', 'Happiness Score'], inplace = True)

corr_df.drop(['Happiness Rank'], inplace = True)

corr_df
sns.heatmap(corr_df, annot=True)
happy_factor_columns = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom', 'Generosity', 'Perceptions of corruption']

matplotlib.rcParams['font.size'] = 20

fig, axes = plt.subplots(3, 2, figsize=(50, 50))



k = 0

for i in range(3):

    for j in range(2):

        column_name = happy_factor_columns[k]

        axes[i,j].set_title('Happiness Score vs. ' + column_name)

        sns.scatterplot(happy_three_df[column_name], happy_three_df['Happiness Score'],

                        hue=happy_three_df['Region'], s=500, ax=axes[i,j]);

        k = k + 1

        axes[i][j].legend(fontsize='20', markerscale=4)
print('There are {} different regions and they are {}'.format(happy_three_df.Region.nunique(), happy_three_df.Region.unique()))
# By grouping the Countries by regions, we can see the number of countries in all ten regions

country_counts_df = happy_three_df.groupby('Region')[['Country']].count()

country_counts_df
region_happy_df = happy_three_df.groupby('Region')[['Happiness Score']].mean().sort_values('Happiness Score', ascending=False)

region_happy_df
# Visualization using a barplot will make it easier to see the amount of difference between the average scores

region_happy_df.reset_index(inplace=True)

sns.set_color_codes("pastel")

matplotlib.rcParams['font.size'] = 10

matplotlib.rcParams['figure.figsize'] = (20, 6)

sns.barplot(x="Region", y="Happiness Score", data=region_happy_df, color="b")
print('From this we can see that based on average Happiness Scores, {} is the happiest Region and {} is the least happiest Region in the world'

      .format(region_happy_df.iloc[0]['Region'], region_happy_df.iloc[-1]['Region']))
# Sort the countries according to the Happiness Score (the initial dataframe is already sorted but we do this just in case and also we need to work on a copy)

country_happy_df = happy_three_df.sort_values('Happiness Score', ascending=False)

country_happy_df.head(5)
print('The most frequent region in the top ten countries is {}'.format(country_happy_df.head(10)['Region'].mode()[0]))

print('The most frequent region in the lowest ten countries is {}'.format(country_happy_df.tail(10)['Region'].mode()[0]))
matplotlib.rcParams['figure.figsize'] = (20, 5)

matplotlib.rcParams['font.size'] = 10

sns.barplot('Region', 'Happiness Score', data=country_happy_df, ci='sd')
print('The happiest country is {} and the least happiest country is {}.'

      .format(country_happy_df.iloc[0]['Country'], country_happy_df.iloc[-1]['Country']))
# We take the first and last row and put it in a new dataframe for easier analysis

compare_df = country_happy_df.tail(1).copy()

compare_df = pd.concat([compare_df, country_happy_df.head(1).copy()])

compare_df.reset_index(drop=True, inplace=True)

compare_df
# We do the following for ease of comparison between the features for both Countries

new_compare_df = compare_df.melt(id_vars = ['Country'])

new_compare_df.drop([0, 1, 16, 17], inplace = True)

new_compare_df.reset_index(drop=True, inplace=True)

# The Happiness Scores for the countries are normalized to values between 0 and 10 so that the other features are more

# comparable in the following barplot

new_compare_df.at[0, 'value'] = new_compare_df.iloc[0]['value'] / 10

new_compare_df.at[1, 'value'] = new_compare_df.iloc[1]['value'] / 10

new_compare_df
sns.barplot(x='variable', y='value', hue='Country', data=new_compare_df);
new_group_df = new_compare_df.groupby('variable')[['value']].diff().dropna()

new_group_df
# Comparing 2015 and 2019

matplotlib.rcParams['figure.figsize'] = (10, 5)

plt.title("Distribution of Happiness Scores in 2015 and 2019")

plt.xlabel('Happiness Score')

plt.ylabel('Number of Countries')

plt.hist(happy_one_df['Happiness Score'], alpha=0.4, bins=[2, 3, 4, 5, 6, 7, 8]);

plt.hist(happy_three_df['Happiness Score'], alpha=0.4, bins=[2, 3, 4, 5, 6, 7, 8]);
high_2019_df = (happy_three_df[happy_three_df['Happiness Score'] > 5])

high_2019_percent = high_2019_df.shape[0] / happy_three_df.shape[0] * 100



low_2019_df = (happy_three_df[happy_three_df['Happiness Score'] <= 5])

low_2019_percent = low_2019_df.shape[0] / happy_three_df.shape[0] * 100



high_2015_df = (happy_one_df[happy_one_df['Happiness Score'] > 5])

high_2015_percent = high_2015_df.shape[0] / happy_one_df.shape[0] * 100



low_2015_df = (happy_one_df[happy_one_df['Happiness Score'] <= 5])

low_2015_percent = low_2015_df.shape[0] / happy_one_df.shape[0] * 100



print('Percent of countries in the higher score range in 2019: {} \nPercent of countries in the lower score range in 2019: {} \nPercent of countries in the higher score range in 2015: {} \nPercent of countries in the lower score range in 2015: {}'

     .format(high_2019_percent, low_2019_percent, high_2015_percent, low_2015_percent))
# Merge the two datasets on the column Country and then find the change in the Happiness Scores

selected_columns = happy_one_df[["Country","Happiness Score"]]

new_df = selected_columns.copy()

merge_df = happy_three_df.merge(new_df, on=['Country'])

merge_df['Score Change'] = merge_df['Happiness Score_x'] - merge_df['Happiness Score_y']

merge_df.head(5)
merge_df['Score Change'].describe()
max_increase = merge_df['Score Change'].idxmax()

max_i_country = merge_df.loc[max_increase]['Country']



max_decrease = merge_df['Score Change'].idxmin()

max_d_country = merge_df.loc[max_decrease]['Country']



print('The country whose happiness increased the most is {}, and the country whose happiness decreased the most is {}.'

     .format(max_i_country, max_d_country))
increase_df_2019 = happy_three_df.loc[happy_three_df['Country'] == max_i_country]

increase_df_2015 = happy_one_df.loc[happy_one_df['Country'] == max_i_country]

increase_df_2019 = pd.concat([increase_df_2019, increase_df_2015])

increase_df_2019
decrease_df_2019 = happy_three_df.loc[happy_three_df['Country'] == max_d_country]

decrease_df_2015 = happy_one_df.loc[happy_one_df['Country'] == max_d_country]

decrease_df_2019 = pd.concat([decrease_df_2019, decrease_df_2015])

decrease_df_2019