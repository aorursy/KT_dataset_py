# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
happy_df_2015 = pd.read_csv("../input/2015.csv")

happy_df_2016 = pd.read_csv("../input/2016.csv")

happy_df_2017 = pd.read_csv("../input/2017.csv")

happy_df_2015.head()
print(happy_df_2015.isnull().sum())
print(happy_df_2016.isnull().sum())
print(happy_df_2017.isnull().sum())
happy_df_2015.describe()
happy_df_2015.plot(kind = 'scatter',x = 'Happiness Rank' , y='Trust (Government Corruption)');

plt.title('Happiness rank based on the Corrupted Government!');
happy_df_2016.plot(kind = 'scatter',x = 'Happiness Rank' , y='Trust (Government Corruption)');

plt.title('Happiness rank based on the Corrupted Government!');
happy_df_2017.columns = happy_df_2017.columns.str.replace('.',' ')

happy_df_2017.head()
happy_df_2017.plot(kind = 'scatter',x = 'Happiness Rank' , y='Generosity');

plt.title('Happiness rank based on the Generosity!');
plt.figure(figsize=(18,9))

sns.heatmap(happy_df_2015.corr(),annot=True);

plt.title("Correlation for 2015");

plt.show();
plt.figure(figsize=(18,9))

sns.heatmap(happy_df_2016.corr(),annot=True);

plt.title("Correlation for 2016");

plt.show();
plt.figure(figsize=(18,9))

sns.heatmap(happy_df_2017.corr(),annot=True);

plt.title("Correlation for 2017");

plt.show();
happy_df_2015.head()
score_more_than_7_2015 = happy_df_2015[happy_df_2015['Happiness Score'] > 7]

score_more_than_7_2016 = happy_df_2016[happy_df_2016['Happiness Score'] > 7]

score_more_than_7_2017 = happy_df_2017[happy_df_2017['Happiness Score'] > 7]

plt.figure(figsize=(15,25))

plt.subplot(221);

score_more_than_7_2015.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);

#plt.xticks(rotation=90);

plt.xlabel('Happiness Score');

plt.ylabel('Countries Respectively');

plt.title('Countries by Happiness Score! for year 2015');

plt.subplot(222);

score_more_than_7_2016.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);

#plt.xticks(rotation=90);

plt.ylabel('Happiness Score');

plt.xlabel('Countries Respectively');

plt.title('Countries by Happiness Score! for year 2016');

plt.subplot(223);

score_more_than_7_2017.plot(kind='barh',x = 'Country',y = 'Happiness Score',ax=plt.gca(),legend=None);

#plt.xticks(rotation=90);

plt.ylabel('Happiness Score');

plt.xlabel('Countries Respectively');

plt.title('Countries by Happiness Score! for year 2017');

plt.show();
# Since we have a lot of features to display, we'll slice respectively

we_need_attrs = ['Country','Happiness Score','Economy (GDP per Capita)','Family']

we_need_attrs_2017 = ['Country','Happiness Score','Economy  GDP per Capita ','Family']

compare_main_features_2015 = score_more_than_7_2015.loc[:,we_need_attrs]

compare_main_features_2016 = score_more_than_7_2016.loc[:,we_need_attrs]

compare_main_features_2017 = score_more_than_7_2017.loc[:,we_need_attrs_2017]

compare_main_features_2017

## Now let's plot this variation to see how much each contributes!!.
plt.figure(figsize=(15,25));

plt.subplot(3,1,1);

compare_main_features_2015.plot.barh(stacked=True,x = 'Country',ax=plt.gca());

plt.subplot(3,1,2);

compare_main_features_2016.plot.barh(stacked=True,x = 'Country',ax=plt.gca());

plt.subplot(3,1,3);

compare_main_features_2017.plot.barh(stacked=True,x = 'Country',ax=plt.gca());

plt.show();
compare_main_features_2015.sort_values(by='Economy (GDP per Capita)',inplace=True)

compare_main_features_2016.sort_values(by='Economy (GDP per Capita)',inplace=True)

compare_main_features_2017.sort_values(by='Economy  GDP per Capita ',inplace=True)
plt.figure(figsize=(15,29))

plt.subplot(3,1,1)

compare_main_features_2015.plot(kind='bar',x = 'Country',ax=plt.gca());

plt.title('Visualizing the GDP PER CAPITA for each Country to see the highest contributor and their Happiness Score!')

plt.xticks(rotation=60)

plt.subplot(3,1,2)

compare_main_features_2016.plot(kind='bar',x = 'Country',ax=plt.gca());

plt.xticks(rotation=60)

plt.subplot(3,1,3)

compare_main_features_2017.plot(kind='bar',x = 'Country',ax=plt.gca());

plt.xticks(rotation=60)

plt.show()
compare_main_features_2015['mean score'] = compare_main_features_2015['Happiness Score'].mean()

compare_main_features_2016['mean score'] = compare_main_features_2016['Happiness Score'].mean()

compare_main_features_2017['mean score'] = compare_main_features_2017['Happiness Score'].mean()

plt.figure(figsize=(20,15))

plt.subplot(2,2,1)

compare_main_features_2015[['Happiness Score','mean score']].plot(kind = 'bar',ax=plt.gca());

#plt.ylabel('Happiness Score vs. mean Happiness Score')

# Since we are not getting Countis as our X- Axis labels, let's define a condition for that

condition = ['Country','Happiness Score','mean score']

conditon_df_2015 = compare_main_features_2015[condition]

conditon_df_2015.sort_values(by='Happiness Score',inplace=True);

plt.ylabel('Happiness Score vs. mean Happiness Score')

plt.xticks(rotation=60)

plt.subplot(2,2,2)

conditon_df_2015.plot(kind='bar',x='Country',ax=plt.gca());

conditon_df_2016 = compare_main_features_2016[condition]

conditon_df_2016.sort_values(by='Happiness Score',inplace=True);

plt.ylabel('Happiness Score vs. mean Happiness Score')

plt.xticks(rotation=60)

plt.subplot(2,2,3)

conditon_df_2016.plot(kind='bar',x='Country',ax=plt.gca());

conditon_df_2017 = compare_main_features_2017[condition]

conditon_df_2017.sort_values(by='Happiness Score',inplace=True);

plt.ylabel('Happiness Score vs. mean Happiness Score')

plt.xticks(rotation=60)

plt.subplot(2,2,4)

conditon_df_2017.plot(kind='bar',x='Country',ax=plt.gca());

plt.ylabel('Happiness Score vs. mean Happiness Score')

plt.xticks(rotation=60)

plt.show()
# we will merge 2015 and 2016 Data Frames, as we are focussing on the Regions

merged_df = pd.concat([happy_df_2015,happy_df_2016])
merged_df.head()
merged_df.info()
# We will drop the columns Upper Confidence Interval, Standard Error, Lower Confidence Interval as they are not present in the 2015 data frame.

merged_df = merged_df.drop(['Upper Confidence Interval','Standard Error','Lower Confidence Interval'],1)

merged_df.info()
# Awesome, now let's see how many Regions are present, i.e. unique values

merged_df.Region.nunique()

print(merged_df.Region.value_counts())
# Ok, so we are setting the cutoff of 40, as the entries more than 40, tends to be in somewhat Happy Countries

# i.e. Western Europe,Latin America and Carribbean, Central and Eastern Europe and finally Sub-Saharna Africa.

# We are trying to find out the countries from Regions and there Happiness Rank!

western_europe = merged_df.groupby('Region').get_group('Western Europe')

latin_america_and_caribbean = merged_df.groupby('Region').get_group('Latin America and Caribbean')

central_eastern_europe = merged_df.groupby('Region').get_group('Central and Eastern Europe')

sub_saharan_africa = merged_df.groupby('Region').get_group('Sub-Saharan Africa')
dfs = [western_europe,latin_america_and_caribbean,central_eastern_europe,sub_saharan_africa]

for i in dfs:

    print(i.info())
western_europe = western_europe.sort_values(by='Happiness Score',ascending = False)

latin_america_and_caribbean = latin_america_and_caribbean.sort_values(by='Happiness Score',ascending = False)

central_eastern_europe = central_eastern_europe.sort_values(by='Happiness Score',ascending = False)

sub_saharan_africa = sub_saharan_africa.sort_values(by='Happiness Score',ascending = False)

western_europe.head()
plt.figure(figsize=(20,15))

plt.subplots_adjust(wspace=0.4)

plt.tick_params(labelsize=20)

plt.subplot(2,2,1)

plt.title('Happiness Score for Western Europe Region')

sns.barplot(x = 'Happiness Score',y='Country',data = western_europe);

plt.subplot(2,2,2)

plt.title('Happiness Score for Latin America and Caribbean Region')

sns.barplot(x = 'Happiness Score',y='Country',data = latin_america_and_caribbean);

plt.subplot(2,2,3)

plt.title('Happiness Score for Central Europe Region')

sns.barplot(x = 'Happiness Score',y='Country',data = central_eastern_europe);

plt.subplot(2,2,4)

plt.title('Happiness Score for Sub Saharan Region')

sns.barplot(x = 'Happiness Score',y='Country',data = sub_saharan_africa);
for i in dfs:

    print(i[:3][['Country','Region']])

    print('--------------------------')
norway = merged_df[merged_df.Country == 'Norway']

norway
# No need to panic...we got 2 values as we merged 2 years data and data differed...that's why Norway became happiest country in 2017.
sorted_df = merged_df.sort_values(by = 'Economy (GDP per Capita)',ascending = False)

sorted_df
#sorted_df = sorted_df.set_index('Country').head()
# Now let's find their ranks!

print("Rank of Countries with Highest GDP's")

print(sorted_df[:5][['Happiness Rank','Economy (GDP per Capita)','Country']])
print("Rank of Countries with Lowest GDP's")

print(sorted_df.iloc[-5:][['Happiness Rank','Economy (GDP per Capita)','Country']])
# Now let's see what else can we do, let's have alook at our data frame!

sorted_df.head()
duplicateRowsDF = sorted_df[sorted_df.duplicated(['Country'])]

duplicateRowsDF.head()
sum_df_df = duplicateRowsDF.groupby(['Dystopia Residual','Economy (GDP per Capita)','Family','Freedom','Generosity','Happiness Score',

                           'Health (Life Expectancy)','Trust (Government Corruption)']).sum()

sum_df_df.head()
country_and_rank_df = merged_df[['Happiness Rank','Country']]

mergeing_duplicated_sum = pd.merge(sum_df_df,country_and_rank_df,on = 'Happiness Rank',how='left')

mergeing_duplicated_sum.head(25)
# Let's plot the happiness scores for these countires and end our Kernel EDA!

plt.figure(figsize=(16,9))

top_15 = sorted_df.iloc[:15]

bottom_15 = sorted_df.iloc[-15:]

plt.subplot(1,2,1)

plt.title('Top 15 Countries based on Happiness Score!');

g1 = sns.barplot(x = 'Country' , y = 'Happiness Score',data = top_15);

# this snip tries to provide the count over the bar!

for p in g1.patches:

    g1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xticks(rotation=80);

plt.subplot(1,2,2)

plt.title('Bottom 15 Countries based on Happiness Score!');

g2 = sns.barplot(x = 'Country' , y = 'Happiness Score',data = bottom_15);

# this snip tries to provide the count over the bar!

for p in g2.patches:

    g2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.xticks(rotation=80);