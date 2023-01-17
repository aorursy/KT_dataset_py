import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()

nobel = pd.read_csv("../input/archive.csv",parse_dates=True)

nobel.head(n=2)

nobel.info()
# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
display(len(nobel['Prize Share']))

# Display the number of prizes won by male and female recipients.
display(nobel['Sex'].value_counts().head(10))
sex=nobel['Sex'].value_counts()
# Display the number of prizes won by the top 10 nationalities.
ctry=nobel['Birth Country'].value_counts().head(10)
ctry

cat=nobel['Category'].value_counts()

year_cat=nobel.groupby(['Year','Category'])['Laureate ID'].count().reset_index()
year_cat
g = sns.FacetGrid(year_cat, col='Category', hue='Category', col_wrap=4, )
g = g.map(plt.plot, 'Year', 'Laureate ID')
g = g.map(plt.fill_between, 'Year', 'Laureate ID', alpha=0.2).set_titles("{col_name} Category")
g = g.set_titles("{col_name}")
# plt.subplots_adjust(top=0.92)
#g = g.fig.suptitle('Evolution of the value of stuff in 16 countries')
 
plt.show()



# and setting the size of all plots.

plt.rcParams['figure.figsize'] = [13, 7]
sns.barplot(x=sex.index,y=sex.values)
plt.xticks(rotation=90)
plt.title('Nobel Prizes by Sex')
plt.show()
year=nobel['Year'].value_counts()

sns.lineplot(x=year.index,y=year.values,color='red')

plt.xticks(rotation=90)
plt.title('Nobel Prizes by Year')
plt.show()

sns.barplot(x=cat.index,y=cat.values)
plt.xticks(rotation=90)
plt.title('Nobel Prizes by Category')
plt.show()

sns.barplot(x=ctry.index,y=ctry.values)
plt.xticks(rotation=90)
plt.title('Top 10 Countries, which got the nobel prizes the most')
plt.show()
city=nobel['Birth City'].value_counts().head(10)
sns.barplot(x=city.index,y=city.values)
plt.xticks(rotation=90)
plt.title('Top 10 City, in which nobel prize winners born')
plt.show()
# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = nobel['Birth Country']=="United States of America"
nobel['decade'] = (np.floor(nobel['Year']/10)*10).astype(int)
prop_usa_winners = nobel.groupby('decade',as_index=False)['usa_born_winner'].mean()

# Display the proportions of USA born winners per decade
display(prop_usa_winners)
# Plotting USA born winners 
ax = sns.lineplot(data=prop_usa_winners, x='decade',y='usa_born_winner')

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
female=nobel[nobel['Sex']=="Female"].nsmallest(1,'Year')

female[['Year','Category','Full Name','Prize']]

nobel['female_winner'] = np.where(nobel['Sex']=="Female", True, False)

prop_female_winners = nobel.groupby(['decade','Category'],as_index=False)['female_winner'].mean()


ax = sns.lineplot(x='decade', y='female_winner', hue='Category', data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
nobel['male_winner'] = np.where(nobel['Sex']=="Male", True, False)

prop_female_winners = nobel.groupby(['decade','Category'],as_index=False)['male_winner'].mean()


ax = sns.lineplot(x='decade', y='male_winner', hue='Category', data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

repeat=nobel.groupby(['Category','Full Name']).filter(lambda group : len(group)>=2)
#repeat[repeat[['Year','Category','Full Name','Birth Country','Sex']].groupby(['Year','Category'])['Full Name'].nunique().reset_index()]>=2
nobel['Birth Year'] = nobel['Birth Date'].str[0:4]
nobel['Birth Year'] = nobel['Birth Year'].replace(to_replace="nan", value=0)
nobel['Birth Year'] = nobel['Birth Year'].apply(pd.to_numeric)

nobel['Age']=nobel['Year']- nobel['Birth Year']
sns.jointplot(x="Year",
        y="Age",
        kind='reg',
        data=nobel)

plt.show()

sns.boxplot(data=nobel,
         x='Category',
         y='Age')

plt.show()

# Plotting the age of Nobel Prize winners
sns.lmplot('Year','Age',data=nobel,lowess=True, aspect=2,  line_kws={'color' : 'black'})
plt.show()
sns.lmplot('Year','Age',data=nobel,lowess=True, aspect=2, hue='Category')
nobel['D Year'] = nobel['Death Date'].str[0:4]
nobel['D Year'] = nobel['D Year'].replace(to_replace="nan", value=0)
nobel['D Year'] = nobel['D Year'].apply(pd.to_numeric)

nobel['lifespan']=nobel['D Year']- nobel['Birth Year']

sns.boxplot(data=nobel,
         x='Category',
         y='lifespan')

plt.show()
sns.boxplot(data=nobel,
         x='Sex',
         y='lifespan',
           hue='Category')
plt.show()
sns.lmplot('Year','lifespan',data=nobel,lowess=True, aspect=2,  line_kws={'color' : 'black'})
plt.show()
sns.countplot(nobel['Laureate Type'])
plt.show()
# The oldest winner of a Nobel Prize as of 2016
old=nobel.nlargest(5,'Age')
display(old[['Category','Full Name','Birth Country','Sex','Age']])



young=nobel.nsmallest(5,'Age')
display(young[['Category','Full Name','Birth Country','Sex','Age']])

org = nobel['Organization Name'].value_counts().reset_index().head(20)

sns.barplot(x='Organization Name',y='index',data=org)
plt.xticks(rotation=90)
plt.ylabel('Organization Name')
plt.xlabel('Count')
plt.show()