import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
%matplotlib inline
df = pd.read_csv('/kaggle/input/psagot2020explorationlecturedata/athlete_events.csv')
df.head()
df.columns
df.Team.head()
df["City"].head()
df[["Games","City"]].head()
df[df.Year == 2006]
df[df.Age > 30]
df[~df.Weight.isnull()]
df[df.Sport.isin(['Basketball', 'Judo'])]
df["BMI"] = 10000 * df.Weight / df.Height.pow(2) 
df.head()
### Enter your code here!

print("row #3: \n")
df.iloc[2]
df.iloc[2:5]
df.loc[1:10][["City", "Year"]]
#vs 
df.loc[1:10,["City", "Year"]]
df.iloc[1:4,0]
df.iloc[1:4,4:8]
tmp_df = pd.DataFrame({'A': [7,7,1,9,9,5]})
tmp_df
time_index = pd.date_range('1/1/1955', periods=6, freq='2h')
time_index
tmp_df = tmp_df.set_index(time_index)
tmp_df
dt = datetime(1955,1,1,4)
tmp_df.loc[dt]
start = datetime(1955,1,1,4)
end = datetime(1955,1,1,8)
tmp_df.loc[start:end]
### enter your code here

print("The average age is {0:.2f}".format(df['Age'].mean()))
print("The std age is {0:.2f}".format(df['Age'].std()))
df.groupby("Sex")[['Age','Weight','Height']].mean()
bins = pd.cut(df.Age, 10)
bins.head()
min_age, max_age = int(df['Age'].min()), int(df['Age'].max()) 
bins = pd.cut(df.Age, range(min_age,max_age+5,5))
bins.head()
df.groupby(bins)['Age'].count()
df.groupby(['Sex',bins])['Height'].mean()
pd.DataFrame(df.groupby(['NOC'])['Medal'].value_counts(sort=False)).head(10)
# Another option: pd.DataFrame(df.groupby(['NOC', 'Medal'])['Medal'].count()).head(10)
### enter your code here

def lower(x):
    return x.lower()
df['City'] = df['City'].apply(lower)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['axes.titlepad'] = 0.7  #change the padding of title from the plots, not necessary
min_age, max_age = int(df['Age'].min()), int(df['Age'].max()) 
df['Age'].hist(by=df.Sex, density=True, bins=range(min_age,max_age+5,5))
plt.suptitle("histogram of age according to gender")
plt.show()
df.groupby('Sex')['Age'].plot.hist(bins=range(min_age,max_age+5,5), alpha=0.3, density=True)
plt.title("Histogram of age according to gender")
plt.legend(); #add legends to plot!
df['Medal'].value_counts(normalize=True).plot.bar()
plt.title("Histogram of medals types");
sports_count = df.groupby(['Season'])['Sport'].value_counts(normalize=True)
sports_count = sports_count[sports_count > 0.03].unstack()
sports_count['Other'] = 1 - sports_count.sum(axis=1)
explode = [0] * len(sports_count.T)
explode[-1] = 0.1 #makes "other" category moves a little bit out, very fancy!
sports_count.T.plot.pie(subplots=True, figsize=(15,7), explode = explode)
plt.suptitle("number of athlets in each sport (spereate season)");
plt.figure(figsize=(10,8))
years_bins = pd.cut(df['Year'], bins= 10)
sport_count_per_year_norm = df.groupby(years_bins)['Sport'].value_counts(normalize=True).unstack().fillna(0)
sns.heatmap(sport_count_per_year_norm.T)
plt.title('sports relative to other sports per year')
plt.figure(figsize=(10,8))
years_bins = pd.cut(df['Year'], bins= 10)
noc_count_per_year_norm = df.groupby(years_bins)['NOC'].value_counts(normalize=True).unstack().fillna(0)
sns.heatmap(noc_count_per_year_norm.T)
plt.title('nocs relative to other nocs per year');
sns.jointplot(x="Year", y="Height", data=df, kind="hex")
# How does the values of 'Season' looks? 
#In what season the Olypmic Games occured? (Yeah you probably knew but you ALWAYS need to check)
df['Season'].value_counts()
# Whats NOC? -> National Olympic Committee(represented by three-letter country codes)
df[df['NOC'].apply(len) != 3]
## Complete your code here

print("num of rows: ", len(df))
print("num of duplicates rows: ", sum(df.duplicated()))
df.drop_duplicates(inplace=True)
print("num of duplicates rows: ", sum(df.duplicated()))
# DO Team's names have hidden duplicates?
df['Team'].value_counts().sort_index().head(15)
df[df['Team'].str.contains('Aldebaran')].groupby('Team')['Year'].value_counts()
df[df.Age.isnull()].head()
df.Age.mean()
a = np.nan
b = 5
a+b
df.isnull().any()
df.isnull().mean() * 100
df[['Height', 'Weight']].isnull().groupby(df['Year']).mean().plot()
plt.title('Null values % of [Height, Weight] through the years');
df['Year'].hist(bins= df['Year'].max() - df['Year'].min())
plt.title('hist of year');
olympics_years_summer = pd.DataFrame(df.loc[df['Season'] == 'Summer', 'Year'].unique(), columns=['Year'])
olympics_years_winter = pd.DataFrame(df.loc[df['Season'] == 'Winter', 'Year'].unique(), columns=['Year'])
olympics_years_winter.sort_values(by='Year').reset_index().diff()['Year'].plot()
plt.title('olympics years winter gap');
olympics_years_summer.sort_values(by='Year').reset_index().diff()['Year'].plot()
plt.title('olympics years summer gap');
df.Medal = df.Medal.fillna("No Medal")
df.Age = df.Age.fillna(df.Age.mean())
#BAD :(
female_mean = df.Age[df.Sex == "F"].mean()
df.loc[df.Sex == "F", 'Age'] = df.loc[df.Sex == "F", 'Age'].fillna(female_mean)
male_mean = df.Age[df.Sex == "M"].mean()
df.loc[df.Sex == "M", 'Age'] = df.loc[df.Sex == "M", 'Age'].fillna(male_mean)

#GOOD :)
df['Age'] = df.groupby("Sex")['Age'].apply(lambda x: x.fillna(x.mean()))
df['Weight'] = df['Weight'].fillna(df.groupby('Year')['Weight'].transform('mean'))
### Replace the code here
df['Height'] = df['Height'].fillna(df.groupby('Year')['Height'].transform('mean'))
df["BMI"] = 10000 * df.Weight / df.Height.pow(2) 
###
df['isHealthy'] = True
df.loc[(df['BMI'] < 18) | (df['BMI'] > 26), 'isHealthy'] = False
df.isnull().any()
df.describe()
df[df['Age'] >= 97]
sns.jointplot(x="Age", y="Weight", data=df)
sns.boxplot(df['Age'], df['Season'])
df['NOC'].value_counts().describe(), df['NOC'].value_counts().plot()
#naive data filling:
noc_to_GDP = pd.read_csv("/kaggle/input/psagot2020explorationlecturedata/dictionary.csv")
noc_to_GDP["Population"] = noc_to_GDP["Population"].fillna(noc_to_GDP["Population"].mean())
noc_to_GDP["GDP per Capita"] = noc_to_GDP["GDP per Capita"].fillna(noc_to_GDP["GDP per Capita"].mean())
noc_to_GDP.head()
df = noc_to_GDP.merge(df, left_on = "Code", right_on = "NOC", how = "right")
df.head()
gdp_per_capita_hist = pd.qcut(df["GDP per Capita"], q=5)
# gdp_per_capita_hist = pd.cut(df["GDP per Capita"], bins=5)

df.groupby(gdp_per_capita_hist)["NOC"].nunique()
df.groupby(gdp_per_capita_hist)['Medal'].value_counts().unstack()
df_medals = df[df['Medal'] != 'No Medal']
count_per_noc = pd.DataFrame(df_medals.groupby('NOC')['Country'].count()).rename(columns={'Country': 'win'})
len_per_noc = pd.DataFrame(df.groupby('NOC')['Country'].count()).rename(columns={'Country': 'win'})
winning_precent = (count_per_noc / len_per_noc).fillna(0)
noc_gdp_count = noc_to_GDP.merge(winning_precent, left_on = "Code", right_on = "NOC", how = "inner")
noc_gdp_count.head()
sns.heatmap(noc_gdp_count.corr())
df_medals = df_medals.set_index('ID')
df_medals['count_medals'] = df_medals.groupby('ID')['Medal'].count()
sns.heatmap(df_medals.corr())
sns.jointplot(x="BMI", y="count_medals", data=df_medals)
sns.set(color_codes=True)

ax = df.groupby([df.Year,df.Sex])['Age'].median().unstack().plot(rot=45, marker='o')
ax.set(ylabel="age median")
plt.title("Age median by Year (for male and female)");
plt.figure(figsize=(10,8))
sns.boxplot(x='Year', y='Age', hue='Sex', data=df)
plt.xticks(rotation=45);
df.loc[df['Medal'] == 'No Medal','Medal'] = np.nan
gdp_to_medals = df.groupby(gdp_per_capita_hist).apply(lambda x: x['Medal'].value_counts()).unstack()
ax = gdp_to_medals.plot(kind='bar', rot = 45)
plt.title("medal count per gdp")
ax.set(ylabel="#medal");
df.loc[df['Medal'].isnull(),'Medal'] = 'No Medal'
num_of_countries_per_gdp = df.groupby(gdp_per_capita_hist)["NOC"].nunique()
ax = num_of_countries_per_gdp.plot.bar(rot = 45)
ax.set(ylabel="num of countries")
plt.title("num of countries on each GDP bin");
num_of_athlete_per_gdp = df.groupby(gdp_per_capita_hist)['ID'].nunique()
ax = num_of_athlete_per_gdp.plot.bar(rot=45)
plt.title("num of athlete per gdp")
ax.set(ylabel="#athlete");
athlete_per_country = num_of_athlete_per_gdp / num_of_countries_per_gdp
ax = (athlete_per_country).plot.bar(rot=45)
plt.title("num of athlete per gdp divided by num of countries for each gdp group")
ax.set(ylabel="#athlete / #countries");
df.loc[df['Medal'] == 'No Medal', 'Medal'] = np.nan
num_of_medals_per_gdp = df.groupby(gdp_per_capita_hist)['Medal'].count()
ax = (num_of_medals_per_gdp / num_of_athlete_per_gdp).plot.bar(rot=45)
plt.title("num of medals per gdp divided by num of athlete for each gdp group")
ax.set(ylabel="#medal / #athlete")
df.loc[df['Medal'].isnull(),'Medal'] = 'No Medal'
print("number of games per season:")
df.groupby([df.Season]).size()
df.groupby([df.Season]).Year.value_counts().sort_index().unstack().T.plot.bar(rot=45, figsize=(10,5))
plt.title("num of records per year(separate season)");
df[df['Sex'] == 'F'].groupby('Year')['Year'].count().plot() ## Bad :(
female_num = df[df['Sex'] == 'F'].groupby('Year')['Year'].count() ## Good :)
(female_num / df.groupby('Year')['Year'].count()).plot()