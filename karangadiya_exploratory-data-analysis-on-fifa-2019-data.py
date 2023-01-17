import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expressions
import matplotlib.pyplot as plt # matplotlib for plotting
import seaborn as sns # seaborn for better graphics
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('../input/data.csv', index_col = 'Unnamed: 0')
# cleaning Wage and Value columns from currency to normal numbers
data['Wage'] = data['Wage'].apply(lambda x: int(re.findall('\d+', x)[0] + '000'))
data['Value'] = data['Value'].apply(lambda x: 'M' in x and int((re.findall('\d+\.*\d*', x)[0] + '000000').replace('.', '')) or int((re.findall('\d+\.*\d*', x)[0] + '000').replace('.', '')))
# getting top ten most popular countries
top_ten_countries = data['Nationality'].value_counts().head(10).index.values
top_ten_countries_data = data.loc[data['Nationality'].isin(top_ten_countries), :]
sns.set(style="white")
plt.figure(figsize=(11, 8))
p = sns.boxplot(x = 'Nationality', y = 'Overall', data = top_ten_countries_data)
sns.set(style="white")
plt.figure(figsize=(11, 8))
p = sns.boxplot(x = 'Nationality', y = 'Potential', data = top_ten_countries_data)
# getting top ten clubs
top_ten_clubs = ['FC Barcelona', 'Real Madrid', 'Manchester City', 'Arsenal', 'Liverpool', 'Manchester United', 'Borussia Dortmund', 'FC Bayern München', 'Juventus', 'Paris Saint-Germain']
top_ten_clubs_data = data.loc[data['Club'].isin(top_ten_clubs), :]
sns.set(style="white")
plt.figure(figsize=(11,8))
p = sns.boxplot(x = 'Club', y = 'Wage', data = top_ten_clubs_data)
p = plt.xticks(rotation=90)
sns.set(style="white")
plt.figure(figsize=(11,8))
p = sns.boxplot(x = 'Club', y = 'Value', data = top_ten_clubs_data)
p = plt.xticks(rotation=90)
p = sns.lineplot(x = 'Age', y = 'Overall', ci = None, data = data, label = 'Overall')
p = sns.lineplot(x = 'Age', y = 'Potential', ci = None, data = data, label = 'Potential')
p = plt.ylabel('Potential vs Overall')
p = plt.legend(loc = 1)
mean_value_per_age = data.groupby('Age')['Value'].mean()
p = sns.barplot(x = mean_value_per_age.index, y = mean_value_per_age.values)
p = plt.xticks(rotation=90)
mean_wage_per_age = data.groupby('Age')['Wage'].mean()
p = sns.barplot(x = mean_wage_per_age.index, y = mean_wage_per_age.values)
p = plt.xticks(rotation=90)
avg_value_by_position = data.groupby('Position')['Value'].mean()
plt.figure(figsize=(11,8))
p = sns.boxplot(x = 'Position', y = 'Value', data = data)
p = plt.xticks(rotation=90)
p = plt.ylim(0, 200000000)
plt.figure(figsize=(50,40))
p = sns.heatmap(data.corr(), annot=True)