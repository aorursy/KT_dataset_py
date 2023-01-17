import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
meets = pd.read_csv('../input/meets.csv')
print('Rows: %s | Columns: %s' % (meets.shape[0], meets.shape[1]))
meets.sample(3)
opl = pd.read_csv('../input/openpowerlifting.csv')
print('Rows: %s | Columns: %s' % (opl.shape[0], opl.shape[1]))
opl.sample(3)
opl.dtypes
df = pd.merge(meets, opl, on='MeetID')
print('Rows: %s | Columns: %s' % (df.shape[0], df.shape[1]))
df.sample(3)
df.columns
df.dtypes
df[['Age', 'BodyweightKg', 'BestSquatKg', 'BestBenchKg', 'BestDeadliftKg', 'TotalKg', 'Wilks']].describe()
df = df.loc[df['BestSquatKg'] > 0]
df = df.loc[df['BestBenchKg'] > 0]
df = df.loc[df['BestDeadliftKg'] > 0]
print('Rows: %s | Columns: %s' % (df.shape[0], df.shape[1]))
# A really nice visualization technique for looking at missing data
# Taken from: https://www.kaggle.com/juli0703/powerlifters-are-data-scientists-too
plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(),cmap='coolwarm',cbar=False,yticklabels=False)
plt.title('Missing Data',fontsize=20)
plt.show()
df.drop(['Squat4Kg', 'Bench4Kg', 'Deadlift4Kg'], axis=1, inplace=True)
kg = 267.50
kg_to_lb =  float(2.20462262185)
lbs = round((float(kg) * kg_to_lb), 4)
print('%s kg is equal to %s lbs' % (kg, lbs))
print(df['MeetCountry'].unique())
print(df['MeetCountry'].nunique())
region_dict = {}
region_dict['North America'] = ['USA', 'Canada', 'Cayman Islands', 'US Virgin Islands', 'Puerto Rico']
region_dict['Western Europe'] = ['Finland', 'Ireland', 'UK', 'England', 'Italy', 'Germany', 'Spain',
                         'Denmark', 'France', 'Sweden', 'Norway', 'Luxembourg', 'Poland',
                         'Iceland', 'Netherlands', 'N.Ireland', 'Greece', 'Scotland', 'Wales', 'Czechia',
                         'Estonia', 'Belarus']
region_dict['Oceania'] = ['Australia', 'New Zealand', 'New Caledonia']
region_dict['Asia'] = ['Indonesia', 'India', 'Malaysia', 'Japan', 'Singapore']
region_dict['Eastern Europe'] = ['Russia', 'Serbia', 'Ukraine', 'Kazakhstan']
region_dict['South America'] = ['Brazil', 'Peru', 'Colombia', 'Argentina'] 
region_dict['Africa'] = ['South Africa']
region_dict['Middle East'] = ['Israel']
tester = 'USA'
print([region for region in region_dict if tester in region_dict[region]][0])
region_list = []
for x in df['MeetCountry']:
    for region in region_dict:
        if x in region_dict[region]:
            region_list.append(region)
        else:
            pass
df['Region'] = region_list
df.sample(2)
print(df['Division'].unique())
print(df['Division'].nunique())
print(df['Equipment'].unique())
print(df['Equipment'].nunique())
ages = df.loc[df['Age'].notnull()]
ages = round(ages['Age'], 0)
plt.figure(figsize=(12,8))
sns.distplot(ages)
plt.show()
plt.figure(figsize=(12,8))
sns.catplot(x="Equipment", y="BestSquatKg", hue='Sex',
            kind="violin",
            split=True, data=df)
plt.show()
g = sns.catplot(x="Region", y="BestSquatKg", hue='Sex',
            kind="violin", split=True, data=df)
g.fig.set_size_inches(15,10)

best_squat = df['BestSquatKg'].loc[df['BestSquatKg'].notnull()]
sns.distplot(best_squat)
plt.figure(figsize=(12,8))
sns.scatterplot(x='Age', y='BestSquatKg', hue='Sex', data=df)
plt.show()
plt.figure(figsize=(12,8))
sns.scatterplot(x='Age', y='BestSquatKg', hue='Region', data=df)
plt.show()
plt.figure(figsize=(12,8))
sns.scatterplot(x='BodyweightKg', y='BestSquatKg', hue='Region', data=df)
plt.show()
plt.figure(figsize=(12,8))
sns.scatterplot(x='BodyweightKg', y='BestSquatKg', hue='Sex', data=df)
plt.show()
best_bench = df['BestBenchKg'].loc[df['BestBenchKg'].notnull()]
sns.distplot(best_bench)
best_dlift = df['BestDeadliftKg'].loc[df['BestDeadliftKg'].notnull()]
sns.distplot(best_dlift)
bench_squat = df[['BestBenchKg', 'BestSquatKg']].loc[df['BestBenchKg'].notnull() & df['BestSquatKg'].notnull()]
bench_squat.head()
with sns.axes_style("white"):
    g_jtplot = sns.jointplot(x=bench_squat['BestBenchKg'], y=bench_squat['BestSquatKg'], kind="hex", color="k")
g_jtplot.fig.set_size_inches(13,10)
all_lifts = df[['BestBenchKg', 'BestSquatKg', 'BestDeadliftKg']].loc[df['BestBenchKg'].notnull() & 
                                                                         df['BestSquatKg'].notnull() & 
                                                                         df['BestDeadliftKg'].notnull()]
all_lifts.head()
g_pair = sns.pairplot(all_lifts)
g_pair.fig.set_size_inches(13,10)
#g = sns.PairGrid(all_lifts)
#g.map_diag(sns.kdeplot)
#g.map_offdiag(sns.kdeplot, n_levels=6);
df_numeric = df[['Age', 'BodyweightKg', 'BestSquatKg', 'BestBenchKg', 'BestDeadliftKg', 'TotalKg', 
                'Place', 'Wilks']]
df_numeric.head(3)
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(df_numeric)
