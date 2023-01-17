import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df1 = pd.read_csv('../input/meets.csv')
df1.shape
df1.head(2)
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Month'] = df1['Date'].apply(lambda x:x.month)
df1['Year'] = df1['Date'].apply(lambda x:x.year)
plt.figure(figsize=(10,7))
df1['MeetCountry'].value_counts()[:10].sort_values(ascending=True).plot(kind='barh')
plt.title('Meets by Country\n',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print('Meets by Country:\n')
print(df1['MeetCountry'].value_counts()[:10])
#Current country populations, as of April 12, 2018

usPop = 325700000
norwayPop = 5230000
canadaPop = 36290000
aussiePop = 24130000
kiwiPop = 4690000


perCapDict = {'United States': len(df1[df1['MeetCountry']=='USA']) / usPop * 100000,
            'Norway': len(df1[df1['MeetCountry']=='Norway']) / norwayPop * 100000,
            'Canada': len(df1[df1['MeetCountry']=='Canada']) / canadaPop * 100000,
            'Australia': len(df1[df1['MeetCountry']=='Australia']) / aussiePop * 100000,
            'New Zealand': len(df1[df1['MeetCountry']=='New Zealand']) / kiwiPop * 100000}

perCapDf = pd.Series(perCapDict)
plt.figure(figsize=(10,5))
perCapDf.plot(kind='barh')
plt.title('Meets per 100,000 People\n',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print('Meets per 100,000 People\n')
for k,v in perCapDict.items():
    print(k[:6] + ': ',(round(v,2)))
plt.figure(figsize=(10,6))
df1.groupby(['Month'])['Month'].count().plot(kind='bar')
plt.title('World Meets by Month\n',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(df1.groupby(['Month'])['Month'].count())
plt.figure(figsize=(14,6))
df1.groupby(['Year'])['Year'].count().plot(kind='bar')
plt.title('World Meets by Year\n',fontsize=20)
plt.xlabel('Year',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
df = pd.read_csv('../input/openpowerlifting.csv')
df.shape
df.head()
plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(),cmap='cool',cbar=False,yticklabels=False)
plt.title('Missing Data?',fontsize=20)
plt.show()
df.drop(['Squat4Kg','Bench4Kg','Deadlift4Kg'],axis=1,inplace=True)
df.head()
df['Name'].value_counts()[:10]
print('Number of unique divisions: ' + str(df['Division'].nunique()))
def age_class(x):
    if x < 13:
        return 'CHILD'
    if x >= 13 and x <= 17:
        return 'YOUTH'
    if x >= 18 and x <= 34:
        return 'ADULT'
    if x >= 35:
        return 'MASTERS'
df['AgeClass'] = df['Age'].apply(age_class)
def squatBody(x):
    return x['BestSquatKg'] / x['BodyweightKg']

def benchBody(x):
    return x['BestBenchKg'] / x['BodyweightKg']

def deadliftBody(x):
    return x['BestDeadliftKg'] / x['BodyweightKg']

def totalLiftBody(x):
    return x['TotalKg'] / x['BodyweightKg']
df['Squat / BW'] = df.apply(squatBody,axis=1)
df['Bench / BW'] = df.apply(benchBody,axis=1)
df['Deadlift / BW'] = df.apply(deadliftBody,axis=1)
df['Total / BW'] = df.apply(totalLiftBody,axis=1)
male = df[df['Sex']=='M']
female = df[df['Sex']=='F']
def male_weight_class(x):
    if x <= 56:
        return '56 Kg'
    if x <= 62 and x > 56:
        return '62 Kg'
    if x <= 69 and x > 62:
        return '69 Kg'
    if x <= 77 and x > 69:
        return '77 Kg'
    if x <= 85 and x > 77:
        return '85 Kg'
    if x <= 94 and x > 85:
        return '94 Kg'
    if x <= 105 and x > 94:
        return '105 Kg'
    if x > 105:
        return '105+ Kg'
        
def female_weight_class(x):
    if x <= 48:
        return '48 Kg'
    if x <= 53 and x > 48:
        return '53 Kg'
    if x <= 58 and x > 53:
        return '58 Kg'
    if x <= 63 and x > 58:
        return '63 Kg'
    if x <= 69 and x > 63:
        return '69 Kg'
    if x <= 75 and x > 69:
        return '75 Kg'
    if x <= 90 and x > 75:
        return '90 Kg'
    if x > 90:
        return '90+ Kg'
male['WeightClassKg'] = male['BodyweightKg'].apply(male_weight_class)
female['WeightClassKg'] = female['BodyweightKg'].apply(female_weight_class)
df = pd.concat([male,female])
df[df['BestBenchKg']<0].head(3)
negative_lifts = len(df[(df['BestBenchKg']<0) | (df['BestDeadliftKg']<0) | (df['BestSquatKg']<0)]) / len(df)

print('Percent chance of one or more lift values to be negative: {}%'.format(negative_lifts*100))
#Make sure to only drop negatives. We want to keep NaN values.

df = df[(df['BestSquatKg'] > 0) | (df['BestSquatKg'].isnull() == True)]
df = df[(df['BestDeadliftKg'] > 0) | (df['BestDeadliftKg'].isnull() == True)]
df = df[(df['BestBenchKg'] > 0) | (df['BestBenchKg'].isnull() == True)]
plt.figure(figsize=(10,7))
df['Sex'].value_counts().plot(kind='bar')
plt.title('Gender Count',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('Percentage of Male lifters: {}%\n'.format(round(len(df[df['Sex']=='M'])/len(df)*100),4))
print(df['Sex'].value_counts())
print(df['Equipment'].value_counts())
#Convert all 'Straps' instances into 'Wraps' instances.

def convert_equipment(x):
    if x == 'Straps':
        return 'Wraps'
    return x
df['Equipment'] = df['Equipment'].apply(convert_equipment)
plt.figure(figsize=(10,7))
df['Equipment'].value_counts().plot(kind='bar')
plt.title('Equipment Used',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('Equipment used: \n')
print(df['Equipment'].value_counts())
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'Age',bins=50,alpha=.5)
plt.title('Lifters Age',fontsize=20)
plt.legend(loc=1)
plt.show()
df[df['Age']==5]
df[df['Age']==95]
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'BodyweightKg',bins=50,alpha=.6)
plt.title('Bodyweight Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.legend(loc=1)
plt.show()
sns.lmplot(x='BodyweightKg',
           y='BestSquatKg',
           data=df.dropna(),
           hue='Equipment',
           markers='x',
           size=7,
           aspect=2)
plt.title('Best Squat by Equipment Used',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('BestSquatKg',fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.show()
print('Equipment Used by Lifters:\n')
print(df['Equipment'].dropna().value_counts())
sns.lmplot(x='BodyweightKg',
           y='BestBenchKg',
           data=df.dropna(),
           hue='Equipment',
           markers='x',
           size=7,
           aspect=2)
plt.title('Best Bench by Equipment Used',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('BestBenchKg',fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.show()
print('Equipment Used by Lifters:\n')
print(df['Equipment'].dropna().value_counts())
sns.lmplot(x='BodyweightKg',
           y='BestDeadliftKg',
           data=df.dropna(),
           hue='Equipment',
           markers='x',
           size=7,
           aspect=2)
plt.title('Best Deadlift by Equipment Used',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('BestDeadliftKg',fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.show()
print('Equipment Used by Lifters:\n')
print(df['Equipment'].dropna().value_counts())
sns.lmplot(x='BodyweightKg',
           y='TotalKg',
           data=df.dropna(),
           hue='Equipment',
           markers='x',
           size=7,
           aspect=2)
plt.title('TotalKg by Equipment Used',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('TotalKg',fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.show()
print('Equipment Used by Lifters:\n')
print(df['Equipment'].dropna().value_counts())
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'BestSquatKg',bins=50,alpha=.6)
plt.title('Best Squat Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BestSquatKg',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'BestBenchKg',bins=50,alpha=.6)
plt.title('Best Bench Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BestBenchKg',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'BestDeadliftKg',bins=50,alpha=.6)
plt.title('Best Deadlift Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BestDeadliftKg',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'TotalKg',bins=50,alpha=.6)
plt.title('Total Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('TotalKg',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'Squat / BW',bins=50,alpha=.6)
plt.title('Times Bodyweight Squated',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Times Bodyweight Squated',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'Bench / BW',bins=50,alpha=.6)
plt.title('Times Bodyweight Benched',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Times Bodyweight Benched',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'Deadlift / BW',bins=50,alpha=.6)
plt.title('Times Bodyweight Deadlifted',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Times Bodyweight Deadlifted',fontsize=15)
plt.legend(loc=1)
plt.show()
g = sns.FacetGrid(df,hue='Sex',size=6,aspect=1.5,legend_out=True)
g = g.map(plt.hist,'Total / BW',bins=50,alpha=.6)
plt.title('Times Total Bodyweight Lifted',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Times Total Bodyweight Lifted',fontsize=15)
plt.legend(loc=1)
plt.show()
plt.figure(figsize=(9,6))
df[df['Squat / BW']>=5]['Name'].value_counts()[:10].sort_values(ascending=True).plot(kind='barh')
plt.title('Times Bodyweight Squated (5 x BW)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(df[df['Squat / BW']>=5]['Name'].value_counts()[:10])
plt.figure(figsize=(9,6))
df[df['Bench / BW']>=4]['Name'].value_counts()[:10].sort_values(ascending=True).plot(kind='barh')
plt.title('Times Bodyweight Benched (4 x BW)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(df[df['Bench / BW']>=4]['Name'].value_counts()[:10])
plt.figure(figsize=(9,6))
df[df['Deadlift / BW']>=4.5]['Name'].value_counts()[:10].sort_values(ascending=True).plot(kind='barh')
plt.title('Times Bodyweight Deadlifted (4.5 x BW)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(df[df['Deadlift / BW']>=4.5]['Name'].value_counts()[:10])
df[df['Deadlift / BW']>=5]
plt.figure(figsize=(9,6))
df[df['Total / BW']>=13]['Name'].value_counts()[:10].sort_values(ascending=True).plot(kind='barh')
plt.title('Times Bodyweight Totaled (13 x BW)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(df[df['Total / BW']>=13]['Name'].value_counts()[:10])
sns.lmplot(data=df.dropna(),
           x='BodyweightKg',
           y='Total / BW',
           hue='Sex',
           size=7)
plt.title('Relative Strength to Bodyweight',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Total / BW',fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.show()
df.columns
df.filter(items=['Name','BodyweightKg','TotalKg','Total / BW','Wilks','Sex'])[df['Sex']=='F'].sort_values(ascending=False,by='Wilks')[:10]
df.filter(items=['Name','BodyweightKg','TotalKg','Total / BW','Wilks','Sex'])[df['Sex']=='M'].sort_values(ascending=False,by='Wilks')[:10]
wilksByName = df.filter(['Name','Sex','BodyweightKg','WeightClassKg','Wilks']).sort_values(ascending=False,by='Wilks')
sns.lmplot(x = 'BodyweightKg',
           y = 'Wilks',
           data = wilksByName.dropna(),
           hue = 'Sex',
           size=7)
plt.title('Wilks by Sex (with Regression)',fontsize=20)
plt.xlabel('BodyweightKg',fontsize=15)
plt.ylabel('Wilks',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
def allometric_scale_squat(x):
    return x['BestSquatKg'] * x['BodyweightKg'] ** (-2/3)

def allometric_scale_bench(x):
    return x['BestBenchKg'] * x['BodyweightKg'] ** (-2/3)

def allometric_scale_deadlift(x):
    return x['BestDeadliftKg'] * x['BodyweightKg'] ** (-2/3)

def allometric_scale_total(x):
    return x['TotalKg'] * x['BodyweightKg'] ** (-2/3)
df['SquatAllometric'] = df.apply(allometric_scale_squat,axis=1)
df['BenchAllometric'] = df.apply(allometric_scale_bench,axis=1)
df['DeadliftAllometric'] = df.apply(allometric_scale_deadlift,axis=1)
df['TotalAllometric'] = df.apply(allometric_scale_total,axis=1)
df.filter(items=['Name','BodyweightKg','TotalKg','Total / BW','Wilks','TotalAllometric','Sex'])[df['Sex']=='F'].sort_values(ascending=False,by='TotalAllometric')[:10]
df.filter(items=['Name','BodyweightKg','TotalKg','Total / BW','Wilks','TotalAllometric','Sex'])[df['Sex']=='M'].sort_values(ascending=False,by='TotalAllometric')[:10]
sns.lmplot(x='BodyweightKg',
           y='SquatAllometric',
           data=df,
           hue='Sex',
           size=7)
plt.title('Allometric Squat Score by Sex (with Regression)',fontsize=20)
plt.xlabel('BodyweightKg',fontsize=15)
plt.ylabel('Allometric Squat Score',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
sns.lmplot(x='BodyweightKg',
           y='BenchAllometric',
           data=df,
           hue='Sex',
           size=7)
plt.title('Allometric Bench Score by Sex (with Regression)',fontsize=20)
plt.xlabel('BodyweightKg',fontsize=15)
plt.ylabel('Allometric Bench Score',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
sns.lmplot(x='BodyweightKg',
           y='DeadliftAllometric',
           data=df,
           hue='Sex',
           size=7)
plt.title('Allometric Deadlift Score by Sex (with Regression)',fontsize=20)
plt.xlabel('BodyweightKg',fontsize=15)
plt.ylabel('Allometric Deadlift Score',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
sns.lmplot(x='BodyweightKg',
           y='TotalAllometric',
           data=df,
           hue='Sex',
           size=7)
plt.title('Allometric Total Score by Sex (with Regression)',fontsize=20)
plt.xlabel('BodyweightKg',fontsize=15)
plt.ylabel('Allometric Total Score',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
