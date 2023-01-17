import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
df = pd.read_csv(filepath_or_buffer='../input/beauty.csv')
type(df)
df.head()
df['wage'].head()
df.shape
df.columns
df.info()
df.describe()
df['wage'].hist();
plt.figure(figsize=(16, 10))
df.hist();
df['female'].value_counts()
df['looks'].value_counts()
df['female'].value_counts()
df['goodhlth'].value_counts(normalize=True)
df.iloc[:6, 5:7]
toy_df = pd.DataFrame({'age':[17, 32, 56], 
                       'salary':[56,69, 120]}, 
                      index=['Kate', 'Leo', 'Max'])
toy_df
toy_df.iloc[1, 1]
toy_df.loc[['Leo', 'Max'], 'age']
df[df['wage'] > 40]
df[(df['wage'] > 10) & (df['female'] == 1)]
df['female'].apply(lambda gender_id : 'female' if gender_id == 1 else 'male').head()
df['female'].map({0: 'male', 1: 'female'}).head()
df.loc[df['female'] == 0, 'wage'].median()
df.loc[df['female'] == 1, 'wage'].median()
for (gender_id, sub_dataframe) in df.groupby('female'):
    #print(gender_id)
    #print(sub_dataframe.shape)
    print('Median wages for {} are {}'.format('men' if gender_id == 0
                                             else 'women', 
                                             sub_dataframe['wage'].median()))
df.groupby('female')['wage'].median()
df.groupby(['female', 'married'])['wage'].median()
pd.crosstab(df['female'], df['married'])
import seaborn as sns
df['educ'].value_counts()
sns.boxplot(x='educ', y='wage', data=df[df['wage']<30]);
