import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.describe().transpose()
sns.distplot(df['bmi'])
sns.distplot(df['age'])
sns.countplot(x=df['children'])
sns.countplot(x=df['region'])
sns.distplot(df['charges'])
sns.scatterplot(x='bmi',y='charges',data=df)
sns.lmplot(x='bmi',y='charges',data=df)
sns.scatterplot(x='bmi',y='charges',hue='smoker',data=df)
sns.lmplot(x='bmi',y='charges',hue='smoker',data=df)
sns.scatterplot(x='bmi',y='charges',hue='region',data=df)
sns.lmplot(x='bmi',y='charges',hue='region',data=df)
sns.scatterplot(x='bmi',y='charges',hue='sex',data=df)
sns.lmplot(x='bmi',y='charges',hue='sex',data=df)
sns.scatterplot(x='age',y='charges',data=df)
sns.lmplot(x='age',y='charges',data=df)
sns.scatterplot(x='age',y='charges',hue='smoker',data=df)
sns.lmplot(x='age',y='charges',hue='smoker',data=df)
sns.scatterplot(x='age',y='charges',hue='region',data=df)
sns.lmplot(x='age',y='charges',hue='region',data=df)
sns.scatterplot(x='age',y='charges',hue='sex',data=df)
sns.lmplot(x='age',y='charges',hue='sex',data=df)
sns.scatterplot(x='age',y='bmi',data=df)
sns.lmplot(x='age',y='bmi',data=df)
sns.scatterplot(x='age',y='charges',hue='bmi',data=df)
sns.swarmplot(x=df['smoker'],y=df['charges'])
sns.catplot(x='sex',data=df,kind='count',hue='smoker')
sns.scatterplot(x='age',y='charges',hue='sex',data=df[df['smoker']=='yes'])
sns.lmplot(x='age',y='charges',hue='sex',data=df[df['smoker']=='yes'])
sns.scatterplot(x='age',y='charges',hue='sex',data=df[df['smoker']=='no'])
sns.lmplot(x='age',y='charges',hue='sex',data=df[df['smoker']=='no'])
sns.swarmplot(x=df['region'],y=df['charges'])
sns.catplot(x='region',data=df,kind='count',hue='smoker')
sns.swarmplot(x=df['region'],y=df['charges'][df['smoker']=='yes'])
sns.swarmplot(x=df['region'],y=df['charges'][df['smoker']=='no'])
sns.swarmplot(x=df['children'],y=df['charges'])
sns.scatterplot(x='age',y='charges',hue='children',data=df)
sns.lmplot(x='age',y='charges',hue='children',data=df)
sns.scatterplot(x='bmi',y='charges',hue='children',data=df)
sns.lmplot(x='bmi',y='charges',hue='children',data=df)