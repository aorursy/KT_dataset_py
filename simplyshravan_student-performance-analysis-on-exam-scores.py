import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('../input/StudentsPerformance.csv')
df.columns
df.head()
df.describe()
sns.pairplot(df)
sns.countplot(x='gender',data=df)
sns.set_style("whitegrid", {'axes.grid' : True})
sns.countplot(x='race/ethnicity',data=df)
sns.countplot(x='parental level of education',data=df)
df['parental level of education'].value_counts().plot.bar(title="Parental Level of education")
df.groupby('test preparation course').count()
df[df['test preparation course']=='none'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores without test preparation")
df[df['test preparation course']!='none'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores with test preparation")
df.columns
df1=df.groupby('race/ethnicity').describe()
df1
df1['math score']['mean'].plot.bar(title="Avg Math score across Ethnicity")
df1['reading score']['mean'].plot.bar(title="Avg Reading score across Ethnicity")
df1['writing score']['mean'].plot.bar(title="Avg Writing score across Ethnicity")
df2=df.groupby('parental level of education').describe()
df2['writing score']['mean'].plot.bar(title="Avg Writing scores on various parental level education")
df2['reading score']['mean'].plot.bar(title="Avg Writing scores on various parental level education")
df2['math score']['mean'].plot.bar(title="Avg Writing scores on various parental level education")
df[df['gender']=='female'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores for females")
df[df['gender']=='male'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores for males")
df[df['lunch']=='standard'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores for standard lunch")
df[df['lunch']!='standard'].mean()[['math score','reading score','reading score']].plot.bar(title="Avg scores for free/reduced lunch")