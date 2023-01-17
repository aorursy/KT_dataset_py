import seaborn as sns

%matplotlib inline

tit = sns.load_dataset('titanic')

tit.head()
sns.countplot(x = 'sex',data = tit)
sns.jointplot(x = 'survived',y = 'age',data= tit,kind= 'kde')
tit.dropna().head()
sns.distplot(tit['fare'])
sns.pairplot(tit,hue = 'sex')
sns.pairplot(tit,hue = 'class')
sns.pairplot(tit,hue = 'embark_town')
sns.boxplot(x = 'class',y = 'age',data = tit,hue = 'who')
sns.barplot(x = 'sex',y = 'fare',data = tit)
sns.jointplot(x = 'fare', y ='age',data = tit,kind = 'hex')
tit.head()
import pandas as pd
a = tit
a.head()
df = a[['survived','fare','age']]
df.head()

hm = df.corr()
sns.heatmap(hm)