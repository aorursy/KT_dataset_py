import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
sns.countplot(train.Target);
for feature in train.columns:
    print (feature)
#create a column to put every place (lugar) together, as they are dummies.
train['Region'] = np.nan
train.loc[train.lugar1 == 1, 'Region'] = 'Central'
train.loc[train.lugar2 == 1, 'Region'] = 'Chorotega'
train.loc[train.lugar3 == 1, 'Region'] = 'Central Pacific'
train.loc[train.lugar4 == 1, 'Region'] = 'Brunca'
train.loc[train.lugar5 == 1, 'Region'] = 'Atlantic Huetar'
train.loc[train.lugar6 == 1, 'Region'] = 'North Huetar'

pd.crosstab(train.Region, train.Target, normalize=0)
train['Area'] = np.nan
train.loc[train.area1 == 1, 'Area'] = 'Urban'
train.loc[train.area2 == 1, 'Area'] = 'Rural'

pd.crosstab(train.Area, train.Target, normalize=0)
pd.crosstab(train.tamhog, train.Target, normalize = 1)
sns.kdeplot(train.age, legend=False)
plt.xlabel("Age");
p = sns.FacetGrid(data = train, hue = 'Target', size = 4, legend_out=True)
p = p.map(sns.kdeplot, 'age')
plt.legend()
plt.title("Age distribution colored by household condition(target)")
p;
train['CookSource'] = np.nan
train.loc[train.energcocinar1 == 1, 'CookSource'] = 'NoKicthen'
train.loc[train.energcocinar2 == 1, 'CookSource'] = 'Electricity'
train.loc[train.energcocinar3 == 1, 'CookSource'] = 'Gas'
train.loc[train.energcocinar4 == 1, 'CookSource'] = 'WoodCharcoal'

pd.crosstab(train.CookSource, train.Target, normalize = 1)