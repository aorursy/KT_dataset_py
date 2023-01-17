import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import os
import warnings
plt.rcParams['figure.figsize'] = [20, 8]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')
apps = pd.read_csv("../input/application_train.csv")
apps.shape
apps.info()
apps.head()
intColumns = []
for c in apps.columns:
    if apps[c].dtype == 'object':
        pass
    else:
        intColumns.append(c)
pos = corr.TARGET.sort_values()[:5].index.values
neg = corr.TARGET.sort_values(ascending=False)[:5].index.values
imp_columns = np.concatenate([pos,neg])
plt.matshow(apps[imp_columns].corr())
apps[imp_columns].corr().TARGET
import seaborn as sns
corr = apps[imp_columns].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
apps[imp_columns].describe()
ggplot(apps, aes(x='factor(TARGET)', y='EXT_SOURCE_3')) + \
geom_boxplot()
ggplot(apps, aes(x='factor(TARGET)', y='EXT_SOURCE_2')) + \
geom_boxplot()
ggplot(apps, aes(x='factor(TARGET)', y='EXT_SOURCE_1')) + \
geom_boxplot()
apps.describe(include='all')
plt.figure(figsize=(10, 20))
apps[apps.columns[apps.isna().sum() > 0]].isna().sum().sort_values().plot(kind='barh')
plt.show()
apps['TARGET'].value_counts().plot(kind='bar')
plt.show()
ggplot(apps.head(10000), aes(x='AMT_INCOME_TOTAL', y = 'AMT_CREDIT', color = 'factor(TARGET)')) +\
geom_point(alpha=0.2)
ggplot(apps, aes(x='AMT_CREDIT', color='factor(TARGET)')) +\
geom_density()
ggplot(apps[apps.AMT_INCOME_TOTAL <= 1000000], aes(x='AMT_INCOME_TOTAL', color='factor(TARGET)')) +\
geom_density()
apps['CNT_CHILDREN'].value_counts().plot(kind='bar')
plt.show()
ggplot(apps, aes(x='NAME_CONTRACT_TYPE'))+\
geom_bar(aes(fill='factor(TARGET)'))
ggplot(apps, aes(x='CODE_GENDER'))+\
geom_bar(aes(fill='factor(TARGET)'))
ggplot(apps, aes(x='FLAG_OWN_CAR'))+\
geom_bar(aes(fill='factor(TARGET)'))
ggplot(apps, aes(x='FLAG_OWN_REALTY'))+\
geom_bar(aes(fill='factor(TARGET)'))
