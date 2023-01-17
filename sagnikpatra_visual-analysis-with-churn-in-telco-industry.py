import numpy as np

import pandas as pd



# we don't like warnings

# you can comment the following 2 lines if you'd like to

import warnings

warnings.filterwarnings('ignore')



# Matplotlib forms basis for visualization in Python

import matplotlib.pyplot as plt



# We will use the Seaborn library

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'
data=pd.read_csv('../input/edadata/telecom_churn.csv')
data.head()
data.shape
data.info()
features = ['Total day minutes', 'Total intl calls']
data[features].describe()
plt.rcParams['figure.figsize']=(10,7)

sns.boxplot('Total day minutes',data=data)
plt.rcParams['figure.figsize']=(10,7)

sns.boxplot('Total intl calls',data=data)
plt.rcParams['figure.figsize']=(10,7)

data[features].hist();
data[features].plot(kind='density', subplots=True, layout=(1, 2), 

                  sharex=False, figsize=(10, 7));
sns.distplot(data['Total intl calls']);
data['Churn'].value_counts()
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))



sns.countplot(x='Churn', data=data, ax=axes[0]);

sns.countplot(x='Customer service calls', data=data, ax=axes[1]);
num = list(set(data.columns) - 

                 set(['State', 'International plan', 'Voice mail plan', 

                      'Area code', 'Churn', 'Customer service calls']))



# Calculate and plot

corr_matrix = data[num].corr()

sns.heatmap(corr_matrix);
num=list(set(num)-set(['Total day charge','Total night charge','Total eve charge','Total intl charge']))
plt.scatter(data['Total day minutes'],data['Customer service calls']);
sns.jointplot(x='Total day minutes', y='Customer service calls', 

              data=data, kind='scatter');
sns.jointplot('Total day minutes', 'Customer service calls', data=data,

              kind="kde", color="g");
sns.countplot(x='Voice mail plan', hue='Churn', data=data);

plt.title('Loyal & Churned with the Voice Mail Plan')
pd.crosstab(data['Voice mail plan'], data['Churn'],normalize=True)
sns.lmplot('Total day minutes', 'Total night minutes', data=data, hue='Churn', fit_reg=False);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))



sns.boxplot(x='Churn', y='Total day minutes', data=data, ax=axes[0]);

sns.violinplot(x='Churn', y='Total day minutes', data=data, ax=axes[1]);
sns.catplot(x='Churn', y='Total day minutes', col='Customer service calls',

               data=data[data['Customer service calls'] < 8], kind="box",

               col_wrap=6, height=5, aspect=.8);
%config InlineBackend.figure_format = 'png'

sns.pairplot(data[num]);
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
ok=data.drop(['Churn','State'],axis=1)

ok['International plan']=ok['International plan'].map({'Yes':1,'No':0})

ok['Voice mail plan']=ok['Voice mail plan'].map({'Yes':1,'No':0})
scaler = StandardScaler()

scaled = scaler.fit_transform(ok)
tsne=TSNE(random_state=18)

tsne_repr = tsne.fit_transform(scaled)
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=.5);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 8))



for i, name in enumerate(['International plan', 'Voice mail plan']):

    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], 

                    c=data[name].map({'Yes': 'orange', 'No': 'blue'}), alpha=.5);

    axes[i].set_title(name);