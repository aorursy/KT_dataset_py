%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
recent_grads = pd.read_csv('../input/recent-grads.csv')
recent_grads.iloc[0]
recent_grads.head()
recent_grads.tail()
print(recent_grads.isnull().any(), '\n\n')
print(recent_grads.count())
recent_grads.loc[recent_grads.isnull().any(axis=1)]
recent_grads.dropna(inplace=True)
recent_grads.drop(['Major_code', 'Rank'], axis=1).corr()
recent_grads.plot(x='Sample_size', y='Employed', 
                  kind='scatter', figsize=(10,6),
                  xlim=(0, 4500), ylim=(0,350000)
                  )
axis_label_dict= {'fontsize': 16, 'fontweight': 'bold'}
title_dict = {'fontsize': 20, 'fontweight': 'bold'}
plt.xlabel('Sample size', fontdict=axis_label_dict)
plt.ylabel('Employed', fontdict=axis_label_dict)
plt.title('Sample size vs Employed',
          fontdict=title_dict)
recent_grads.plot(x='ShareWomen', y='Unemployment_rate', 
                  kind='scatter', figsize=(10,6),
                  xlim=(-0.05,1.0), ylim=(-0.05,0.25)
                  )
plt.xlabel('Share of Women', fontdict=axis_label_dict)
plt.ylabel('Unemployment rate', fontdict=axis_label_dict)
plt.title('Share of Women vs Unemployment rate',
          fontdict=title_dict)
recent_grads.plot(x='ShareWomen', y='Median', 
                  kind='scatter', figsize=(10,6),
                  xlim=(-0.05,1.0)
                  )
plt.xlabel('Share of Women', fontdict=axis_label_dict)
plt.ylabel('Median', fontdict=axis_label_dict)
plt.title('Share of Women vs Median',
          fontdict=title_dict)
pd.get_dummies(recent_grads[['Median', 'Major_category', 'ShareWomen']], columns=['Major_category']).corr()
import seaborn as sns
sns.set_style('white')
colors= ['#d50000', '#f8bbd0', '#aa00ff', '#6200ea',
         '#2962ff', '#00bfa5', '#00c853', '#aeea00', 
         '#a57f17', '#ff6d00', '#bb3300', '#1b5e20',
         '#ce7eb0', '#0e223e', '#6d2957', '#8a72c8']

lm = sns.lmplot(data=recent_grads, x='Median', y='ShareWomen',
                hue='Major_category', fit_reg=False,
                size=5, aspect=2, scatter_kws={"s": 35},
                palette=sns.color_palette(colors, n_colors=16),
                legend_out=True)

axes = lm.axes
axes[0,0].set_ylim(-0.05, 1.0)
axes[0,0].set(xlabel='Median', ylabel='Share of Women')
lm._legend.set_title('Major Category')
