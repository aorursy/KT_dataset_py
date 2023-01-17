import pandas as pd

pokemon = pd.read_csv('../input/pokemon.csv')

pokemon.head(3)
len(pokemon[pokemon['type2'].isnull()])
df = pokemon[pokemon['type2'].isnull()].loc[

    :, ['sp_attack', 'sp_defense', 'attack', 'defense', 'speed', 'hp', 'type1']

]

X = df.iloc[:, :-1].values



from sklearn.preprocessing import normalize

X_norm = normalize(X)



y = df.iloc[:, -1].values
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=3)

lda.fit(X_norm, y)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import numpy as np



fig, ax = plt.subplots(1, 1, figsize=(12, 10))



sns.heatmap(pd.DataFrame(lda.coef_, 

                         columns=df.columns[:-1], 

                         index=[lda.classes_]), 

            ax=ax, cmap='RdBu', annot=True)



plt.suptitle('LDA Feature Coefficients')

pass
pd.Series(np.abs(lda.coef_).sum(axis=1), index=lda.classes_).sort_values().plot.bar(

    figsize=(12, 6), title="LDA Class Coefficient Sums"

)
lda.explained_variance_ratio_
X_hat = lda.fit_transform(X, y)



import matplotlib as mpl



colors = mpl.cm.get_cmap(name='tab20').colors

categories = pd.Categorical(pd.Series(y)).categories

ret = pd.DataFrame(

    {'C1': X_hat[:, 0], 'C2': X_hat[:, 1], 'Type': pd.Categorical(pd.Series(y))}

)



fig, ax = plt.subplots(1, figsize=(12, 6))



for col, cat in zip(colors, categories):

    (ret

         .query('Type == @cat')

         .plot.scatter(x='C1', y='C2', color=col, label=cat, ax=ax,

                       s=100, edgecolor='black', linewidth=1,

                       title='Two-Component LDA Decomposition')

         .legend(bbox_to_anchor=(1.2, 1))

    )