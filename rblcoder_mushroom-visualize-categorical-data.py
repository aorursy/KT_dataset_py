

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns
df_mush = pd.read_csv('../input/mushrooms.csv')
df_mush.head()
df_mush.describe().T
# https://www.kaggle.com/fedi1996/boston-crime-analysis-with-plotly



import plotly.express as px



habitat_counts = df_mush['habitat'].value_counts()

values = habitat_counts.values

categories = pd.DataFrame(data=habitat_counts.index, columns=["habitat"])

categories['values'] = values



fig = px.treemap(categories, path=['habitat'], values=values, height=700,

                 title="Counts by Habitat", 

                 color_discrete_sequence = px.colors.sequential.RdBu)

fig.data[0].textinfo = 'label+text+value'

fig.show()
df_mush[df_mush['class']=='e'].describe().T
df_mush[df_mush['class']=='p'].describe().T
df_mush.isnull().sum()
df_mush.info()
#https://github.com/santosjorge/cufflinks/issues/185

!pip install plotly

!pip install cufflinks

import cufflinks as cf

cf.set_config_file(offline=True)
df_mush.columns
for col in df_mush.columns[1:]:

    pd.crosstab(df_mush['class'], df_mush[col], margins=True, normalize=True).iplot(kind='bar', title = 'Counts of mushroom class in '+col)

# pd.crosstab(df_mush['class'], df_mush['cap-shape'], margins=True, normalize=True).iplot(kind='bar', title = 'Counts of mushroom class in cap-shape')

# pd.crosstab(df_mush['class'], df_mush['cap-surface'], margins=True, normalize=True).iplot(kind='bar')
for col in df_mush.columns[1:]:

    pd.crosstab(df_mush[col], df_mush['class'], normalize=True).iplot(kind='bar', title ='Types of ' + col)

#https://stackoverflow.com/questions/12286607/making-heatmap-from-pandas-dataframe

#https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe

from IPython.display import display, HTML
for col in df_mush.columns[1:]:

    df_c = pd.crosstab(df_mush['class'], df_mush[col], normalize=True)

    style = df_c.style.background_gradient(cmap='Blues')

    display(style)
for col in df_mush.columns[1:]:

    df_c = pd.crosstab(df_mush[col], df_mush['class'], normalize=True)

    style = df_c.style.background_gradient(cmap='Blues')

    display(style)
dum = pd.get_dummies(df_mush, prefix=df_mush.columns)
dum.corr()['class_e'].nlargest(5)
dum.corr()['class_e'].nsmallest(5)
dum.corr()['class_p'].nlargest(5)
dum.corr()['class_p'].nsmallest(5)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

X=enc.fit_transform(df_mush.drop(columns=['class']))

y=preprocessing.LabelEncoder().fit_transform(df_mush['class'])

rf = RandomForestClassifier()

print(cross_val_score(rf, X, y, cv=3))