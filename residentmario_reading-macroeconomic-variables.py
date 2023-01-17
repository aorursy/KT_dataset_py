import pandas as pd

macro = pd.read_csv("../input/data.csv", parse_dates=['Date'])

dictionary = pd.read_csv("../input/documentation.csv")

macro = (

    macro

        .set_index('Variable')

        .join(dictionary.set_index('file_name'))

        .reset_index()

        .rename(columns={'index': 'variable', 

                         'Date': 'date', 

                         'Value': 'value'})

)



macro.head(3)
macro.shape
len(macro.description.unique())
import seaborn as sns

sns.set_style('whitegrid')



(macro

     .query('description == "U.S. Index of Crop Production 1862-1930"')

     .set_index('date')

     .value

).plot.line(title='United States Crop Production Index', figsize=(12, 6))

sns.despine()
macro.date.dt.year.value_counts().sort_index().plot.line(

    title='Years by Number of Indicators Reporting', figsize=(12, 6)

)
macro.variable.value_counts().plot.line(title='Indicators by Length of Reporting', 

                                        figsize=(12, 6))
macro.value.value_counts().head(20).plot.bar(

    figsize=(12, 6), title='Most Common Indicator Values'

)
indicator_names = macro.description.unique()

indicator_names = pd.Series(indicator_names).dropna().values
from wordcloud import WordCloud

w = WordCloud(scale=2).generate(" ".join(indicator_names))



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))

plt.imshow(w, interpolation='bilinear')

plt.axis("off")
macro = macro.dropna(subset=['description'])
import numpy as np

macro.loc[

    np.where(macro.description.str.contains('Coal ').values)

].description.unique()