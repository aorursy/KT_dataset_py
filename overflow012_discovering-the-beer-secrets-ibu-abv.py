%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from  scipy.stats import pearsonr as pearson



df = pd.read_csv("../input/beers.csv")

df.describe()
df = df[df["ibu"].notnull()]

clean_df = df[df["abv"].notnull()]



print (clean_df["style"].value_counts()[:10])



(clean_df["style"].value_counts()[:10]).plot(kind = 'bar')
styles = (clean_df["style"].value_counts()[:10]).keys()



print(styles)
fig, axes = plt.subplots(4, 3, sharex = True, sharey = True, figsize=(10,10)) # One beer style by figure  

fig, global_ax = plt.subplots(figsize=(12,12)) # All beers styles in one figure.

x_max = clean_df["ibu"].max() # Get the max ibu value

y_max = clean_df["abv"].max() # Get the max abv value



for style, ax in zip(styles, axes.ravel()):

    ibu_data = clean_df["ibu"][clean_df["style"] == style].values

    abv_data = clean_df["abv"][clean_df["style"] == style].values

    

    ax.set_title(style)

    ax.plot(ibu_data, abv_data, marker = 'o', linestyle = '')

    ax.legend(numpoints=1, loc='lower right', fontsize = 10)



    global_ax.plot(ibu_data, abv_data, marker = 'o', label = style, linestyle = '')



global_ax.legend(numpoints=1, loc='lower right', fontsize = 10)



plt.show()
print(pearson(clean_df["ibu"].values, clean_df["abv"].values)[0])