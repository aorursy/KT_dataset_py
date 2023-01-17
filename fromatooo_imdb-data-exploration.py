import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
f = pd.read_csv('../input/IMDB-Movie-Data.csv')
print(f.shape)
f.head()
plt.hist(f.Year)

plt.show()
f.Year.value_counts(normalize=True)
plt.hist(f.Rating)

plt.show()
f.Rating.value_counts(normalize=True).head(10)
f.Rating.describe()
f[f.Rating == 1.9]
pd.set_option('display.max_colwidth', -1)

f[f.Title == 'Disaster Movie'].Description
f[f.Rating == 9.0]
# i copied this function from here: https://stackoverflow.com/questions/29530355/plotting-multiple-histograms-in-grid



def draw_histograms(df, variables, n_rows, n_cols):

    fig=plt.figure()

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=10,ax=ax)

        ax.set_title(var_name+" Distribution")

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()

    

f_col = ['Runtime (Minutes)', 'Votes', 'Revenue (Millions)', 'Metascore']

    

draw_histograms(f, f_col, 2, 2)
f_corr = f.corr()



sns.heatmap(f_corr)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()



f_corr
fig = plt.figure()

fig.add_subplot(321)

plt.scatter(f.Rating, f.Metascore)

fig.add_subplot(322)

plt.scatter(f.Votes, f['Revenue (Millions)'])

fig.add_subplot(323)

plt.scatter(f.Year, f.Votes)

fig.add_subplot(324)

plt.scatter(f.Rating, f.Votes)

fig.add_subplot(325)

plt.scatter(f.Rating, f['Revenue (Millions)'])

fig.add_subplot(326)

plt.scatter(f.Metascore, f['Revenue (Millions)'])

fig.tight_layout()

plt.show()