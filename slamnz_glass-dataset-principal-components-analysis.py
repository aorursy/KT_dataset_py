from pandas import read_csv

data = read_csv('../input/glass.csv')

X = data.drop("Type",1)

y = data["Type"]
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(X)
pca.n_components_
from pandas import DataFrame

DataFrame(pca.explained_variance_.round(2), index=["P" + str(i) for i in range(1,10)], columns=["Explained Variance"]).T
DataFrame(pca.explained_variance_ratio_.round(2), index = ["P" + str(i) for i in range(1,10)], columns=["Explained Variance Ratio"]).T
components_applied = DataFrame(pca.transform(X))

components_applied.columns = ["P" + str(i) for i in range(1,10)]

components_applied.round(2).head()
from pandas import concat

from IPython.display import display



for p in components_applied.columns:

    

    new_df = X.copy()

    new_df[p] = components_applied[p]

    display(DataFrame(new_df.corr().round(2)[p]).drop(p,0).T)
from seaborn import kdeplot, distplot, set_style, despine

from matplotlib.pyplot import figure, show, title, subplots

from pandas import Series



set_style("whitegrid")

set_style({"axes.grid":False})



for_plotting = components_applied.copy()

for_plotting["Type"] = y



n = 10

fig, axes = subplots(int(10/2),2, figsize=(12.5,20))

fig.tight_layout()

i = 0

j = 0





for p in components_applied.columns:

    

    for t in for_plotting["Type"].unique():

        data_by_type = for_plotting[for_plotting["Type"] == t]

        kdeplot(Series(data_by_type[p], name = t), shade=True, ax=axes[i][j])

        #distplot(Series(data_by_type[p], name = t), kde=False, ax=axes[i][j])

        axes[i][j].set_title(p, loc="left")

        axes[i][j].set_xlabel("",visible=False)

        axes[i][j].set_yticklabels([],visible=False)

        despine(left=True)

    i += 1

    

    if(i == n/2):

        i = 0

        j += 1

    

show()
from itertools import combinations

from seaborn import lmplot



scatter_data = components_applied.copy()

scatter_data["label"] = y



for a,b in combinations(components_applied.columns,2):

    lmplot(data=scatter_data,x=a,y=b, hue="label", fit_reg=False)

    show()