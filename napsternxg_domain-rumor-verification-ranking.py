# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_context("poster")

sns.set_style("ticks")
df_emergent = pd.read_csv("../input/emergent.csv")

df_emergent.head()
df_emergent.claim_source_domain.value_counts()
df_t = df_emergent.pivot_table(index="claim_source_domain",

                       columns="claim_label",

                       values="emergent_page", aggfunc=len)



df_t.sort_values("TRUE", ascending=False)
df_t.sort_values("FALSE", ascending=False)
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression
idx = df_emergent["claim_label"] != "Unverified"

X = pd.get_dummies(df_emergent["claim_source_domain"], sparse=True)[idx]

y = df_emergent["claim_label"][idx]

X.shape, y.shape
def plot_heat(values, index, columns):

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

    fig, (ax, cbar_ax) = plt.subplots(2,

        figsize=(6,22), gridspec_kw=grid_kws)         # Sample figsize in inches

    sns.heatmap(

        pd.DataFrame(

            values,

            columns=columns,

            index=index).sort_values("TRUE", ascending=False),

        #square=True,

        ax=ax,

        cbar_ax=cbar_ax,

        cbar_kws={"orientation": "horizontal"},

        annot=True

    )

    return fig, ax, cbar_ax



#fig.tight_layout()
model = MultinomialNB()

model.fit(X, y)
plot_heat(

    np.vstack([-model.coef_, model.coef_]).T,

    X.columns, model.classes_)
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")

model.fit(X, y)
plot_heat(

    np.vstack([-model.coef_, model.coef_]).T,

    X.columns, model.classes_)