import pandas as pd

%matplotlib inline

pd.set_option("max_rows", 10)

pd.set_option("max_columns", 100)

from seaborn import set_style

set_style("darkgrid")

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("../input/allProp.csv", na_values=["\n"])

data.head()
# Checking if we have null values and type columns

data.info()
data.describe()
%%javascript

IPython.OutputArea.auto_scroll_threshold = 9999;
fig, axes = plt.subplots(data['ano'].nunique(),1, figsize=(8,100))

for (year, group), ax in zip(data.groupby("ano"), axes.flatten()):

    group.groupby(["autor1.txtNomeAutor"]).size().nlargest(5).plot(kind="barh",ax=ax,title=year)
fig, axes = plt.subplots(data['ano'].nunique(),1, figsize=(8,100))

for (year, group), ax in zip(data.groupby("ano"), axes.flatten()):

    group.groupby(["autor1.txtSiglaPartido"]).size().nlargest(10).plot(kind="barh",ax=ax,title=year)
data.groupby("autor1.txtSiglaPartido").size().nlargest(10).plot(kind="barh", figsize=(8,8))
# Creating new column with size of group by 2 features(year,political party)

data['groupby_ano_partido'] = data.groupby(["ano","autor1.txtSiglaPartido"])['ano'].transform('size')

# Removing extra white spaces on strings

data['autor1.txtSiglaPartido'] = data["autor1.txtSiglaPartido"].str.strip()

data = data[(data["autor1.txtSiglaPartido"] == "PT") |

     (data["autor1.txtSiglaPartido"] == "PSDB") |

     (data["autor1.txtSiglaPartido"] == "PMDB")]

data.groupby(["ano","autor1.txtSiglaPartido"]).mean().unstack("autor1.txtSiglaPartido")["groupby_ano_partido"].plot(figsize=(15,7),xticks=data['ano'])