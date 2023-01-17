# Import the packages
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
cr = pd.read_csv("../input/cereal.csv")
cr = cr.sort_values(by=["mfr","name"])
cr.head(10)
sns.set_style(style="darkgrid")
sns.boxplot(x="mfr", y="rating", data=cr)
nabisco = cr.loc[cr.mfr == "N"]
nabisco.sort_values(by="rating", ascending=False)
nabisco.rating.mean()
cr.loc[cr.rating >= 90]
sns.regplot(x="sugars", y="rating", data=cr)
cr_new = cr.loc[cr.sugars >= 0]
sns.regplot(x="sugars", y="rating", data=cr_new)
model = smf.ols(formula = "rating~sugars", data=cr_new).fit()
model.summary()
model.pvalues