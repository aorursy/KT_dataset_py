# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/database.csv")
df.head()
df.loc[df["birth_year"] == "530s", "birth_year"] = 530
df.loc[df["birth_year"] == "1237?", "birth_year"] = 1237
df = df.drop(df.index[df.loc[:,"birth_year"] == "Unknown"])
df.loc[:, "birth_year"] = pd.to_numeric(df.loc[:,"birth_year"])
df.info()
df_stats = df.drop(["article_id", "full_name", "city", "state", "latitude", "longitude"], axis = 1)
df_stats.info()
import statsmodels.api as sm
import statsmodels.formula.api as smf
df_stats.info()
from patsy import dmatrices

y, X = dmatrices('historical_popularity_index~sex+birth_year+country+continent+industry+occupation+domain+article_languages+np.log(page_views)+np.log(average_views)', data=df_stats, return_type='dataframe')

df_stats = df_stats.dropna(axis = 0, how = "any")
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
sum(np.isnan(model_norm_residuals))
model_norm_residuals2 = model_norm_residuals[~np.isnan(model_norm_residuals)]
# fitted values (need a constant term for intercept)
model_fitted_y = res.fittedvalues

# model residuals
model_residuals = res.resid

# normalized residuals
model_norm_residuals = res.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)
import seaborn as sns
import matplotlib.pyplot as plt
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, "historical_popularity_index", data = df_stats,
                          lowess=True,
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_residuals[i]));
from statsmodels.graphics.gofplots import ProbPlot
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals2)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));
