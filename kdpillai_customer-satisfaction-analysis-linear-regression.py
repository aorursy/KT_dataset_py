import pandas as pd

import numpy as np

df = pd.read_csv("../input/rintro-chapter7.csv")

df.head()
df.describe()
import seaborn as sns

sns.set(style="white")

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot)

#g.map_upper(sns.scatterplot)

g.map_upper(sns.regplot)

g.map_diag(sns.kdeplot, lw=3)
df['logdistance'] = np.log(df['distance'])

df.head()
sns.set(style="white")

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot)

#g.map_upper(sns.scatterplot)

g.map_upper(sns.regplot)

g.map_diag(sns.kdeplot, lw=3)
df.columns



X = df.drop(columns= ['overall','distance', 'num.child' ])

X = pd.get_dummies(data=X, drop_first=True, columns = ['weekend'])

X.head()
Y = df[['overall']]

Y.head()
import statsmodels.api as sm

X = sm.add_constant(X)

model = sm.OLS(Y,X)

results = model.fit()

results.summary()
# predicted  values

model_fitted_y  = results.fittedvalues

# model residuals

model_residuals = results.resid

# normalized residuals

model_norm_residuals = results.get_influence().resid_studentized_internal

# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals

model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals

model_leverage = results.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals

model_cooks = results.get_influence().cooks_distance[0]

import matplotlib.pyplot as plt

plot_lm_1 = plt.figure()

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, df.columns[-1], data=df,

                          lowess=True,

                          scatter_kws={'alpha': 0.5},

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals');
from statsmodels.graphics.gofplots import ProbPlot

QQ = ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations

abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i,

                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
plot_lm_4 = plt.figure();

plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);

sns.regplot(model_leverage, model_norm_residuals,

              scatter=False,

              ci=False,

              lowess=True,

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});

plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)

plot_lm_4.axes[0].set_ylim(-3, 5)

plot_lm_4.axes[0].set_title('Residuals vs Leverage')

plot_lm_4.axes[0].set_xlabel('Leverage')

plot_lm_4.axes[0].set_ylabel('Standardized Residuals');



  # annotations

leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

for i in leverage_top_3:

    plot_lm_4.axes[0].annotate(i,

                                 xy=(model_leverage[i],

                                     model_norm_residuals[i]));