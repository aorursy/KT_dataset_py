! conda install -y -c conda-forge hvplot==0.5.2 bokeh==1.4.0 imbalanced-learn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from toolz.curried import pipe, map, partial

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import hvplot.pandas
import holoviews as hv
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import Pipeline
from statsmodels.genmod.families import links
import statsmodels.api as sm 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
hv.extension('bokeh')
data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx').dropna(1, how='all')

((data
 .isna().mean().reset_index()
 .rename(columns={"index": 'Measurement', 0: '% Missing'})
 .hvplot.table()) + 
(data
 .isna().apply(np.logical_not).sum().reset_index()
 .rename(columns={"index": 'Measurement', 0: 'Number Not NaN'})
 .hvplot.table())).cols(2)
excluded = ['Patient addmited to regular ward (1=yes, 0=no)',
             'Patient addmited to semi-intensive unit (1=yes, 0=no)',
             'Patient addmited to intensive care unit (1=yes, 0=no)']

features = (data
            .select_dtypes(np.number)
            .drop(columns=excluded))

isna = features.isna().sum()

features = features.loc[:, isna < isna.median()] # we get rid of the worst offending features with the worst coverage

power = PowerTransformer()
scaler = StandardScaler()
imputer = IterativeImputer(sample_posterior=True)

X = (pipe(features,
          power.fit_transform, # bayesian ridge will assume data is normally distributed
          scaler.fit_transform, # we scale cause we are using regularization
          imputer.fit_transform,
          scaler.inverse_transform, # invert the transforms to original space
          power.inverse_transform,
          partial(pd.DataFrame, columns=features.columns, index=features.index))
     .replace({np.inf: np.nan, -np.inf:np.nan})
     .astype(np.float)
     .join(data
           .loc[:, excluded]
           .assign(negative = (data
                               .loc[:,'SARS-Cov-2 exam result']
                               .replace({'negative':1, 'positive':0})))
           .assign(positive = lambda df: 1 - df.sum(0))
           .idxmax(1)
           .to_frame('care'))
     .replace({'nan', np.nan})
     .dropna())

missing_weights_rows = (X.join(features
                          .isna()
                          .apply(np.logical_not)
                          .mean(1)
                          .to_frame('missing'), how='left')
                   .missing)

missing_weights_cols = (features
                        .iloc[X.index, :]
                        .loc[:, X.drop(columns=['care']).columns]
                       .isna()
                       .apply(np.logical_not)
                       .mean(0))

X_weighted = (X
                .drop(columns=['care'])
                .pipe(lambda df: pd.DataFrame(StandardScaler().fit_transform(df)))# standard scale
                .apply(lambda c: c * missing_weights_rows.to_numpy()) # doubly-weight the data according to its missingness
                .apply(lambda c: c * missing_weights_cols.to_numpy(), 1))

X_weighted.columns = X.drop(columns=['care']).columns

pipeline = Pipeline([('pca', PCA(2))])
Z_proj = (pipeline
     .fit_transform(X_weighted))

components = ([f'Components {i} ({round(v*100)}%)' for i, v in enumerate(pipeline.named_steps['pca'].explained_variance_ratio_)] 
              if hasattr(pipeline.named_steps['pca'], 'explained_variance_ratio_') 
              else ['Component 1', 'Component 2'])

(pd.DataFrame(Z_proj, columns=components)
 .assign(care=X.care)
 .dropna()
 .hvplot.scatter(x=components[0], y=components[1],
                 color='care', title='Not NaN Weighted Principle Components of Patient Outcomes',
                 width=1000, height=400))
care_level = X.care.replace({v: k for k, v in enumerate(['negative',
                                                         'Patient addmited to regular ward (1=yes, 0=no)',
                                                         'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                                         'Patient addmited to intensive care unit (1=yes, 0=no)'])})
care_level.hvplot.hist(title='Distribution of Care')
fa = FactorAnalysis(4)
Z = fa.fit_transform(X_weighted)

(pd.DataFrame(fa.components_,
              columns=X_weighted.columns)
 .assign(Factors = [f'Factor {i+1}' for i in range(fa.components_.shape[0])])
 .melt(id_vars = 'Factors', var_name='Variable', value_name='Loading')
 .hvplot.heatmap(x='Variable', y='Factors', C='Loading', 
                  height=600, width=800, colorbar=True, cmap='spectral_r')
 .opts(xrotation=90, title='Factor Loadings'))
Z_orth = Z

endog = care_level
exog = sm.add_constant(pd.DataFrame(Z_orth, columns = [f'Factor {i+1}' for i in range(Z_orth.shape[1])], index = endog.index))
glm_model = sm.GLM(endog,
                     exog,
                     family=sm.families.NegativeBinomial(link = links.Power(0.25)))

results = glm_model.fit()
results.summary()
fig, ax = plt.subplots()

ax.scatter(endog, results.resid_pearson)
ax.hlines(0, 0, 1)
ax.set_xlim(0, 1)
ax.set_title('Residual Dependence Plot')
ax.set_ylabel('Pearson Residuals')
ax.set_xlabel('Fitted values')
fig, ax = plt.subplots()

resid = results.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
sm.graphics.plot_partregress_grid(results)
(pd.DataFrame(np.dot(results
                     .params
                     .drop('const')
                     .to_frame('coef')
                     .T, fa.components_) + fa.mean_,
              columns=X.drop(columns=['care']).columns,
              index =['Coefficients'])
.T.hvplot.bar()
.opts(xrotation=90, width=800, height=600, title='Reprojected GLM Factor Weights'))
