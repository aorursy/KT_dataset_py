# Set warning messages

import warnings

# Show all warnings in IPython

warnings.filterwarnings('always')

# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# Import built-in modules

import sys

import platform

import os

from pathlib import Path



# Import external modules

from IPython import __version__ as IPy_version

import IPython.display as ipyd

import numpy as np

import pandas as pd

from sklearn import __version__ as skl_version

import matplotlib as mpl

import matplotlib.pyplot as plt

from bokeh import __version__ as bk_version

from scipy import __version__ as scipy_version

import scipy.stats as sps

from statsmodels import __version__ as sm_version

import statsmodels.api as sm

import statsmodels.formula.api as smf



# Check they have loaded and the versions are as expected

assert platform.python_version_tuple() == ('3', '6', '6')

print(f"Python version:\t\t{sys.version}")

assert IPy_version == '7.13.0'

print(f'IPython version:\t{IPy_version}')

assert np.__version__ == '1.18.2'

print(f'numpy version:\t\t{np.__version__}')

assert pd.__version__ == '0.25.3'

print(f'pandas version:\t\t{pd.__version__}')

assert skl_version == '0.22.2.post1'

print(f'sklearn version:\t{skl_version}')

assert mpl.__version__ == '3.2.1'

print(f'matplotlib version:\t{mpl.__version__}')

assert bk_version == '2.0.1'

print(f'bokeh version:\t\t{bk_version}')

assert scipy_version == '1.4.1'

print(f'scipy version:\t\t{scipy_version}')

assert sm_version == '0.11.0'

print(f'statsmodels version:\t{sm_version}')
# Load Bokeh for use in a notebook

from bokeh.io import output_notebook

output_notebook()
# Output exact environment specification, in case it is needed later

print("Capturing full package environment spec")

print("(But note that not all these packages are required)")

!pip freeze > requirements_Kaggle.txt

!jupyter --version > jupyter_versions.txt
# Simulate data

sample_size = 200

Beta_true = np.array([np.log(0.3), 0.2])

X_des_mx = np.array([

    [1] * sample_size,

    sps.norm(loc=10, scale=3).rvs(size=sample_size, random_state=11)

]).T

y_vec = sps.poisson(

    mu=np.exp(np.matmul(X_des_mx, Beta_true))

).rvs(size=sample_size, random_state=76)



data_df = pd.concat([

    pd.DataFrame(

        X_des_mx,

        columns=[f"x_{val}" for val in range(X_des_mx.shape[1])]

    ),

    pd.DataFrame({'y': y_vec})

], sort=False, axis=1)

data_df.describe().style.format('{:.3f}')
data_df.plot.scatter('x_1', 'y')

plt.show()
# Get the data

data = sm.datasets.fair.load_pandas().data

data["affairs"] = np.ceil(data["affairs"])

selected_cols = 'affairs rate_marriage age yrs_married'.split()

data = data[selected_cols]
# "Condensed" data is one row per unique combination of 

# explanatory variables *and* response variable

dc = data.copy().groupby(selected_cols).agg(

    freq=('affairs', 'size')).reset_index()

dc.head()
# "Aggregated" data is one row per unique combination of 

# explanatory variables *only*, with the response variable

# either summed or averaged.

df_a = data.copy().groupby(selected_cols[1:]).agg(**{

    'affairs_' + func_str: ('affairs', func_str) for 

    func_str in ['mean', 'sum','count']

}).reset_index()

df_a.head()
# Initialise DataFrame for holding model info

mod_fields = ['Data', 'Target', 'Weight', 'Exposure']

mods_df = pd.DataFrame(np.empty(0, dtype=np.dtype(

    [(field_name, np.dtype('O')) for field_name in mod_fields] +

    [('GLMResults', np.dtype('O')),]

)))
explanatory_vars = 'rate_marriage age yrs_married'.split(' ')



# Original data

mods_df.loc[0, mod_fields] = ['Original', 'count', 'None', 'None']

mods_df.loc[0, 'GLMResults'] = smf.glm(

    'affairs ~ rate_marriage + age + yrs_married',

    data=data, family=sm.families.Poisson()

).fit()



# Condensed data with freq_weights

mods_df.loc[1, mod_fields] = ['Condensed', 'count', 'freq', 'None']

mods_df.loc[1, 'GLMResults'] = smf.glm(

    'affairs ~ rate_marriage + age + yrs_married',

    data=dc, family=sm.families.Poisson(), 

    freq_weights=dc['freq']

).fit()



# Condensed with var_weights

mods_df.loc[2, mod_fields] = ['Condensed', 'count', 'var', 'None']

mods_df.loc[2, 'GLMResults'] = smf.glm(

    'affairs ~ rate_marriage + age + yrs_married',

    data=dc, family=sm.families.Poisson(),

    var_weights=dc['freq']

).fit()



# Aggregated sum with exposure

mods_df.loc[3, mod_fields] = ['Aggregated', 'count_sum', 'None', 'exposure']

mods_df.loc[3, 'GLMResults'] = smf.glm(

    'affairs_sum ~ rate_marriage + age + yrs_married',

    data=df_a, family=sm.families.Poisson(),

    exposure=df_a['affairs_count']

).fit()

# Same, but with offset = log(exposure)

mods_df.loc[4, mod_fields] = ['Aggregated', 'count_sum', 'None', 'offset']

mods_df.loc[4, 'GLMResults'] = smf.glm(

    'affairs_sum ~ rate_marriage + age + yrs_married',

    data=df_a, family=sm.families.Poisson(),

    offset=np.log(df_a['affairs_count'])

).fit()



# Aggregated mean with var_weights

mods_df.loc[5, mod_fields] = ['Aggregated', 'count_mean', 'var', 'None']

mods_df.loc[5, 'GLMResults'] = smf.glm(

    'affairs_mean ~ rate_marriage + age + yrs_married',

    data=df_a, family=sm.families.Poisson(),

    var_weights=np.asarray(df_a['affairs_count'])

).fit()

# Same but using freq_weights

mods_df.loc[6, mod_fields] = ['Aggregated', 'count_mean', 'freq', 'None']

mods_df.loc[6, 'GLMResults'] = smf.glm(

    'affairs_mean ~ rate_marriage + age + yrs_married',

    data=df_a, family=sm.families.Poisson(),

    freq_weights=np.asarray(df_a['affairs_count'])

).fit()



# Aggregated sum with *no* exposure or offset

mods_df.loc[7, mod_fields] = ['Aggregated', 'count_sum', 'None', 'None']

mods_df.loc[7, 'GLMResults'] = smf.glm(

    'affairs_mean ~ rate_marriage + age + yrs_married',

    data=df_a, family=sm.families.Poisson(),

).fit()
summary_df = mods_df.GLMResults.apply({

    'coef': lambda x: x.params,

    'se': lambda x: x.bse,

    'pvalue': lambda x: x.pvalues,

    'stats': lambda r: pd.Series({

        'loglik': r.llf,

        'deviance': r.deviance,

        'chi2': r.pearson_chi2

    }),

    'model': lambda r: pd.Series({

        'df_model': r.model.df_model,

        'df_resid': r.model.df_resid,

        'chi2': r.pearson_chi2

    }),

}).set_index(pd.MultiIndex.from_frame(mods_df[mod_fields])).T

summary_df.style.format('{:,.4f}').apply(

    lambda row_sers: [

        'background-color: yellow' if not_equal else '' 

        for not_equal in np.append(False, np.abs(

            row_sers.values[1:] - row_sers.values[:-1]) > 1e-8)

    ], axis=1

)
assert np.abs(mods_df.GLMResults[3].model.exposure - np.log(df_a['affairs_count'])).max() < 1e-15

print("Note: The resulting `exposure` attribute is the log of the exposure data field")
# When *predicting* using the results of the models, exposure is *not*

# permitted as an input. It is assumed to be 1 (i.e. log of it is zero).

# In that way, you are getting out a rate over a period of 1 = a count.

assert np.max(np.abs(

    mods_df.GLMResults[4].predict(dc[explanatory_vars]) - 

    mods_df.GLMResults[5].predict(dc[explanatory_vars])

)) < 1e-10

assert np.max(np.abs(

    mods_df.GLMResults[3].predict(dc[explanatory_vars]) - 

    mods_df.GLMResults[5].predict(dc[explanatory_vars])

)) < 1e-10



# However, the fittedvalues are targetting the response

# e.g. 'Aggregated sum with exposure', fitted values *include* the exposure

assert np.max(np.abs(

    mods_df.GLMResults[3].fittedvalues / df_a.affairs_count - 

    mods_df.GLMResults[5].fittedvalues

)) < 1e-10

assert np.max(np.abs(

    mods_df.GLMResults[4].fittedvalues / df_a.affairs_count - 

    mods_df.GLMResults[5].fittedvalues

)) < 1e-10

print("Correct: Assertions have passed")
# For target of count_sum, the fittedvalues are already adjusted

# but the predicted values assume a period of 1, so have to be adjusted

assert np.abs(mods_df.GLMResults[3].fittedvalues.sum() - data.affairs.sum()) < 1e-10

assert np.abs(

    (mods_df.GLMResults[3].predict(df_a[explanatory_vars]) * df_a.affairs_count).sum() -

    data.affairs.sum()

) < 1e-10

assert np.abs(mods_df.GLMResults[4].fittedvalues.sum() - data.affairs.sum()) < 1e-10

assert np.abs(

    (mods_df.GLMResults[4].predict(df_a[explanatory_vars]) * df_a.affairs_count).sum() -

    data.affairs.sum()

) < 1e-10



# For target of count_mean, both need to be adjusted

assert np.abs(

    (mods_df.GLMResults[5].fittedvalues * df_a.affairs_count).sum() - 

    data.affairs.sum()

) < 1e-10

assert np.abs(

    (mods_df.GLMResults[5].predict(df_a[explanatory_vars]) * df_a.affairs_count).sum() -

    data.affairs.sum()

) < 1e-10

print("Correct: Assertions have passed")
# Reminder of what each model is

mods_df
print("==== Break deviance down into individual deviances ====")

# Original model

D_indiv_0 = 2 * (

    data.affairs * np.log(

        data.affairs / mods_df.GLMResults[0].fittedvalues,

        # Extra code to cope with: 0 * ln(0) = 0

        out=np.zeros(len(data.affairs)),

        where=data.affairs != 0

    ) - data.affairs + mods_df.GLMResults[0].fittedvalues

)

assert np.abs(D_indiv_0.sum() - mods_df.GLMResults[0].deviance) < 1e-10



# Model with exposure

D_indiv_3 = 2 * (

    df_a.affairs_sum * np.log(

        df_a.affairs_sum / mods_df.GLMResults[3].fittedvalues,

        out=np.zeros(len(df_a.affairs_sum)),

        where=df_a.affairs_sum != 0

    ) - df_a.affairs_sum + mods_df.GLMResults[3].fittedvalues

)

assert np.abs(D_indiv_3.sum() - mods_df.GLMResults[3].deviance) < 1e-10



# Model with var_weights

D_indiv_5 = 2 * df_a.affairs_count * (

    df_a.affairs_mean * np.log(

        df_a.affairs_mean / mods_df.GLMResults[5].fittedvalues,

        out=np.zeros(len(df_a.affairs_mean)),

        where=df_a.affairs_mean != 0

    ) - df_a.affairs_mean + mods_df.GLMResults[5].fittedvalues

)

assert np.abs(D_indiv_5.sum() - mods_df.GLMResults[5].deviance) < 1e-10

print("Correct: All assertions passed")
print("==== Recreate residuals to understand them ====")

print("Tests done on each of 'exposure' and 'var_weights' models\n")

assert np.max(np.abs(

    mods_df.GLMResults[3].resid_response -

    (df_a.affairs_sum - mods_df.GLMResults[3].fittedvalues)

)) < 1e-15

assert np.max(np.abs(

    mods_df.GLMResults[5].resid_response -

    (df_a.affairs_mean - mods_df.GLMResults[5].fittedvalues)

)) < 1e-15

print("resid_response: y_true - fittedvalues (= endog - fittedvalues)\n")



assert np.max(np.abs(

    mods_df.GLMResults[3].resid_pearson -

    (df_a.affairs_sum - mods_df.GLMResults[3].fittedvalues) / 

    np.sqrt(mods_df.GLMResults[3].fittedvalues)

)) < 1e-15

assert np.max(np.abs(

    mods_df.GLMResults[5].resid_pearson -

    (df_a.affairs_mean - mods_df.GLMResults[5].fittedvalues) /

    np.sqrt(mods_df.GLMResults[5].fittedvalues / df_a.affairs_count)

)) < 1e-14

print(

    "resid_pearson: "

    "(y_true - fittedvalues) / sqrt(var(fittedvalues) / var_weights)\n"

    "\t= (y_true - fittedvalues) / sqrt(fittedvalues / var_weights)\n"

    "\t(since var(x) = x for Poisson regression)\n"

    "\t= resid_response adjusted for variance at different values of response\n"

)



assert np.max(np.abs(

    mods_df.GLMResults[3].get_influence().resid_studentized - (

        mods_df.GLMResults[3].resid_pearson / 

        np.sqrt(1 - mods_df.GLMResults[3].get_influence().hat_matrix_diag)

    )

)) < 1e-15

assert np.max(np.abs(

    mods_df.GLMResults[5].get_influence().resid_studentized - (

        mods_df.GLMResults[5].resid_pearson / 

        np.sqrt(1 - mods_df.GLMResults[3].get_influence().hat_matrix_diag)

    )

)) < 1e-15

print(

    "resid_studentized: resid_pearson / sqrt(1 - h_i)\n"

    "\t= (y_true - fittedvalues) / sqrt(var(fittedvalues) * (1 - h_i) / var_weights)\n"

    "\t= resid_response adjusted for variance at different values of response *and* leverage of point\n"

    "\t(leverage = is the observation unusual given the distn of X?)\n"

    "*Note*: These are sometimes called 'standardized peason residuals'\n"

)



assert np.max(np.abs(

    np.sign(

        df_a.affairs_sum - mods_df.GLMResults[3].fittedvalues

    ) * np.sqrt(D_indiv_3) - mods_df.GLMResults[3].resid_deviance

)) < 1e-12

assert np.max(np.abs(

    np.sign(

        df_a.affairs_mean - mods_df.GLMResults[5].fittedvalues

    ) * np.sqrt(D_indiv_5) - mods_df.GLMResults[5].resid_deviance

)) < 1e-12

print(

    "resid_deviance: sign(y_true - fittedvalues) * sqrt(D_i)\n"

    "\t= sign(y_true - fittedvalues) * sqrt(2 * var_weight_i * (ll(y_true_i, y_true_i) - ll(y_true_i, y_pred_i)))\n"

    "\t= sign(y_true - fittedvalues) * sqrt(2 * var_weight_i * d(y_true_i, y_pred_i))"

)