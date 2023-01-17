# Set warning messages

import warnings

# Show all warnings in IPython

warnings.filterwarnings('always')

# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Other warnings that sometimes come up

warnings.filterwarnings("ignore", message="unclosed file <_io.TextIOWrapper")
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

from sklearn.model_selection import train_test_split

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from bokeh import __version__ as bk_version

from scipy import __version__ as scipy_version

from statsmodels import __version__ as sm_version

import statsmodels.api as sm

import statsmodels.formula.api as smf

import xgboost as xgb

import scipy.stats as sps



# Import project modules

from bucketplot import __version__ as bplt_version

import bucketplot as bplt



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

assert sns.__version__ == '0.10.0'

print(f'seaborn version:\t{sns.__version__}')

assert bk_version == '2.0.1'

print(f'bokeh version:\t\t{bk_version}')

assert scipy_version == '1.4.1'

print(f'scipy version:\t\t{scipy_version}')

assert sm_version == '0.11.0'

print(f'statsmodels version:\t{sm_version}')

assert bplt_version == '0.0.2'

print(f'bucketplot version:\t{bplt_version}')
# Bokeh imports

from bokeh.layouts import gridplot

from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.models.ranges import Range1d

from bokeh.models.axes import LinearAxis



# Load Bokeh for use in a notebook

from bokeh.io import output_notebook

output_notebook()
# Output exact environment specification, in case it is needed later

print("Capturing full package environment spec")

print("(But note that not all these packages are required)")

!pip freeze > requirements_Kaggle.txt

!jupyter --version > jupyter_versions.txt
input_folder_path = Path('/kaggle/input')

claims_data_filepath = (

    input_folder_path / 'french-motor-claims-datasets-fremtpl2freq' / 'freMTPL2freq.csv'

)



GLM_folder_path = input_folder_path / 'models-of-french-motor-claims'

assert GLM_folder_path.is_dir()

GLM_data_filepath = GLM_folder_path / 'df_validation_GLM_preds.gzip'

assert GLM_data_filepath.is_file()



RF_folder_path = input_folder_path / 'alex-f-french-motor-claims-analysis'

assert RF_folder_path.is_dir()

RF_data_filepath = RF_folder_path / 'Alex_Farquharson_rf_dataframe.gzip'

assert RF_data_filepath.is_file()



XGB_folder_path = input_folder_path / 'chuns-french-motor-claims-project'

assert XGB_folder_path.is_dir()

XGB_data_filepath = XGB_folder_path / 'xgb_filtered_pred_valid_set_new.gzip'

assert XGB_data_filepath.is_file()



print("Correct: All locations are available as expected")
# Load full modelling data set

expected_dtypes = {

    **{col: np.dtype('int64') for col in [

        'IDpol', 'ClaimNb', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']},

    **{col: np.dtype('float64') for col in ['Exposure']},

    **{col: np.dtype('O') for col in ['Area', 'VehBrand', 'VehGas', 'Region']},

}

df_raw = pd.read_csv(claims_data_filepath, delimiter=',', dtype=expected_dtypes)
# Check it has loaded OK

nRows, nCols = (678013, 12)

assert df_raw.shape == (nRows, nCols)

print(f"Correct: Shape of DataFrame is as expected: {nRows} rows, {nCols} cols")

assert df_raw.dtypes.equals(pd.Series(expected_dtypes)[df_raw.columns])

print("Correct: Data types are as expected")

assert df_raw.isna().sum().sum() == 0

print("Correct: There are no missing values in the raw dataset")
# Get index sorted with ascending IDpol, just in case it is out or order

df_all = df_raw.sort_values('IDpol').reset_index(drop=True)



# Proportions we want to split in (must sum to 1)

split_props = pd.Series({

    'train': 0.7,

    'validation': 0.15,

    'holdout': 0.15

})



# Split out training data

df_train, df_not_train = train_test_split(

    df_all, test_size=(1 - split_props['train']), random_state=51, shuffle=True

)

# Split remaining data between validation and holdout

df_validation, df_holdout = train_test_split(

    df_not_train, test_size=split_props['holdout'] / (1 - split_props['train']), random_state=13, shuffle=True

)
# Check all rows have been accounted for

pd.concat([df_train, df_validation, df_holdout]).sort_index().equals(df_all)
# Sort to make it easier to compare

# We know that IDpol is unique

df_validation = df_validation.sort_values('IDpol')

act_ClaimNb_validation = df_validation.ClaimNb.sum()



# Print number of rows and fields

df_validation.shape
expl_var_names = [

    col_name for col_name in df_validation.columns.to_list() 

     if col_name not in ['IDpol', 'ClaimNb', 'Exposure', 'Frequency']

]

print("Explanatory variables\n" + '\t'.join(expl_var_names))
# From GLM

df_GLM_preds = pd.read_pickle(

    GLM_data_filepath

).sort_values('IDpol')
# Reasonableness checks

assert df_GLM_preds.shape[0] == df_validation.shape[0]

assert (df_validation.IDpol == df_GLM_preds.IDpol).all()

assert (df_validation.ClaimNb == df_GLM_preds.ClaimNb).all()

assert df_GLM_preds.iloc[:,:12].equals(df_validation)

print("Correct: Reasonableness checks have passed for the GLM data")
pred_ClaimNb_GLM = (df_GLM_preds.pred_freq * df_validation.Exposure).sum()

print(f"GLM predicted total number of claims:\t{pred_ClaimNb_GLM:.1f}")

print(f"Actual total number of claims:\t\t{act_ClaimNb_validation:.1f}")

print(f"Difference:\t\t\t\t{pred_ClaimNb_GLM - act_ClaimNb_validation:.1f}")
# From GBM (i.e. xgboost)

df_XGB_preds = pd.read_pickle(

    XGB_data_filepath

).sort_values('IDpol').reset_index(drop=True)

# Cast IDpol to integer to match modelling data

df_XGB_preds.IDpol = df_XGB_preds.IDpol.astype(np.dtype('int64'))
# Reasonableness checks

assert df_XGB_preds.shape[0] == df_validation.shape[0]

assert (df_validation.reset_index(

    drop=True).IDpol == df_XGB_preds.IDpol).all()

assert (df_validation.reset_index(

    drop=True).ClaimNb == df_XGB_preds.ClaimNb).all()

assert df_validation.reset_index(drop=True)[

    ['IDpol', 'ClaimNb', 'Exposure']].equals(df_XGB_preds.iloc[:,:3])

print("Correct: Reasonableness checks have passed for the XGB data")

print(

    "Note that, for the XGB data:\n"

    "\t-The index has been reset, but we can match to the validation data by IDpol"

)
pred_ClaimNb_XGB = (df_XGB_preds.pred_ClaimNb * df_validation.reset_index().Exposure).sum()

print(f"XGB predicted total number of claims:\t{pred_ClaimNb_XGB:.1f}")

print(f"Actual total number of claims:\t\t{act_ClaimNb_validation:.1f}")

print(f"Difference:\t\t\t\t{pred_ClaimNb_XGB - act_ClaimNb_validation:.1f}")
# From RF

df_RF_preds = pd.read_pickle(RF_data_filepath)
assert df_RF_preds.shape[0] == df_validation.shape[0]

assert (df_RF_preds.index == df_validation.index).all()

assert (df_RF_preds.ClaimNb == df_validation.ClaimNb).all()

assert np.max(np.abs(

    df_RF_preds.Exposure - df_validation.assign(

        Exp_4dps=lambda x: np.round(x.Exposure, 4)

    ).Exp_4dps

)) < 1e-14

print("Correct: Reasonableness checks have passed for the RF data")

print(

    "Note that, for the RF data:\n"

    "\t-IDpol is not included but we can match to the validation data by index\n"

    "\t-The Exposure field on the RF is rounded to 4dps"

)
pred_ClaimNb_RF = df_RF_preds['Random Forest Predictions'].sum()

print(f"RF predicted total number of claims:\t{pred_ClaimNb_RF:.1f}")

print(f"Actual total number of claims:\t\t{act_ClaimNb_validation:.1f}")

print(f"Difference:\t\t\t\t{pred_ClaimNb_RF - act_ClaimNb_validation:.1f}")
df_validation_all = df_validation.assign(

    act_freq=lambda x: x.ClaimNb / x.Exposure

).merge(

    df_RF_preds.assign(

        RF_pred_freq=lambda x: x['Random Forest Predictions'] / x.Exposure

    )[['RF_pred_freq']],

    how='inner', left_index=True, right_index=True

).merge(

    df_GLM_preds.rename(columns={

        'pred_freq': 'GLM_pred_freq'

    })[['IDpol', 'GLM_pred_freq']],

    how='inner', left_on='IDpol', right_on='IDpol'

).merge(

    df_XGB_preds.assign(

        XGB_pred_freq=lambda x: x.pred_ClaimNb / x.Exposure

    )[['IDpol', 'XGB_pred_freq']],

    how='inner', left_on='IDpol', right_on='IDpol'

)

# Reasonableness checks on the result

assert df_validation_all.shape[0] == df_validation.shape[0]
# Look at result (excluding explanatory variables)

df_validation_all.loc[:, ~df_validation_all.columns.isin(expl_var_names)].head()
stat_cols = ['GLM_pred_freq', 'act_freq']

lift_plt_data_df = bplt.get_agg_plot_data(

    df_validation_all,

    stat_cols=stat_cols,

    stat_wgt='Exposure',

    bucket_wgt='Exposure',

    set_config="lift",

    n_bins=10

)

lift = {

    stat_col: lift_plt_data_df[stat_col + "_wgt_av"].agg(

        lambda x: x.iloc[-1] / x.iloc[0])

    for stat_col in stat_cols

}

print(f'Lift on predicted:\t{lift[stat_cols[0]]:.3f}')

print(f'Lift on actuals:\t{lift[stat_cols[1]]:.3f}')

lift_plt = bplt.create_plot(lift_plt_data_df, stat_cols=stat_cols)

show(lift_plt)
stat_cols = ['RF_pred_freq', 'act_freq']

lift_plt_data_df = bplt.get_agg_plot_data(

    df_validation_all,

    stat_cols=stat_cols,

    stat_wgt='Exposure',

    bucket_wgt='Exposure',

    set_config="lift",

    n_bins=10

)

lift = {

    stat_col: lift_plt_data_df[stat_col + "_wgt_av"].agg(

        lambda x: x.iloc[-1] / x.iloc[0])

    for stat_col in stat_cols

}

print(f'Lift on predicted:\t{lift[stat_cols[0]]:.3f}')

print(f'Lift on actuals:\t{lift[stat_cols[1]]:.3f}')

lift_plt = bplt.create_plot(lift_plt_data_df, stat_cols=stat_cols)

show(lift_plt)
stat_cols = ['XGB_pred_freq', 'act_freq']

lift_plt_data_df = bplt.get_agg_plot_data(

    df_validation_all,

    stat_cols=stat_cols,

    stat_wgt='Exposure',

    bucket_wgt='Exposure',

    set_config="lift",

    n_bins=10

)

lift = {

    stat_col: lift_plt_data_df[stat_col + "_wgt_av"].agg(

        lambda x: x.iloc[-1] / x.iloc[0])

    for stat_col in stat_cols

}

print(f'Lift on predicted:\t{lift[stat_cols[0]]:.3f}')

print(f'Lift on actuals:\t{lift[stat_cols[1]]:.3f}')

lift_plt = bplt.create_plot(lift_plt_data_df, stat_cols=stat_cols)

show(lift_plt)
stat_cols = ['RF_pred_Nb', 'ClaimNb']

lift_plt_data_df = bplt.get_agg_plot_data(

    df_validation_all.assign(

        GLM_pred_Nb=lambda x: x.GLM_pred_freq * x.Exposure,

        RF_pred_Nb=lambda x: x.RF_pred_freq * x.Exposure,

        XGB_pred_Nb=lambda x: x.XGB_pred_freq * x.Exposure,

    ),

    stat_cols=stat_cols,

    stat_wgt=None,

    bucket_wgt='Exposure',

    order_by=stat_cols[0],

    cut_by='cum_wgt',

    #x_axis_var=stat_cols[1],

    n_bins=10

)



# Plot actual average against predicted average

lift_plt = bplt.create_plot(lift_plt_data_df.assign(

    x_left=lambda x: x[stat_cols[1] + '_wgt_av'],

    x_right=lambda x: x[stat_cols[1] + '_wgt_av'],

    x_mid=lambda x: x[stat_cols[1] + '_wgt_av'],

), stat_cols=stat_cols)

show(lift_plt)



# Plot lines of actual average and predicted average

lift_plt = bplt.create_plot(lift_plt_data_df, stat_cols=stat_cols)

show(lift_plt)



# Calculate lift

lift = {

    stat_col: lift_plt_data_df[stat_col + "_wgt_av"].agg(

        lambda x: x.iloc[-1] / x.iloc[0])

    for stat_col in stat_cols

}

print(f'Lift on predicted:\t{lift[stat_cols[0]]:.3f}')

print(f'Lift on actuals:\t{lift[stat_cols[1]]:.3f}')
predictions_df = df_validation_all.assign(

    GLM_pred_Nb=lambda x: x.GLM_pred_freq * x.Exposure,

    RF_pred_Nb=lambda x: x.RF_pred_freq * x.Exposure,

    XGB_pred_Nb=lambda x: x.XGB_pred_freq * x.Exposure,

)

weights_colm = 'Exposure'

predicted_colm, actual_colm = stat_cols

q = 10



# Get weighted quantiles and add grouping to the DataFrame

order = predictions_df[

    weights_colm  # bucket_wgt

].iloc[predictions_df[

    predicted_colm  # order_by

].argsort()].cumsum()

quantiles = np.linspace(0, 1, q + 1)

bins = pd.cut(order / order.iloc[-1], quantiles, labels=False).sort_index()

predictions_df['weighted_cut'] = bins

predictions_df.head()



predicted_mean = []

actual_mean = []

for x in np.arange(10):

    pred = predictions_df[predictions_df['weighted_cut'] == x][predicted_colm].mean()

    predicted_mean.append(pred)

    actual = predictions_df[predictions_df['weighted_cut'] == x][actual_colm].mean()

    actual_mean.append(actual)



# Check on the above

assert (predictions_df.groupby('weighted_cut').agg(

    pred=(predicted_colm, 'mean'),

    actual=(actual_colm, 'mean'),

    n_rows=('IDpol', 'size'),

    pred_sum=(predicted_colm, 'sum'),

    actual_sum=(actual_colm, 'sum'),

).assign(

    pred_mean=lambda x: x.pred_sum / x.n_rows,

    actual_mean=lambda x: x.actual_sum / x.n_rows,

    diff_pred=lambda x: x.pred_mean - predicted_mean,

    diff_actual=lambda x: x.actual_mean - actual_mean,

)[['diff_pred', 'diff_actual']].sum() == [0,0]).all()



means = pd.DataFrame(data = list(zip(predicted_mean,actual_mean)), columns = ['predicted','actual'])



# Plot actual average against predicted average

sns.scatterplot(data=means,x='actual',y='actual')

sns.scatterplot(data=means, x='actual',y='predicted')



# Calculate lift

a = means.iloc[9]['actual'] / means.iloc[0]['actual']

b = means.iloc[9]['predicted'] / means.iloc[0]['predicted']

print(predicted_colm[:-12], 'actual differentiation', a)

print(predicted_colm[:-12], 'model differentiation', b)

print(predicted_colm[:-12], 'factor', b/a)
stat_cols = ['RF_pred_Nb', 'ClaimNb']

lift_plt_data_df = bplt.get_agg_plot_data(

    df_validation_all.assign(

        GLM_pred_Nb=lambda x: x.GLM_pred_freq * x.Exposure,

        RF_pred_Nb=lambda x: x.RF_pred_freq * x.Exposure,

        XGB_pred_Nb=lambda x: x.XGB_pred_freq * x.Exposure,

    ),

    stat_cols=stat_cols,

    stat_wgt=None,

    bucket_wgt='Exposure',

    order_by=stat_cols[0],

    cut_by='cum_wgt',

    #x_axis_var=stat_cols[1],

    n_bins=10

)



# Plot actual average against predicted average

lift_plt = bplt.create_plot(lift_plt_data_df.assign(

    x_left=lambda x: x[stat_cols[1] + '_wgt_av'],

    x_right=lambda x: x[stat_cols[1] + '_wgt_av'],

    x_mid=lambda x: x[stat_cols[1] + '_wgt_av'],

), stat_cols=stat_cols)

show(lift_plt)



# Plot lines of actual average and predicted average

lift_plt = bplt.create_plot(lift_plt_data_df, stat_cols=stat_cols)

show(lift_plt)



# Calculate lift

lift = {

    stat_col: lift_plt_data_df[stat_col + "_wgt_av"].agg(

        lambda x: x.iloc[-1] / x.iloc[0])

    for stat_col in stat_cols

}

print(f'Lift on predicted:\t{lift[stat_cols[0]]:.3f}')

print(f'Lift on actuals:\t{lift[stat_cols[1]]:.3f}')
# Packages needed for this section

import xgboost as xgb

import scipy.stats as sps
# Simulate data

size = 10000



df = pd.DataFrame({

    'x1': sps.randint(low=0, high=2).rvs(size=size, random_state=67),

    'x2': sps.randint(low=0, high=2).rvs(size=size, random_state=92),

    'exposure': sps.uniform(loc=1, scale=9).rvs(size=size, random_state=67) * 0.3,

}).assign(

    frequency=lambda x: np.where((x.x1 == 1) & (x.x2 == 1), 2, 1),

    claims=lambda x: sps.poisson(mu=x.frequency * x.exposure).rvs(size=size, random_state=14),

)
# xgboost: set up

param0 = {

    "objective": "count:poisson",

    "eval_metric": "poisson-nloglik",

    "eta": 1,

    "subsample": 1,

    "colsample_bytree": 1,

    "min_child_weight": 1,

    "max_depth": 2,

    "lambda": 0,

}



# It is a simple pattern in the data, 

# so should be able to get close with few rounds

num_boost_round = 1
# 1: Try to use the 'weight' argument

xgtrain1 = xgb.DMatrix(

    df[['x1', 'x2']],

    label = df.claims,

    weight = df.exposure

)

xgb_mod1 = xgb.train(

    dtrain=xgtrain1, params=param0,

    num_boost_round=num_boost_round,

)

df = df.assign(

    XGB_P1_Freq=xgb_mod1.predict(xgtrain1),

)
# 2: Try to set an offset in the DMatrix

xgtrain2 = xgb.DMatrix(

    df.assign(

        offset=lambda x: np.log(x.exposure)

    )[['x1', 'x2', 'offset']],

    label = df.claims,

)

xgb_mod2 = xgb.train(

    dtrain=xgtrain2, params=param0,

    num_boost_round=num_boost_round,

)

df = df.assign(

    XGB_P2_Freq=xgb_mod2.predict(xgtrain2),

)
# 3: Try to set base margin as exposure

xgtrain3 = xgb.DMatrix(

    df[['x1', 'x2']],

    label = df.claims,

)



xgtrain3_w_bm = xgb.DMatrix(

    df[['x1', 'x2']],

    label = df.claims,

)

xgtrain3_w_bm.set_base_margin(np.log(df.exposure))



assert xgtrain3.get_base_margin().shape[0] == 0

assert np.max(np.abs(xgtrain3_w_bm.get_base_margin() - np.log(df.exposure))) < 1e-6



xgb_mod3 = xgb.train(

    dtrain=xgtrain3_w_bm, params=param0,

    num_boost_round=num_boost_round,

)



df = df.assign(

    # If you do *not* set base margin, the assumption is 0.5 *not* 1

    XGB_P3_Freq_no_exp=lambda x: xgb_mod3.predict(xgtrain3) / 0.5,

    XGB_P3_Freq_w_exp=lambda x: xgb_mod3.predict(xgtrain3_w_bm) / x.exposure,

    

    XGB_P3_Nb_no_exp=lambda x: x.XGB_P3_Freq_no_exp * x.exposure,

    XGB_P3_Nb_w_exp=lambda x: x.XGB_P3_Freq_w_exp * x.exposure,

)
df.groupby(['x1', 'x2']).mean()
fig, ax = plt.subplots(figsize=(20, 10))

xgb.plot_tree(xgb_mod3, num_trees=0, ax=ax)

plt.show()
import inspect

import textwrap

import re

import functools
# Example data

size = 20

example_df = pd.DataFrame({

    'cat': pd.Series(['A','B'])[sps.randint(low=0, high=2).rvs(size=size, random_state=67)],

    'field1': np.linspace(1, 10, size),

    'field2': np.linspace(10, -70, size),

    'exp': sps.uniform(loc=1, scale=9).rvs(size=size, random_state=67) * 0.3,

})
# We want to put the following into a function with variables.

# But we also want to be able to extract this query as code.

example_df.assign(

    wgt=lambda x: x.exp,

    field1_x_exp=lambda x: x.field1 * x.wgt

).groupby('cat').agg(

    wgt_sum=('wgt', 'sum'),

    field1_wgt_sum=('field1_x_exp', 'sum'),

).assign(

    field1_wgt_av=lambda x: x.field1_wgt_sum / x.wgt_sum

)
# Here is the parametrised query

df = example_df

wgt_col = 'exp'

stat_cols = ['field1']

cut_by = 'cat'



df.assign(

    wgt=lambda x: x[wgt_col],

    **{

        stat_col + '_x_exp': 

        lambda x, stat_col=stat_col: x[stat_col] * x.wgt 

        for stat_col in stat_cols

    },

).groupby(cut_by).agg(

    wgt_sum=('wgt', 'sum'),

    **{

        stat_col + '_wgt_sum': (

            stat_col + '_x_exp',

            'sum'

        ) for stat_col in stat_cols

    },

).assign(

    **{

        stat_col + '_wgt_av': 

        lambda x, stat_col=stat_col: x[stat_col + '_wgt_sum'] / x.wgt_sum 

        for stat_col in stat_cols

    },

)
def get_assign_dict(assign_dict, dict_name, replacement_dict):

    replace = "**" + dict_name

    replace_with = ',\n    '.join([

        key + "=" + inspect.getsource(val).strip(

        ).replace(

            "stat_col", f"'{key}'"

        ).replace(

            f", '{key}'='{key}'", ""

        )

        for key, val in assign_dict.items()

    ])

    replacement_dict[replace] = replace_with

    return(assign_dict)
def get_inner_code(

    func,

    from_after_re=r'#\s*<Query begin>.*\n',

    to_before_re=r'\n\s*#\s*<Query end>',

):

    code_raw_str = inspect.getsource(get_wgt_av)

    

    # Remove the first row, and de-indent the remainder

    code_body_str = textwrap.dedent(

        re.sub(r'.+:\n', r'', code_raw_str)

    )

    

    # Find the specified start and end patterns

    from_idx = re.search(from_after_re, code_body_str).end()

    to_idx = re.search(to_before_re, code_body_str).start()

    

    return(code_body_str[from_idx:to_idx])
# Put it in a function

def get_wgt_av(df, wgt_col, stat_cols, cut_by=None, return_code=False):

    # Capture the arg names. Must do this first

    arg_names = list(locals().keys())

    

    # Set default parameter values

    if cut_by is None:

        cut_by = 'cat'

    

    # Create dictionary to map argument names to string values

    replacement_dict = dict()

    for var_name in arg_names:

        var_val = locals()[var_name]

        if var_name in ['df']:

            continue

        if isinstance(var_val, str):

            replacement_dict[var_name] = f"'{var_val}'"

            continue

        replacement_dict[var_name] = var_val

    

    # Unpack iterable arguments

    extra_cols_on_input_df = {

        stat_col + '_x_exp': 

        lambda x, stat_col=stat_col: x[stat_col] * x.wgt 

        for stat_col in stat_cols

    }

    replacement_dict['**extra_cols_on_input_df'] = ',\n    '.join([

        f"{stat_col}_x_exp="

        f"lambda x: x['{stat_col}'] * x.wgt"

        for stat_col in stat_cols

    ])

    

    agg_cols = {

        stat_col + '_wgt_sum': (

            stat_col + '_x_exp', 'sum'

        ) for stat_col in stat_cols

    }

    replacement_dict['**agg_cols'] = ',\n    '.join([

        f"{stat_col}_wgt_sum="

        f"('{stat_col}_x_exp', 'sum')"

        for stat_col in stat_cols

    ])

    

    extra_cols_on_result = {

        stat_col + '_wgt_av': 

        lambda x, stat_col=stat_col: x[stat_col + '_wgt_sum'] / x.wgt_sum 

        for stat_col in stat_cols

    }

    replacement_dict['**extra_cols_on_result'] = ',\n    '.join([

        f"{stat_col}_wgt_av="

        f"lambda x: x['{stat_col}_wgt_sum'] / x.wgt_sum"

        for stat_col in stat_cols

    ])

    

    # If we're just getting the code, no need to run the query below

    if return_code:

        query_code = get_inner_code(get_wgt_av)

        code_w_vals = functools.reduce(

            lambda code_str, arg_item: code_str.replace(*arg_item),

            {key: str(val) for key, val in replacement_dict.items()}.items(),

            query_code

        )

        return(code_w_vals)

    

    # <Query begin> # (This is a special command, do not modify)

    res = df.assign(

        wgt=lambda x: x[wgt_col],

        **extra_cols_on_input_df,

    ).groupby(cut_by).agg(

        wgt_sum=('wgt', 'sum'),

        **agg_cols,

    ).assign(

        **extra_cols_on_result,

    )

    # <Query end> # (This is a special command, do not modify)

    return(res)
# Test it works

code_example = get_wgt_av(

    df = example_df,

    wgt_col = 'exp',

    stat_cols = ['field1'],

    cut_by = 'cat',

    return_code=True

)

print(code_example)
run_this_chunk = True

if run_this_chunk:

    exec(code_example)

    assert res.equals(get_wgt_av(

        df = example_df,

        wgt_col = 'exp',

        stat_cols = ['field1'],

        cut_by = 'cat',

    ))

    print("Correct: Evaluated string equals function result")

    display(res)
# Another test

params = {

    'wgt_col': 'exp',

    'stat_cols': ['field1', 'field2'],

    'cut_by': 'cat'

}

code_example2 = get_wgt_av(

    df = example_df,

    **params,

    return_code=True

)

print(code_example2)
run_this_chunk = True

if run_this_chunk:

    exec(code_example2)

    assert res.equals(get_wgt_av(

        df = example_df,

        **params,

    ))

    print("Correct: Evaluated string equals function result")

    display(res)