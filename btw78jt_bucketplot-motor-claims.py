# Set warning messages

import warnings

# Show all warnings in IPython

warnings.filterwarnings('always')

# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Other warnings that sometimes come up

warnings.filterwarnings("ignore", message="unclosed file <_io.TextIOWrapper")

warnings.filterwarnings("ignore", message="Anscombe residuals currently unscaled")
# Determine whether this notebook is running on Kaggle

from pathlib import Path



on_kaggle = False

print("Current working directory: " + str(Path('.').absolute()))

if str(Path('.').absolute()) == '/kaggle/working':

    on_kaggle = True
# Import built-in modules

import sys

import platform

import os

from pathlib import Path

import functools

import inspect



# Import external modules

from IPython import __version__ as IPy_version

import numpy as np

import pandas as pd

import bokeh

import bokeh.palettes

import bokeh.io

import bokeh.plotting

from sklearn import __version__ as skl_version

from sklearn.model_selection import train_test_split

import matplotlib as mpl

import matplotlib.pyplot as plt

from statsmodels import __version__ as sm_version

import statsmodels.api as sm

import statsmodels.formula.api as smf



# Import project modules

if not on_kaggle:

    # Allow modules to be imported relative to the project root directory

    from pyprojroot import here

    root_dir_path = here()

    if not sys.path[0] == root_dir_path:

        sys.path.insert(0, str(root_dir_path))

import bucketplot as bplt



# For development, allow the project modules to be reloaded every time they are used

%load_ext autoreload

%aimport bucketplot

%autoreload 1



# Check they have loaded and the versions are as expected

assert platform.python_version_tuple() == ('3', '6', '6')

print(f"Python version:\t\t{sys.version}")

assert IPy_version == '7.13.0'

print(f'IPython version:\t{IPy_version}')

assert np.__version__ == '1.18.2'

print(f'numpy version:\t\t{np.__version__}')

assert pd.__version__ == '0.25.3'

print(f'pandas version:\t\t{pd.__version__}')

assert bokeh.__version__ == '2.0.1'

print(f'bokeh version:\t\t{bokeh.__version__}')

assert skl_version == '0.22.2.post1'

print(f'sklearn version:\t{skl_version}')

assert mpl.__version__ == '3.2.1'

print(f'matplotlib version:\t{mpl.__version__}')

assert sm_version == '0.11.0'

print(f'statsmodels version:\t{sm_version}')

print(f'bucketplot version:\t{bplt.__version__}')
# Set the matplotlib defaults

plt.style.use('seaborn')

plt.rcParams['figure.figsize'] = 8, 6



# Load Bokeh for use in a notebook

bokeh.io.output_notebook()
# Output exact environment specification, in case it is needed later

print("Capturing full package environment spec")

print("(But note that not all these packages are required)")

!pip freeze > requirements_snapshot.txt

!jupyter --version > jupyter_versions_snapshot.txt
# Configuration variables

if on_kaggle:

    claims_data_filepath = Path('/kaggle/input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv')

else:

    claims_data_filepath = Path('freMTPL2freq.csv')

if claims_data_filepath.is_file():

    print("Correct: CSV file is available for loading")

else:

    print("Warning: CSV file not yet available in that location")
expected_dtypes = {

    **{col: np.dtype('int64') for col in [

        'IDpol', 'ClaimNb', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']},

    **{col: np.dtype('float64') for col in ['Exposure']},

    **{col: np.dtype('O') for col in ['Area', 'VehBrand', 'VehGas', 'Region']},

}
%%time

# The first download can take approx 1 min on Binder

if not claims_data_filepath.is_file():

    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():

        warnings.filterwarnings(

            "ignore", category=UserWarning,

            message='Version 1 of dataset freMTPL2freq is inactive'

        )

        print("Fetching data...")

        df_raw = fetch_openml(data_id=41214, as_frame=True, cache=False).frame.apply(

            lambda col_sers: col_sers.astype(expected_dtypes[col_sers.name])

        ).sort_values('IDpol').reset_index(drop=True)

    # Cache data within the repo, so we don't have to download it many times

    print("Saving data...")

    df_raw.to_csv(claims_data_filepath, index=False)

    print("Save complete")

else:

    print("Loading data from CSV...")

    df_raw = pd.read_csv(

        claims_data_filepath, delimiter=',', dtype=expected_dtypes,

        # Get index sorted with ascending IDpol, just in case it is out or order

    ).sort_values('IDpol').reset_index(drop=True)

    print("Load complete")
# Reasonableness checks that it has loaded as expected

nRows, nCols = (678013, 12)

assert df_raw.shape == (nRows, nCols)

print(f"Correct: Shape of DataFrame is as expected: {nRows} rows, {nCols} cols")

assert df_raw.dtypes.equals(pd.Series(expected_dtypes)[df_raw.columns])

print("Correct: Data types are as expected")

assert df_raw.isna().sum().sum() == 0

print("Correct: There are no missing values in the dataset")
# Hard-coded stats for reasonableness checks

mean_approx = pd.Series({

    'ClaimNb': 0.0532,

    'Exposure': 0.5288,

    'VehPower': 6.45,

    'VehAge': 7.04,

    'DrivAge': 45.50,

    'BonusMalus': 59.8,

    'Density': 1792.,

})
nrows_sample = int(1e4)

if not on_kaggle or on_kaggle:  # <<<<<<<<<<<<<<<<<<< TODO: redo this

    df_raw, df_unused = train_test_split(

        df_raw, train_size=nrows_sample, random_state=35, shuffle=True

    )
# Check it is as expected, within reason

tol_pc = 0.05

df_sample_means = df_raw[mean_approx.index].mean()

assert df_sample_means.between(

    mean_approx * (1 - tol_pc),

    mean_approx * (1 + tol_pc)

).all()

print("Correct: Reasonableness checks have passed")
def get_df_extra(df_raw):

    """

    Given a DataFrame of that contains the raw data columns (and possibly additional columns), 

    return the DataFrame with additional pre-processed columns

    """

    df_extra = df_raw.copy()

    

    # Calculate frequency per year on each row

    df_extra['Frequency'] = df_extra['ClaimNb'] / df_extra['Exposure']

    

    return(df_extra)
# Run pre-processing to get a new DataFrame

df = get_df_extra(df_raw)
expl_var_names = [

    col_name for col_name in df.columns.to_list() 

     if col_name not in ['IDpol', 'ClaimNb', 'Exposure', 'Frequency']

]

print("Explanatory variables\n" + '\t'.join(expl_var_names))

simple_features = expl_var_names[:9]

print("\nOf which the following are simple features\n" + '\t'.join(simple_features))
# Split training into train and test

df_train, df_test = train_test_split(

    df, test_size=0.3, random_state=34, shuffle=True

)

print("Train sample size: " + str(df_train.shape))
# Add indicator column

df = df.assign(

    split=lambda df: np.select(

        [df.index.isin(df_train.index)],

        ['Train'],

        default='Test'

    )

)

df['split'].value_counts()
%%time

GLMres_mean = smf.glm(

    "ClaimNb ~ 1",

    data=df_train, exposure=np.asarray(df_train['Exposure']),

    family=sm.families.Poisson(sm.genmod.families.links.log()),

).fit()

print(GLMres_mean.summary())
# Check that this is the mean model

mean_mod_pred = np.exp(GLMres_mean.params[0])

assert np.abs(

    GLMres_mean.family.link.inverse(GLMres_mean.params[0]) - 

    GLMres_mean.predict(pd.DataFrame([1]))[0]

) < 1e-10

assert np.abs(

    df_train.ClaimNb.sum() / df_train.Exposure.sum() - 

    mean_mod_pred

) < 1e-10

print("Correct: Reasonableness tests have passed")
veh_features =['VehPower', 'VehAge', 'VehBrand', 'VehGas']
%%time

# Takes a few secs

GLMres_veh = smf.glm(

    "ClaimNb ~ " +  ' + '.join(veh_features),

    data=df_train, exposure=np.asarray(df_train['Exposure']),

    family=sm.families.Poisson(sm.genmod.families.links.log()),

).fit()

print(GLMres_veh.summary())
%%time

# Takes approx 10 secs

GLMres_simple = smf.glm(

    "ClaimNb ~ " +  ' + '.join(simple_features),

    data=df_train, exposure=np.asarray(df_train['Exposure']),

    family=sm.families.Poisson(sm.genmod.families.links.log()),

).fit()

print(GLMres_simple.summary())
%%time

# Score all data (training and test)

df = df.assign(

    Freq_pred_mean=lambda df: GLMres_mean.predict(df),

    Freq_pred_veh=lambda df: GLMres_veh.predict(df),

    Freq_pred_simple=lambda df: GLMres_simple.predict(df),

)
# Check reasonableness

# The actual sum of ClaimNB should exactly match each model's predicted sum on the training data

pred_claims_df = df.assign(

    ClaimNb_pred_mean=lambda df: df['Freq_pred_mean'] * df['Exposure'],

    ClaimNb_pred_veh=lambda df: df['Freq_pred_veh'] * df['Exposure'],

    ClaimNb_pred_simple=lambda df: df['Freq_pred_simple'] * df['Exposure'],

).groupby('split').agg(

    n_obs=('split', 'size'),

    Exposure=('Exposure', 'sum'),

    ClaimNb=('ClaimNb', 'sum'),

    ClaimNb_pred_mean=('ClaimNb_pred_mean', 'sum'),

    ClaimNb_pred_veh=('ClaimNb_pred_veh', 'sum'),

    ClaimNb_pred_simple=('ClaimNb_pred_simple', 'sum'),

)

assert np.max(np.abs(

    pred_claims_df.loc['Train'].iloc[-3:] - pred_claims_df.loc['Train', 'ClaimNb']

)) < 1e-8

pred_claims_df
x_var, stat_var = 'DrivAge', 'Frequency'

df.plot.scatter(x_var, stat_var)

plt.show()
# Look at first few rows

df.head()
# Weights

# df['Exposure'] - none are zero

# df['ClaimNb'] - many are zero



# Numeric

# df['Density'] - close to continuous with a large skew

# df['DrivAge'] - between discrete (ordinal) and continuous

# df['Exposure'] - close to continuous, odd distribution

# df['Frequency'] - continuous with a concentration at 0

# df['Freq_pred_veh'] and df['Freq_pred_simple'] - continuous and positive



# Non-numeric

# df['Area'] - nominal with a low number of levels

# df['Region'] and df['VehBrand'] - nominal with a higher number of levels
def divide_n(df, bucket_var, n_bins=10, bucket_col='bucket'):

    """

    Assign each row of `df` to a bucket by dividing the range of the 

    `bucket_var` column into `n_bins` number of equal width intervals.

    

    df: DataFrame

    bucket_var: Name of the column of df to use for dividing

    n_bins: positive integer number of buckets

    bucket_col: Name of the resulting `bucket` column

    

    Returns: df with the additional `bucket` column 

        The `bucket` column is Categorical data type consisting of Intervals

        that partition the interval from just below min(bucket_var) to 

        max(bucket_var).

    """

    df_w_buckets = df.assign(**{bucket_col: (

        lambda df: pd.cut(df[bucket_var], bins=n_bins)

    )})

    return(df_w_buckets)
bucket_var = 'Density'

# bucket_var = 'Exposure'

# bucket_var = 'DrivAge'

tmp1 = df.pipe(divide_n, bucket_var, 10)

tmp1.groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('Exposure', 'sum'),

    stat_sum=('ClaimNb', 'sum'),

    x_min=(bucket_var, 'min'),

    x_max=(bucket_var, 'max'),

)
# Edge cases

# Resulting bucket with no obs

pd.Series([0, 1]).to_frame('val').pipe(

    divide_n, 'val', 3

).groupby('bucket').agg(n_rows=('bucket', 'size'))



# Constant bucket_var

pd.Series([0, 0]).to_frame('val').pipe(

    divide_n, 'val', 2

).groupby('bucket').agg(n_rows=('bucket', 'size'))



# n_bins = 1

pd.Series([0, 1]).to_frame('val').pipe(

    divide_n, 'val', 1

).groupby('bucket').agg(n_rows=('bucket', 'size'))
# Missing vals

unit_w_miss = pd.Series([0, 1, np.nan]).to_frame('val').pipe(

    divide_n, 'val', 3

)

display(unit_w_miss)  # Given a bucket 'NaN'...

display(  # ...which is not included after grouping

    unit_w_miss.groupby('bucket').agg(n_rows=('bucket', 'size'))  

)
# Use a missing indicator to cope with missing values

unit_filled = pd.Series([0, 1, np.nan]).to_frame('val').assign(

    val_miss_ind=lambda df: df.val.isna() + 0,

    val=lambda df: df.val.fillna(0),

).pipe(

    divide_n, 'val', 3).pipe(

    # Would be more natural to use all_levels() in this case

    divide_n, 'val_miss_ind', 2, 'bucket_miss_ind'  

)

display(unit_filled)

unit_filled.groupby(['bucket_miss_ind', 'bucket']).agg(

    n_rows=('bucket', 'size')

)
def custom_width(

    df, bucket_var,

    width, boundary=0,

    first_break=None, last_break=None,

    bucket_col='bucket'

):

    """

    Assign each row of `df` to a bucket by dividing the range of the 

    `bucket_var` column into `n_bins` number of equal width intervals.

    

    df: DataFrame

    bucket_var: Name of the column of df to use for dividing.

    width: Positive width of the buckets

    boundary: Edge of one of the buckets, if the data extended that far

    first_break: All values below this (if any) are grouped into one bucket

    last_break: All values above this (if any) are grouped into one bucket

    bucket_col: Name of the resulting `bucket` column

    

    Returns: df with the additional `bucket` column 

        The `bucket` column is Categorical data type consisting of Intervals

        that partition the interval from just below min(bucket_var) to 

        max(bucket_var).

    """

    var_min, var_max = df[bucket_var].min(), df[bucket_var].max()

    extended_min = var_min - 0.001 * np.min([(var_max - var_min), width])



    # Set bucket edges

    start = np.floor((extended_min - boundary) / width) * width + boundary

    stop = np.ceil((var_max - boundary) / width) * width + boundary

    num = int((stop - start) / width) + 1

    breaks_all = np.array([

        extended_min,

        *np.linspace(start, stop, num)[1:-1],

        var_max,

    ])

    

    # Clip lower and upper buckets

    breaks_clipped = breaks_all

    if first_break is not None or last_break is not None:

        breaks_clipped = np.unique(np.array([

            breaks_all.min(),

            *np.clip(breaks_all, first_break, last_break),

            breaks_all.max(),

        ]))

    

    df_w_buckets = df.assign(**{bucket_col: (

        lambda df: pd.cut(df[bucket_var], bins=breaks_clipped)

    )})

    return(df_w_buckets)
# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 3, 17.5, None, None

# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 3, 0.5, 30, 70

# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 100, 0.5, None, None

bucket_var, width, boundary, first_break, last_break = 'Density', 100, 0.5, None, 1500.5

tmp6 = custom_width(df, bucket_var, width, boundary, first_break, last_break)

tmp6_grpd = tmp6.groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('Exposure', 'sum'),

    stat_sum=('ClaimNb', 'sum'),

    x_min=(bucket_var, 'min'),

    x_max=(bucket_var, 'max'),

    x_nunique=(bucket_var, 'nunique'),

).assign(

    bucket_width=lambda df: df.index.categories.length

)

tmp6_grpd
def weighted_quantiles(df, bucket_var, n_bins=10, bucket_wgt=None, bucket_col='bucket'):

    """

    Assign each row of `df` to a bucket by splitting column `bucket_var`

    into `n_bins` weighted quantiles, weighted by `bucket_wgt`.

    

    bucket_var: Column name of the values to find the quantiles.

        Must not be constant (i.e. just one value for all rows).

    n_bins: Target number of quantiles, but could end up with fewer because

        there are only a finite number of potential cut points.

    bucket_wgt: Weights to use to calculate the weighted quantiles.

        If None (default) or 'const' then equal weights are used for all rows.

        Must be non-negative with at least one postive value.

    bucket_col: Name of the resulting `bucket` column.

    

    Returns: df with the additional `bucket` column 

        The `bucket` column is Categorical data type consisting of Intervals

        that partition the interval from 0 to sum(bucket_wgt).

    """

    if bucket_wgt is None:

        bucket_wgt = 'const'

    if bucket_wgt == 'const' and 'const' not in df.columns:

        df = df.assign(const = 1)



    res = df.sort_values(bucket_var).assign(**{

        'cum_rows_' + bucket_wgt: lambda df: (

            df[bucket_wgt].cumsum()

        ),

        # Ensure that the quantiles cannot split rows with the same value of bucket_var

        'cum_' + bucket_wgt: lambda df: (

            df.groupby(bucket_var)['cum_rows_' + bucket_wgt].transform('max')

        ),

        bucket_col: (

            lambda df: pd.cut(df['cum_' + bucket_wgt], bins=n_bins)

        )

    })

    return(res)
# bucket_var, bucket_wgt = 'Density', 'const'

# bucket_var, bucket_wgt = 'Density', 'Exposure'

# bucket_var, bucket_wgt = 'Density', 'Frequency'

# bucket_var, bucket_wgt = 'DrivAge', 'Exposure'

# bucket_var, bucket_wgt = 'Region', 'Exposure'  # Does *not* make sense to order by nominal variable 'Region'

# bucket_var, bucket_wgt = 'Freq_pred_mean', 'Exposure'  # Does *not* make sense for bucket_var to be constant

bucket_var, bucket_wgt = 'Freq_pred_veh', 'Exposure'  # Example for lift chart

tmp2 = weighted_quantiles(df, bucket_var, 8, bucket_wgt)

tmp2_grpd = tmp2.groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('Exposure', 'sum'),

    stat_sum=('ClaimNb', 'sum'),

    x_min=(bucket_var, 'min'),

    x_max=(bucket_var, 'max'),

    x_nunique=(bucket_var, 'nunique'),

)

tmp2.head()
# Cases

# It is still possible to end up with no rows in a bucket

pd.Series([2, 2, 3, 3]).to_frame('val').assign(

    bucket=lambda df: pd.cut(df['val'], bins=5)

).groupby('bucket').agg(n_rows=('bucket', 'size'))
# Illustration of why we don't want to split rows that have the same value of bucket_var

pd.DataFrame({

    'bucket_var': [0, 0, 1],

    'bucket_wgt': [1, 1, 1],

}).sort_values('bucket_wgt').assign(

    cum_wgt_rows=lambda df: df['bucket_wgt'].cumsum(),

    bucket_rows=lambda df: pd.cut(df['cum_wgt_rows'], bins=3),

    cum_wgt=lambda df: df.groupby('bucket_var')['cum_wgt_rows'].transform('max'),

    bucket=lambda df: pd.cut(df['cum_wgt'], bins=3),

)
def all_levels(df, bucket_var, include_levels=None, ret_map=False, bucket_col='bucket'):

    """

    Assign each row of `df` to a bucket according to the unique 

    values of `bucket_var`.

    

    bucket_var: Column name of the values to split on.

        Missing values will not be assigned to an interval.

    include_levels: Level values to guarantee to include 

        even if they do not appear in the values of bucket_var.

        Missing values are ignored.

    ret_map: Whether to also return the bucket_map Series.

    bucket_col: Name of the resulting `bucket` column.

    

    Returns: 

        df with the additional `bucket` column

            The `bucket` column is Categorical data type consisting of 

            Intervals that partition a range, plus possible NaN.

        If ret_map is True, also return a Series mapping bucket values

            to bucket intervals.

    """

    # Format inputs

    if include_levels is not None:

        if not isinstance(include_levels, pd.Series):

            include_levels = pd.Series(include_levels)

    

    # Get the mapping from level value to an appropriate interval

    buckets_vals = pd.concat([

        df[bucket_var], include_levels

    ]).drop_duplicates().sort_values(

    ).reset_index(drop=True).dropna().to_frame('val')

    

    # Add a column of intervals (there may be some intervals with no rows)

    if np.issubdtype(df[bucket_var].dtype, np.number):

        # If the values are numeric then take the smallest width

        min_diff = np.min(np.diff(buckets_vals['val']))

        buckets_map = buckets_vals.assign(

            interval=lambda df: pd.cut(df['val'], pd.interval_range(

                start=df['val'].min() - min_diff/2,

                end=df['val'].max() + min_diff/2,

                freq=min_diff

            ))

        )

    else:

        # If the values are not numeric then take unit intervals

        buckets_map = buckets_vals.assign(

            interval=lambda df: pd.interval_range(start=0., periods=df.shape[0], freq=1.)

        )

    

    # Convert to a Series

    buckets_map = buckets_map.reset_index(drop=True)

    

    # Assign buckets and map to intervals

    res = df.assign(**{bucket_col: lambda df: (

        df[bucket_var].astype(

            # Cast the bucket variable as Categorical

            pd.CategoricalDtype(buckets_map['val'], ordered=True)

        ).cat.rename_categories(

            # Swap the bucket levels with the bucket intervals

            buckets_map.set_index('val')['interval']

        )

    )})

    

    if ret_map:

        return(res, buckets_map)

    return(res)
# bucket_var, include_levels = 'DrivAge', None  # Discrete all levels

bucket_var, include_levels = 'Area', 'X'  # Categorical all levels

# bucket_var, include_levels = 'DrivAge', pd.Series([18.5])

# bucket_var, include_levels = 'Area', np.nan  # With missing vals

# bucket_var, include_levels = 'Area', None  # Slightly different

tmp3, tmp3_bucket_map = df.pipe(all_levels, bucket_var, include_levels, ret_map=True)

tmp3_grpd = tmp3.groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('Exposure', 'sum'),

    stat_sum=('ClaimNb', 'sum'),

    x_min=(bucket_var, 'min'),

    x_max=(bucket_var, 'max'),

    x_nunique=(bucket_var, 'nunique'),

)

# Use the bucket_map to assign labels to each bucket interval

tmp3_grpd.assign(

    x_label=lambda df: pd.Categorical(df.index).rename_categories(

        tmp3_bucket_map.set_index('interval')['val']

    )

)
# Missing vals

unit_w_miss, bucket_map = pd.Series([0, 1, np.nan]).to_frame('val').pipe(

    lambda df: all_levels(df, 'val', ret_map=True)

)

display(unit_w_miss)

display(bucket_map)
# Use a missing indicator to cope with missing values

unit_filled, b_map = pd.Series([0, 1, np.nan]).to_frame('val').assign(

    val_miss_ind=lambda df: df.val.isna(),

    val=lambda df: df.val.fillna(0),

).pipe(divide_n, 'val', 3).pipe(

    all_levels, 'val_miss_ind', bucket_col='bucket_miss_ind', ret_map=True 

)

display(unit_filled)

unit_filled.groupby(['bucket_miss_ind', 'bucket']).agg(

    n_rows=('bucket', 'size')

).assign(y_label=lambda df: (

    df.index.get_level_values('bucket_miss_ind').rename_categories(

        b_map.set_index('interval')['val'].to_dict()

    )

))
# Interesting case: We can now group a nominal variable first by all_levels

# and then by weighted_quantiles, to group the levels in order of increasing

# stat_wgt_av. This is a possible way to group levels of a nominal variable

# that makes sense.

bucket_var, include_levels = 'Region', None

tmp4_grpd = df.pipe(

    all_levels, bucket_var, include_levels

).groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('Exposure', 'sum'),

    stat_sum=('ClaimNb', 'sum'),

    x_min=(bucket_var, 'min'),

    x_max=(bucket_var, 'max'),

    x_nunique=(bucket_var, 'nunique'),

).assign(

    stat_wgt_av=lambda df: df['stat_sum'] / df['stat_wgt_sum']

)

tmp4_grpd
tmp4_grpd.rename_axis(index='index').pipe(

    weighted_quantiles, 'stat_wgt_av', 8, 'stat_wgt_sum'

).groupby('bucket').agg(

    n_obs=('bucket', 'size'),

    stat_wgt_sum=('stat_wgt_sum', 'sum'),

    stat_sum=('stat_sum', 'sum'),

    x_min=('x_min', 'min'),

    x_max=('x_min', 'max'),

    x_nunique=('x_min', 'nunique'),

).assign(

    stat_wgt_av=lambda df: df['stat_sum'] / df['stat_wgt_sum']

).style.bar(subset='stat_wgt_av')
def agg_wgt_av(

    df_w_buckets, stat_wgt=None,

    x_var=None, stat_vars=None,

    bucket_col=None, split_col_val=None,

):

    """

    Group by bucket and calculate aggregate values in each bucket

    

    df_w_buckets: Result of an 'assign_buckets' function.

        i.e. a DataFrame with a `bucket` column the is Categorical

        with Interval categories that partition a range.

        Rows with missing `bucket` value are excluded from the grouping.

    stat_wgt: Weights for the weighted distributions of stat_vars.

        If None (default) then it is set to 'const' and equal weights are used

        for all rows. Must be non-negative with at least one postive value.

    x_var: Column name of variable that will be plotted on the x axis.

        If None, no x axis variables are calculated.

    stat_vars: 

        If None (default) or empty list, no values are calculated.

    bucket_col: Name of bucket column to group by.

        Must be in df_w_buckets. Default 'bucket'.

    split_col_val:

        None (default): Do not split buckets.

        str: Name of column to split buckets by.

        tuple: Name of column, and constant value to assign (default '__all__').

    

    Returns: DataFrame with one row per group and aggregate statistics.

    """

    # Set defaults

    if x_var is None:

        x_var_lst = []

    else:

        x_var_lst = [x_var]

    if stat_wgt is None:

        stat_wgt = 'const'

        df_w_buckets = df_w_buckets.assign(**{stat_wgt: 1})

    if stat_vars is None:

        stat_vars = []

    if bucket_col is None:

        bucket_col = 'bucket'

    if split_col_val is None:

        split_col_lst = []

    

    # Format inputs

    if not isinstance(stat_vars, list):

        stat_vars = [stat_vars]

    if isinstance(split_col_val, str):

        split_col_lst = [split_col_val]

    if isinstance(split_col_val, tuple):

        if len(split_col_val) == 1:

            split_col_val = split_col_val[0], '__all__'

        df_w_buckets = df_w_buckets.assign(**{split_col_val[0]: split_col_val[1]})

        split_col_lst = [split_col_val[0]]

    

    # Variables for which we want the (weighted) distribution in each bucket

    agg_vars_all = stat_vars

    if x_var is not None and np.issubdtype(df_w_buckets[x_var].dtype, np.number):

        agg_vars_all = [x_var] + agg_vars_all

    # Ensure they are unique (and maintain order)

    agg_vars = pd.Series(agg_vars_all).drop_duplicates()

    

    df_agg = df_w_buckets.assign(

        **{col + '_x_wgt': (

            lambda df, col=col: df[col] * df[stat_wgt]

        ) for col in agg_vars},

    ).groupby(

        # Group by the buckets

        [bucket_col] + split_col_lst, sort=False

    ).agg(

        # Aggregate calculation for rows in each bucket

        n_obs=(bucket_col, 'size'),  # It is possible that a bucket contains zero rows

        **{col: (col, 'sum') for col in [stat_wgt]},

        **{stat_var + '_wgt_sum': (

            stat_var + '_x_wgt', 'sum'

        ) for stat_var in agg_vars},

        **{"x_" + func: (x_var, func) 

           for func in ['min', 'max'] for x_var in x_var_lst}

    ).sort_index().assign(

        # Calculate the weighted average of the stats

        **{stat_var + '_wgt_av': (

            lambda df, stat_var=stat_var: df[stat_var + '_wgt_sum'] / df[stat_wgt]

        ) for stat_var in agg_vars},

    )

    

    return(df_agg)
def agg_wgt_av(

    df_w_buckets, stat_wgt=None,

    x_var=None, stat_vars=None,

    bucket_col=None, split_col=None,

):

    """

    Group by bucket and calculate aggregate values in each bucket

    

    df_w_buckets: Result of an 'assign_buckets' function.

        i.e. a DataFrame with a `bucket` column the is Categorical

        with Interval categories that partition a range.

        Rows with missing `bucket` value are excluded from the grouping.

    stat_wgt: Weights for the weighted distributions of stat_vars.

        If None (default) then it is set to 'const' and equal weights are used

        for all rows. Must be non-negative with at least one postive value.

    x_var: Column name of variable that will be plotted on the x axis.

        If None, no x axis variables are calculated.

    stat_vars: 

        If None (default) or empty list, no values are calculated.

    bucket_col: Name of bucket column to group by.

        Must be in df_w_buckets. Default 'bucket'.

    split_col:

        None (default): Do not split buckets.

        str: Name of column to split buckets by.

    

    Returns: DataFrame with one row per group and aggregate statistics.

    """

    # Set defaults

    if x_var is None:

        x_var_lst = []

    else:

        x_var_lst = [x_var]

    if stat_wgt is None:

        stat_wgt = 'const'

        df_w_buckets = df_w_buckets.assign(**{stat_wgt: 1})

    if stat_vars is None:

        stat_vars = []

    if bucket_col is None:

        bucket_col = 'bucket'

    if split_col is None:

        split_col_lst = []

    else:

        split_col_lst = [split_col]

    

    # Format inputs

    if not isinstance(stat_vars, list):

        stat_vars = [stat_vars]

    

    # Variables for which we want the (weighted) distribution in each bucket

    agg_vars_all = stat_vars

    if x_var is not None and np.issubdtype(df_w_buckets[x_var].dtype, np.number):

        agg_vars_all = [x_var] + agg_vars_all

    # Ensure they are unique (and maintain order)

    agg_vars = pd.Series(agg_vars_all).drop_duplicates()

    

    df_agg = df_w_buckets.assign(

        **{col + '_x_wgt': (

            lambda df, col=col: df[col] * df[stat_wgt]

        ) for col in agg_vars},

    ).groupby(

        # Group by the buckets

        [bucket_col] + split_col_lst, sort=False

    ).agg(

        # Aggregate calculation for rows in each bucket

        n_obs=(bucket_col, 'size'),  # It is possible that a bucket contains zero rows

        **{col: (col, 'sum') for col in [stat_wgt]},

        **{stat_var + '_wgt_sum': (

            stat_var + '_x_wgt', 'sum'

        ) for stat_var in agg_vars},

        **{"x_" + func: (x_var, func) 

           for func in ['min', 'max'] for x_var in x_var_lst}

    ).sort_index().assign(

        # Calculate the weighted average of the stats

        **{stat_var + '_wgt_av': (

            lambda df, stat_var=stat_var: df[stat_var + '_wgt_sum'] / df[stat_wgt]

        ) for stat_var in agg_vars},

    )

    

    return(df_agg)
# Example for lift chart

bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'

x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']

tmp7_w_buckets = df.pipe(weighted_quantiles, bucket_var, 4, bucket_wgt)

tmp7_agg_all = tmp7_w_buckets.pipe(agg_wgt_av, stat_wgt, x_var, stat_vars)

tmp7_agg_all
tmp7_agg_split = agg_wgt_av(tmp7_w_buckets, stat_wgt, x_var, stat_vars, split_col='split')

tmp7_agg_split
# Use a missing indicator to cope with missing values

unit_filled, b_map = pd.Series([0, 1, np.nan]).to_frame('val').assign(

    val_miss_ind=lambda df: df.val.isna(),

    val=lambda df: df.val.fillna(-1),

).pipe(divide_n, 'val', 2).pipe(

    all_levels, 'val_miss_ind', bucket_col='bucket_miss_ind', ret_map=True 

)

display(unit_filled)

unit_agg_all = unit_filled.pipe(agg_wgt_av, x_var='val')

display(unit_agg_all)

unit_agg_split = unit_filled.pipe(agg_wgt_av, stat_vars='val', split_col='val_miss_ind')

display(unit_agg_split)
# TODO: Other examples
# TODO: Move to util functions



def expand_lims(df, pct_buffer_below=0.05, pct_buffer_above=0.05, include_vals=None):

    """

    Find the range over all columns of df. Then expand these 

    below and above by a percentage of the total range.

    

    df: Consider all values in all columns

    include_vals: Additional values to consider

    

    Returns: Series with rows 'start' and 'end' of the expanded range

    """

    # If a Series is passed, convert it to a DataFrame

    try:

        df = df.to_frame()

    except:

        pass

    # Case where df has no columns, just fill in default vals

    if df.shape[1] == 0:

        res_range = pd.Series({'start': 0, 'end': 1})

        return(res_range)

    if include_vals is None:

        include_vals = []

    if not isinstance(include_vals, list):

        include_vals = [include_vals]

    

    res_range = pd.concat([

        df.reset_index(drop=True),

        # Add a column of extra values to the DataFrame to take these into account

        pd.DataFrame({'_extra_vals': include_vals}),

    ], axis=1).apply(

        # Get the range (min and max) over the DataFrame

        ['min', 'max']).agg({'min': 'min', 'max': 'max'}, axis=1).agg({

        # Expanded range

        'start': lambda c: c['max'] - (1 + pct_buffer_below) * (c['max'] - c['min']),

        'end': lambda c: c['min'] + (1 + pct_buffer_above) * (c['max'] - c['min']),

    })

    return(res_range)
# Example lift chart

bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'

x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']

tmp8_agg = df.pipe(

    weighted_quantiles, bucket_var, 10, bucket_wgt

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars

)

tmp8_agg.assign(

    # Get the coordinates for plot: interval edges

    x_left=lambda df: df.index.categories.left,

    x_right=lambda df: df.index.categories.right,

    x_point=lambda df: (df['x_right'] + df['x_left'])/2.,

)
# Functions to set the x-axis edges `x_left` and `x_right`

def x_edges_min_max(df_agg):

    """

    Set the x-axis edges to be the min and max values of `x_var`.

    Does not make sense to use this option when min and max are not numeric.

    This might result in zero width intervals, in which case a warning is given.

    """

    if not np.issubdtype(df_agg['x_min'].dtype, np.number):

        raise ValueError(

            "\n\tx_edges_min_max: This method can only be used when"

            "\n\tx_min and x_max are numeric data types."

        )

        

    if (df_agg['x_min'] == df_agg['x_max']).any():

        warning(

            "x_edges_min_max: At least one bucket has x_min == x_max, "

            "so using this method will result in zero width intervals."

        )

    

    res = df_agg.assign(

        # Get the coordinates for plot: interval edges

        x_left=lambda df: df['x_min'],

        x_right=lambda df: df['x_max'],

    )

    return(res)





def x_edges_interval(df_agg, bucket_col='bucket'):

    """Set the x-axis edges to be the edges of the bucket interval"""

    res = df_agg.assign(

        x_left=lambda df: [intval.left for intval in df.index.get_level_values(bucket_col)],

        x_right=lambda df: [intval.right for intval in df.index.get_level_values(bucket_col)],

    )

    return(res)





def x_edges_unit(df_agg, bucket_col='bucket'):

    """

    Set the x-axis edges to be the edges of equally spaced intervals

    of width 1.

    """

    res = df_agg.assign(

        x_left=lambda df: pd.Categorical(df.index.get_level_values(bucket_col)).codes,

        x_right=lambda df: df['x_left'] + 1,

    )

    return(res)
# Functions to set the x-axis point

def x_point_mid(df_agg):

    """Set the x_point to be mid-way between x_left and x_right"""

    res = df_agg.assign(

        x_point=lambda df: (df['x_left'] + df['x_right']) / 2.

    )

    return(res)



def x_point_wgt_av(df_agg, x_var):

    """

    Set the x_point to be the weighted average of x_var within the bucket,

    weighted by stat_wgt.

    """

    if not (x_var + '_wgt_av') in df_agg.columns:

        raise ValueError(

            "\n\tx_point_wgt_av: This method can only be used when"

            "\n\tthe weighted average has already been calculated."

        )

    

    res = df_agg.assign(

        x_point=lambda df: df[x_var + '_wgt_av']

    )

    return(res)
def x_label_none(df_agg):

    res = df_agg.copy()

    if 'x_label' in df_agg.columns:

        res = res.drop(columns='x_label')

    return(res)



def x_label_map(df_agg, bucket_map, bucket_col='bucket'):

    res = df_agg.assign(

        x_label=lambda df: pd.Categorical(

            df.index.get_level_values(bucket_col)

        ).rename_categories(

            bucket_map.set_index('interval')['val']

        )

    )

    return(res)
# TODO: Fill missing values in these functions

# def y_quad_cumulative(df_agg, stat_wgt, bucket_col='bucket'):

#     res = df_agg.assign(

#         quad_upr=lambda df: df.groupby([bucket_col])[stat_wgt].transform('cumsum'),

#         quad_lwr=lambda df: df.groupby([bucket_col])['quad_upr'].shift(fill_value=0).fillna(method='ffill'),

#     )

#     return(res)



def y_quad_cumulative(df_agg, stat_wgt, bucket_col='bucket'):

    res = df_agg.assign(

        quad_upr=lambda df: df.groupby([bucket_col])[stat_wgt].transform('cumsum'),

        quad_lwr=lambda df: df.groupby([bucket_col])['quad_upr'].shift(fill_value=0),

    )

    return(res)



def y_quad_area(df_agg, stat_wgt, bucket_col='bucket'):

    res = df_agg.assign(

        x_width=lambda df: df['x_right'] - df['x_left'],

        quad_upr=lambda df: df.groupby([bucket_col])[stat_wgt].transform('cumsum') / df['x_width'],

        quad_lwr=lambda df: df.groupby([bucket_col])['quad_upr'].shift(fill_value=0),

    )

    return(res)



def y_quad_proportion(df_agg, stat_wgt, bucket_col='bucket'):

    res = df_agg.assign(

        quad_upr=lambda df: (

            df.groupby([bucket_col])[stat_wgt].transform('cumsum') / 

            df.groupby([bucket_col])[stat_wgt].transform('sum')

        ),

        quad_lwr=lambda df: df.groupby([bucket_col])['quad_upr'].shift(fill_value=0),

    )

    return(res)
# Examples

stat_wgt='Exposure'

tmp7_agg_split.pipe(

    x_edges_interval

#     x_edges_min_max

#     x_edges_min_max

).pipe(

#     y_quad_cumulative, stat_wgt

    y_quad_area, stat_wgt

#     y_quad_proportion, stat_wgt

)
pipe_funcs_df = pd.DataFrame(

    columns=['task', 'func', 'alias'],

    data = [

        ('x_edges', x_edges_interval, ['interval']),

        ('x_edges', x_edges_min_max, ['min_max']),

        ('x_edges', x_edges_unit, ['unit']),

        ('x_point', x_point_mid, ['mid']),

        ('x_point', x_point_wgt_av, ['wgt_av']),

        ('x_label', x_label_none, ['none']),

        ('x_label', x_label_map, ['map']),

        ('y_quad', y_quad_cumulative, ['cum']),

        ('y_quad', y_quad_area, ['area']),

        ('y_quad', y_quad_proportion, ['prop']),

    ],

).assign(

    name=lambda df: df['func'].apply(lambda f: f.__name__),

    arg_names=lambda df: df['func'].apply(

        lambda f: [

            arg_name for arg_name, val 

            in inspect.signature(f).parameters.items()

        ][1:]  # Not the "df" argument

    ),

    req_arg_names=lambda df: df['func'].apply(

        lambda f: [

            arg_name for arg_name, val 

            in inspect.signature(f).parameters.items()

            if val.default == inspect.Parameter.empty

        ][1:]  # Not the "df" argument

    ),

).set_index(['task', 'name'])



pipe_funcs_df
def get_pipeline_func(

    task, search_term,

    kwarg_keys=None, calling_func='',

    pipe_funcs_df=pipe_funcs_df

):

    """

    TODO: Write docstring <<<<<<<<<<<<<

    """

    # Set defaults

    if kwarg_keys is None:

        kwarg_keys = []

    

    # Find function row

    task_df = pipe_funcs_df.loc[task,:]

    func_row = task_df.loc[task_df.index == search_term, :]    

    if func_row.shape[0] != 1:

        func_row = task_df.loc[[search_term in ali for ali in task_df.alias], :]

    if func_row.shape[0] != 1:

        raise ValueError(

            f"\n\t{calling_func}: Cannot find '{search_term}' within the"

            f"\n\tavailable '{task}' pipeline functions."

        )

        

    # Check required arguments are supplied

    for req_arg in func_row['req_arg_names'][0]:

        if not req_arg in kwarg_keys:

            raise ValueError(

                f"\n\t{calling_func}: To use the '{search_term}' as a '{task}' pipeline"

                f"\n\tfunction, you must specify '{req_arg}' as a keyword argument."

            )

    return(func_row['func'][0], func_row['arg_names'][0])
# Examples

# get_pipeline_func('x_edges', 'min_max')

# get_pipeline_func('x_edges', 'x_edges_interval')

# get_pipeline_func('x_point', 'foo', calling_func='from_here')  # Throws an error

get_pipeline_func('x_point', 'wgt_av', ['x_var'])
def add_coords(

    df_agg_all,

    x_edges=None, x_point=None, x_label=None,

    y_quad=None,

    **kwargs,

):

    """

    Given a DataFrame where each row is a bucket, add x-axis 

    properties to be used for plotting. See pipe_funcs_df for 

    available options.

    

    x_edges: How to position the x-axis edges.

        Default: 'interval'

    x_point: Where to position each bucket point on the x-axis.

        Default: 'mid'

    x_label: Option for x-axis label.

        Default: 'none'

    y_quad: How to plot the histogram quads.

        Default: 'cum'

    **kwargs: Additional arguments to pass to the functions.

    """

    # Set variables for use throughout the function

    calling_func = 'add_coords'

    kwarg_keys = list(kwargs.keys())

    

    # Set defaults

    if x_edges is None:

        x_edges = 'interval'

    if x_point is None:

        x_point = 'mid'

    if x_label is None:

        x_label = 'none'

    if y_quad is None:

        y_quad = 'cum'

    

    # Get pipeline functions

    full_func, arg_names = get_pipeline_func('x_edges', x_edges, kwarg_keys, calling_func)

    x_edges_func = functools.partial(full_func, **{

        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))

    })

    

    full_func, arg_names = get_pipeline_func('x_point', x_point, kwarg_keys, calling_func)

    x_point_func = functools.partial(full_func, **{

        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))

    })



    full_func, arg_names = get_pipeline_func('x_label', x_label, kwarg_keys, calling_func)

    x_label_func = functools.partial(full_func, **{

        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))

    })

    

    full_func, arg_names = get_pipeline_func('y_quad', y_quad, kwarg_keys, calling_func)

    y_quad_func = functools.partial(full_func, **{

        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))

    })

    

    # Apply the functions

    res = df_agg_all.pipe(

        lambda df: x_edges_func(df)

    ).pipe(

        lambda df: x_point_func(df)

    ).pipe(

        lambda df: x_label_func(df)

    ).pipe(

        lambda df: y_quad_func(df)

    )

    return(res)
pipe_funcs_df
def create_bplot(

    df_for_plt, stat_wgt, stat_vars,

    cols=bokeh.palettes.Dark2[8],

):

    """Create bucket plot object from aggregated data"""

    # Add a second index level, if it does not have one already

    if len(df_for_plt.index.names) == 1:

        df_for_plt = df_for_plt.assign(

            split='__all__'

        ).set_index([df_for_plt.index, 'split'])

    

    # Set up the figure

    bkp = bokeh.plotting.figure(

        title="Bucket plot", x_axis_label="X-axis name", y_axis_label=stat_wgt, 

        tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",

        plot_width=800, plot_height=500

    )



    # Plot the histogram squares...

    bkp.quad(

        top=df_for_plt['quad_upr'], bottom=df_for_plt['quad_lwr'],

        left=df_for_plt['x_left'], right=df_for_plt['x_right'],

        fill_color="khaki", line_color="white", legend_label="Weight"

    )

    # ...at the bottom of the graph

    bkp.y_range = bokeh.models.ranges.Range1d(

        **expand_lims(df_for_plt[['quad_upr', 'quad_lwr']], 0, 1.2)

    )



    bkp.legend.location = "top_left"

    bkp.legend.click_policy="hide"



    # Plot the weight average statistic points joined by straight lines

    # Set up the secondary axis

    bkp.extra_y_ranges['y_range_2'] = bokeh.models.ranges.Range1d(

        **expand_lims(df_for_plt[[stat_var + '_wgt_av' for stat_var in stat_vars]])

    )

    bkp.add_layout(bokeh.models.axes.LinearAxis(

        y_range_name='y_range_2',

        axis_label="Weighted average statistic"

    ), 'right')



    for var_num, stat_var in enumerate(stat_vars):

        for split_level in df_for_plt.index.levels[1]:

            # The following parameters need to be passed to both circle() and line()

            stat_line_args = {

                'x': df_for_plt.xs(split_level, level=1)['x_point'],

                'y': df_for_plt.xs(split_level, level=1)[stat_var + '_wgt_av'],

                'y_range_name': 'y_range_2',

                'color': cols[var_num],

                'legend_label': stat_var,

            }

            bkp.circle(**stat_line_args, size=4)

            bkp.line(**stat_line_args)

    

    return(bkp)
# Example lift chart

bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'

x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']

tmp8_for_plt = df.pipe(

    weighted_quantiles, bucket_var, 10, bucket_wgt

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars, 

    split_col='split'

).pipe(

    add_coords, stat_wgt=stat_wgt, y_quad='prop', x_edges='min_max'

)

bkp = create_bplot(tmp8_for_plt, stat_wgt, stat_vars)

bokeh.plotting.show(bkp)
help(add_coords)
x_var, stat_wgt, stat_vars = 'DrivAge', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']

bucket_var = 'DrivAge'

df_for_plt = df.pipe(

#     divide_n, bucket_var, 10

#     all_levels, bucket_var

#     custom_width, bucket_var, 3, 17.5

    custom_width, bucket_var, 3, 17.5, None, 68.5

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars,

    # split_col='split'

).pipe(

    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',

    y_quad='area', 

    # x_edges='unit'

)

bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)

bkp.legend.location = "top_right"

bokeh.plotting.show(bkp)
x_var, stat_wgt, stat_vars = 'Density', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']

bucket_var, bucket_wgt = x_var, stat_wgt

df_for_plt = df.pipe(

#     divide_n, bucket_var, 10

    custom_width, bucket_var, 100, 0.5, None, 1000

#     weighted_quantiles, bucket_var, 10, bucket_wgt

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars,

    split_col='split'

).pipe(

    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',

#     x_edges='min_max', x_point='wgt_av', x_var=x_var,

#     y_quad='area',

    x_edges='unit',

)

bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)

#bkp.legend.location = "top_right"

bokeh.plotting.show(bkp)
df_for_plt
x_var, stat_wgt, stat_vars = 'Area', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']

bucket_var, bucket_wgt = x_var, stat_wgt

df_for_plt = df.pipe(

    all_levels, bucket_var

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars,

    split_col='split',

).pipe(

    add_coords, stat_wgt=stat_wgt,

)

bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)

bokeh.plotting.show(bkp)
# Interesting case: Group a nominal variable first by all_levels and then by 

# weighted_quantiles, to group the levels in order of increasing stat_wgt_av.

x_var, stat_wgt, stat_vars = 'Region', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']

bucket_var, bucket_wgt = x_var, stat_wgt

df_for_plt = df.pipe(

    all_levels, bucket_var, bucket_col='split'

).pipe(

    agg_wgt_av, stat_wgt, x_var, stat_vars, bucket_col='split'

).pipe(

    weighted_quantiles, 'Frequency_wgt_av', 5, stat_wgt

).pipe(

    agg_wgt_av, stat_wgt, 'Frequency_wgt_av',

    stat_vars=['Frequency_wgt_av', 'Freq_pred_simple_wgt_av'],

    # split_col='split' # NOT CURRENTLY WORKING PROPERLY

).pipe(

    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',

)

bkp = create_bplot(df_for_plt, stat_wgt, stat_vars=['Frequency_wgt_av', 'Freq_pred_simple_wgt_av'])

bokeh.plotting.show(bkp)
# Example of allowing an additional var for grouping as missing_ind

miss_ind_grpd = pd.Series([0, 1, np.nan]).to_frame('val').assign(

    missing_ind=lambda df: df['val'].isna(),  # Get missing_ind

    val_filled=lambda df: df['val'].fillna(df['val'].min()),   # Fill missing vals

).pipe(

    lambda df: divide_n(df, 'val_filled', 3)

).groupby(['bucket', 'missing_ind']).agg(

    n_rows=('bucket', 'size'),

)

display(miss_ind_grpd)

miss_ind_grpd.unstack(fill_value=0)
# Unfinished functions to merge and split df_agg_all and df_agg_split

def get_agg_all(df_agg, split_col_val=None):

    if split_col_val is None:

        split_col_val = ('split', '__all__')

    df_agg_all = df_agg.xs(

        split_col_val[1], level=split_col_val[0]

    ).dropna(axis=1, how='all')

    return(df_agg_all)



def get_agg_splits(df_agg, split_col_val=None):

    if split_col_val is None:

        split_col_val = ('split', '__all__')

    df_agg_splits = df_agg.loc[

        df_agg.index.get_level_values(split_col_val[0]) != split_col_val[1],:

    ].dropna(axis=1, how='all')

    return(df_agg_splits)



# NOT COMPLETE



def agg_split_merge(

    df_w_buckets, stat_wgt=None,

    x_var=None, stat_vars=None,

    bucket_var=None, split_var=None,

):

    if split_var is None:

        df_agg_all = df_w_buckets.pipe(

            agg_wgt_av, stat_wgt, x_var, stat_vars, bucket_var, split_var

        )

    df_agg_split = df_w_buckets.pipe(

        agg_wgt_av, stat_wgt, None, stat_vars, bucket_var, split_var

    )
# Previous attempt involved passing arguments between functions

BARGS_DEFAULT = {

    'stat_wgt': 'const',

    'bucket_wgt': 'stat_wgt',

    'n_bins': 10,

    'order_by': 'NA',

    'stat_vars': [],

}



def update_bargs(bargs_new, bargs_prev, func_name, bargs_default=BARGS_DEFAULT):

    """

    Update the Bucket arguments so that they can be tracked and passed between functions.

    

    bargs_new: Items to add to bargs, or overwrite previous values

    bargs_prev: Current items in bargs

    bargs_default: Values to take if the corresponding bargs_new value is None

    func_name: To include in the error message.

    

    Specifically, return a dictionary of bargs with the following items:

    Items from bargs_new with non-None values take precedence.

    For items with None value in bargs_new:

        If the key exists in bargs_prev, take that item.

        Else if the key exists in bargs_default, that that item.

        Else throw an error, i.e. every key from bargs_new must be in the output.

    For keys in bargs_prev but not in bargs_new, take that item.

    """

    # Convert input data types

    if bargs_prev is None:

        bargs_prev = dict()

    # Allocate a non-None value for every key in bargs_new or bargs_prev

    res = dict()

    for key in {**bargs_new, **bargs_prev}.keys():

        if key in bargs_new.keys() and bargs_new[key] is not None:

            res[key] = bargs_new[key]

        elif key in bargs_prev.keys():

            res[key] = bargs_prev[key]

        elif key in bargs_default.keys():

            res[key] = bargs_default[key]

        else: 

            raise ValueError(

                f"{func_name}: '{key}' is required but has not been supplied"

            )

    return(res)