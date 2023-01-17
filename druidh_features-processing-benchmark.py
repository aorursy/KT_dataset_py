import time
import numpy as np
import pandas as pd

tr = pd.read_csv('../input/training_set.csv')
groups = tr.groupby(['object_id'])
# Version using pandas Series
def compute_all_aggregated_features(df):
    # Compute weighted mean
    a_s = df['flux'] * np.power(df['flux'] / df['flux_err'], 2)
    b_s = np.power(df['flux'] / df['flux_err'], 2)
    wmean = np.sum(a_s) / np.sum(b_s)

    flux_med = np.median(df['flux'])
    # Compute normed flux
    normed_flux = (df['flux'].max() - df['flux'].min()) / wmean

    # normed_median_flux
    normed_median_flux = np.median(np.abs(df['flux'] - flux_med) / wmean)
    
    return pd.Series([wmean, flux_med, normed_flux, normed_median_flux])

# Same as above but using arrays
def compute_all_aggregated_features_v2(df):
    # Compute weighted mean
    a_s = df['flux'].values * np.power(df['flux'].values / df['flux_err'].values, 2)
    b_s = np.power(df['flux'].values / df['flux_err'].values, 2)
    wmean = np.sum(a_s) / np.sum(b_s)

    flux_med = np.median(df['flux'].values)
    # Compute normed flux
    normed_flux = (np.max(df['flux'].values) - np.min(df['flux'].values)) / wmean

    # normed_median_flux
    normed_median_flux = np.median(np.abs(df['flux'] - flux_med) / wmean)
    
    return pd.Series([wmean, flux_med, normed_flux, normed_median_flux])

# Arrays version + optimizations
def compute_all_aggregated_features_v3(df):
    flux = df['flux'].values
    flux_err = df['flux_err'].values
    # Compute weighted mean
    b_s = np.power(flux / flux_err, 2)
    a_s = flux * b_s
    wmean = np.sum(a_s) / np.sum(b_s)

    flux_med = np.median(flux)
    # Compute normed flux
    normed_flux = (np.max(flux) - np.min(flux)) / wmean

    # normed_median_flux
    normed_median_flux = np.median(np.abs(flux - flux_med) / wmean)
    
    return pd.Series([wmean, flux_med, normed_flux, normed_median_flux])

t1 = time.time()
z1 = groups.apply(compute_all_aggregated_features)
t2 = time.time()
z2 = groups.apply(compute_all_aggregated_features_v2)
t3 = time.time()
z3 = groups.apply(compute_all_aggregated_features_v3)
t4 = time.time()
print('pandas.Series: {0:.3f} s'.format(t2-t1))
print('arrays: {0:.3f} s'.format(t3-t2))
print('arrays+optimizations: {0:.3f} s'.format(t4-t3))

# Check results
for col in z1.columns:
    a = np.abs(z1[col] - z2[col])
    b = np.abs(z1[col] - z3[col])
    print(col, np.max(a), np.max(b))
def get_initial_aggregations():
    return {
        'flux': ['min', 'max', 'mean', 'median', 'std']
    }

def get_initial_aggregations_v2(df):
    flux = df['flux'].values
    return pd.Series([np.min(flux), np.max(flux), np.mean(flux), np.median(flux), np.std(flux, ddof=1)])

t1 = time.time()
z1 = groups.agg(get_initial_aggregations())
t2 = time.time()
z2 = groups.apply(get_initial_aggregations_v2)
t3 = time.time()
print('aggregations: {0:.3f} s'.format(t2-t1))
print('apply with arrays: {0:.3f} s'.format(t3-t2))
# Check results
z1.columns = ['_'.join([i, j]) for i, j in z1.columns]
z2.columns = z1.columns
for col in z1.columns:
    a = np.abs(z1[col] - z2[col])
    print(col, np.max(a))