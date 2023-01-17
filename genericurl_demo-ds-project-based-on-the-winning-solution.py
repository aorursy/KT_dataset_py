import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numba   # JIT compiler for python

import matplotlib.pyplot as plt  # graphics

import lightgbm as lgb  # Gradient boosting

import scipy.stats  # stats

import gc   # Garabage collector

from sklearn import metrics

import seaborn as sns



RANDOM_SEED = 27



data_dir = '../input/vsb-power-line-fault-detection'
meta_train_df = pd.read_csv(data_dir + '/metadata_train.csv')

meta_test_df = pd.read_csv(data_dir + '/metadata_test.csv')
meta_train_df.head(10)
meta_train_df.shape
meta_train_df.target.value_counts()
meta_train_df.id_measurement.nunique()
meta_train_df.groupby('id_measurement').target.sum().value_counts()
meta_train_df[meta_train_df.target==0].sample(5, random_state=RANDOM_SEED)
meta_train_df[meta_train_df.target==1].sample(5, random_state=RANDOM_SEED)
train_df = pd.read_parquet(data_dir + '/train.parquet')
train_df.shape
negative_signal_ids = meta_train_df[meta_train_df.id_measurement==1287].signal_id.values

positive_signal_ids = meta_train_df[meta_train_df.id_measurement==2649].signal_id.values



negative_sample = train_df.iloc[:, negative_signal_ids].values

positive_sample = train_df.iloc[:, positive_signal_ids].values
plt.figure(figsize=(18, 4))

plt.title('Normal powerline')

plt.plot(negative_sample, alpha=0.8);



plt.figure(figsize=(18, 4))

plt.title('Faulty powerline')

plt.plot(positive_sample, alpha=0.8);
assert numba.__version__ == '0.46.0'
@numba.jit(nopython=True)

def ema_residuals(x, alpha=0.01):

    """

    Flatten signal

    Based on: https://www.kaggle.com/miklgr500/flatiron

    """

    new_x = np.zeros_like(x)

    ema = x[0]

    for i in range(1, len(x)):

        ema = ema*(1-alpha) + alpha*x[i]

        new_x[i] = x[i] - ema

    return new_x
@numba.jit(nopython=True)

def ema(x, alpha=0.01):

    """

    Flatten signal

    Based on: https://www.kaggle.com/miklgr500/flatiron

    """

    new_x = np.zeros_like(x)

    ema = x[0]

    for i in range(1, len(x)):

        ema = ema*(1-alpha) + alpha*x[i]

        new_x[i] = ema

    return new_x
flat_negative_sample = np.zeros_like(negative_sample)

flat_positive_sample = np.zeros_like(positive_sample)



for i in range(3):

    flat_negative_sample[:,i] = ema_residuals(negative_sample[:,i])

    flat_positive_sample[:,i] = ema_residuals(positive_sample[:,i])
plt.figure(figsize=(18, 4))

plt.title('Normal powerline')

plt.plot(flat_negative_sample, alpha=0.8);



plt.figure(figsize=(18, 4))

plt.title('Faulty powerline')

plt.plot(flat_positive_sample, alpha=0.8);
flat_negative_sample = np.zeros_like(negative_sample)

flat_positive_sample = np.zeros_like(positive_sample)



for i in range(3):

    flat_negative_sample[:,i] = ema(negative_sample[:,i])

    flat_positive_sample[:,i] = ema(positive_sample[:,i])
plt.figure(figsize=(18, 4))

plt.title('Normal powerline')

plt.plot(flat_negative_sample, alpha=0.8);



plt.figure(figsize=(18, 4))

plt.title('Faulty powerline')

plt.plot(flat_positive_sample, alpha=0.8);
@numba.jit(nopython=True)

def drop_missing(intersect,sample):

    """

    Find intersection of sorted numpy arrays

    

    Since intersect1d sort arrays each time, it's effectively inefficient.

    Here you have to sweep intersection and each sample together to build

    the new intersection, which can be done in linear time, maintaining order. 



    Source: https://stackoverflow.com/questions/46572308/intersection-of-sorted-numpy-arrays

    Creator: B. M.

    """

    i=j=k=0

    new_intersect=np.empty_like(intersect)

    while i< intersect.size and j < sample.size:

        if intersect[i]==sample[j]: # the 99% case

            new_intersect[k]=intersect[i]

            k+=1

            i+=1

            j+=1

        elif intersect[i]<sample[j]:

            i+=1

        else : 

            j+=1

    return new_intersect[:k]



@numba.jit(nopython=True)

def _local_maxima_1d_window_single_pass(x, w):

    

    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)

    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)

    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)

    m = 0  # Pointer to the end of valid area in allocated arrays



    i = 1  # Pointer to current sample, first one can't be maxima

    i_max = x.shape[0] - 1  # Last sample can't be maxima

    while i < i_max:

        # Test if previous sample is smaller

        if x[i - 1] < x[i]:

            i_ahead = i + 1  # Index to look ahead of current sample



            # Find next sample that is unequal to x[i]

            while i_ahead < i_max and x[i_ahead] == x[i]:

                i_ahead += 1

                    

            i_right = i_ahead - 1

            

            f = False

            i_window_end = i_right + w

            while i_ahead < i_max and i_ahead < i_window_end:

                if x[i_ahead] > x[i]:

                    f = True

                    break

                i_ahead += 1

                

            # Maxima is found if next unequal sample is smaller than x[i]

            if x[i_ahead] < x[i]:

                left_edges[m] = i

                right_edges[m] = i_right

                midpoints[m] = (left_edges[m] + right_edges[m]) // 2

                m += 1

                

            # Skip samples that can't be maximum

            i = i_ahead - 1

        i += 1



    # Keep only valid part of array memory.

    midpoints = midpoints[:m]

    left_edges = left_edges[:m]

    right_edges = right_edges[:m]

    

    return midpoints, left_edges, right_edges



@numba.jit(nopython=True)

def local_maxima_1d_window(x, w=1):

    """

    Find local maxima in a 1D array.

    This function finds all local maxima in a 1D array and returns the indices

    for their midpoints (rounded down for even plateau sizes).

    It is a modified version of scipy.signal._peak_finding_utils._local_maxima_1d

    to include the use of a window to define how many points on each side to use in

    the test for a point being a local maxima.

    Parameters

    ----------

    x : ndarray

        The array to search for local maxima.

    w : np.int

        How many points on each side to use for the comparison to be True

    Returns

    -------

    midpoints : ndarray

        Indices of midpoints of local maxima in `x`.

    Notes

    -----

    - Compared to `argrelmax` this function is significantly faster and can

      detect maxima that are more than one sample wide. However this comes at

      the cost of being only applicable to 1D arrays.

    """    

        

    fm, fl, fr = _local_maxima_1d_window_single_pass(x, w)

    bm, bl, br = _local_maxima_1d_window_single_pass(x[::-1], w)

    bm = np.abs(bm - x.shape[0] + 1)[::-1]

    bl = np.abs(bl - x.shape[0] + 1)[::-1]

    br = np.abs(br - x.shape[0] + 1)[::-1]



    m = drop_missing(fm, bm)



    return m
a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 0, 0, 3, 0, 5, 0, 0, 0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])



p1 = local_maxima_1d_window(a, w=1)

p3 = local_maxima_1d_window(a, w=3)

p4 = local_maxima_1d_window(a, w=4)



plt.plot(a, marker='o')

plt.scatter(p1, a[p1]+0.2, color='red', label='1')

plt.scatter(p3, a[p3]+0.4, color='orange', marker='x', label='3')

plt.scatter(p4, a[p4]+0.6, color='grey', marker='^', label='4')

plt.legend()

plt.show()
def get_peaks(

    x, 

    window=25,

    visualise=False,

    visualise_color=None,

):

    """

    Find the peaks in a signal trace.

    Parameters

    ----------

    x : ndarray

        The array to search.

    window : np.int

        How many points on each side to use for the local maxima test

    Returns

    -------

    peaks_x : ndarray

        Indices of midpoints of peaks in `x`.

    peaks_y : ndarray

        Absolute heights of peaks in `x`.

    x_flatten_abs : ndarray

        An absolute flattened version of `x`.

    """

    

    x_flatten = ema_residuals(x)

    x_flatten_abs = np.abs(x_flatten)

    

    peaks_indices = local_maxima_1d_window(x_flatten_abs, window)

    heights = x_flatten_abs[peaks_indices]

    

    peaks_sorted_indices = np.argsort(heights)[::-1]

    

    peaks_indices = peaks_indices[peaks_sorted_indices]

    heights = heights[peaks_sorted_indices]

    

    ky = heights

    kx = np.arange(1, heights.shape[0]+1)

    

    conv_length = 9



    grad = np.diff(ky, 1)/np.diff(kx, 1)

    grad = np.convolve(grad, np.ones(conv_length)/conv_length)#, mode='valid')

    grad = grad[conv_length-1:-conv_length+1]

    

    knee_x = plateau_detection(grad, -0.01, plateau_length=1000)

    knee_x -= conv_length//2

    

    if visualise:

        plt.plot(grad, color=visualise_color)

        plt.axvline(knee_x, ls="--", color=visualise_color)

    

    peaks_x = peaks_indices[:knee_x]

    peaks_y = heights[:knee_x]

    

    ii = np.argsort(peaks_x)

    peaks_x = peaks_x[ii]

    peaks_y = peaks_y[ii]

        

    return peaks_x, peaks_y





@numba.jit(nopython=True)

def plateau_detection(grad, threshold, plateau_length=5):

    """Detect the point when the gradient has reach a plateau"""

    

    count = 0

    loc = 0

    for i in range(grad.shape[0]):

        if grad[i] > threshold:

            count += 1

        

        if count == plateau_length:

            loc = i - plateau_length

            break

            

    return loc
fig, ax = plt.subplots(1, 1, figsize=(10, 6))



sigids = [2323, 10, 4200, 4225]

colours = ['blue', 'red', 'orange', 'grey']

for i, sigid in enumerate(sigids):

    d = train_df.iloc[:, sigid].values.astype(np.float)

    get_peaks(d, visualise=True, visualise_color=colours[i])



plt.xlim([0, 4000])

plt.axhline(-0.01, color='black', ls='--')

plt.yscale("symlog")

plt.xscale("symlog")



plt.xlabel('Sorted peak index')

plt.ylabel('Gradient')

plt.suptitle('Example of peak filtering')



plt.show()
sigids = [2323, 10, 4200, 4225]



for sigid in sigids:

    d = train_df.iloc[:, sigid].values.astype(np.float)



    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    plt.plot(d, alpha=0.75)



    px, py = get_peaks(d)

    

    plt.scatter(px, d[px], color="red")

    plt.show()
@numba.jit(nopython=True, parallel=True)

def calculate_current_phase(data):

    """Calculate the current phase shift relative to sine wave.

    Assumes the signal is 800000 data points long

    """

    n = 800000

    assert data.shape[0] == n

    

    # uses 50Hz Fourier coefficient

    omegas = np.exp(-2j * np.pi * np.arange(n) / n)

    res = np.zeros(data.shape[1], dtype=omegas.dtype)

    for i in numba.prange(data.shape[1]):

        res[i] = omegas.dot(data[:, i].astype(omegas.dtype))

            

    return np.angle(res, deg=False)





def to_angle(x, phase):

    dt = 1/800000

    return (np.degrees(2*np.pi*dt*x + phase) + 90) % 360
sampl_id = 0



x = train_df.iloc[:,sampl_id:sampl_id+1].values

phase = calculate_current_phase(x)

angles = to_angle(np.arange(800000), phase)



plt.figure(figsize=(18, 4))

plt.plot(x, alpha=.8)



plt.figure(figsize=(18, 4))

plt.plot(angles, x, alpha=.8)
def process_measurement(data_df, meta_df):

    """

    Process three signal traces in measurment to find the peaks

    and calculate features for each peak.

    Parameters

    ----------

    data_df : pandas.DataFrame

        Signal traces.

    meta_df : pandas.DataFrame

        Meta data for measurement

    Returns

    -------

    peaks : pandas.DataFrame

        Data for each peak in the three traces in `data`.

    """

    peaks = []

    for i, sig_id in enumerate(meta_df.signal_id):

        mat = []

        signal = data_df.iloc[:, i].values.astype(np.float)

        px, h = get_peaks(signal)

        mat.append(px)

        mat.append(h)

        mat.append([sig_id]*len(px))

        peaks.append(np.asarray(mat))    



    peaks = pd.DataFrame(

        np.concatenate(peaks, axis=1).T,

        columns=['px', 'height', 'signal_id']

    )



    # Calculate the phase resolved location of each peak

    phase_50hz = calculate_current_phase(data_df.values)



    phase_50hz = pd.DataFrame(

        phase_50hz,

        columns=['phase_50hz']

    )

    phase_50hz['signal_id'] = meta_df['signal_id'].values

    peaks = pd.merge(peaks, phase_50hz, on='signal_id', how='left')



    peaks['phase_aligned_x'] = to_angle(peaks['px'], peaks['phase_50hz'])



    # Calculate the phase resolved quarter for each peak

    peaks['Q'] = pd.cut(peaks['phase_aligned_x'], [0, 90, 180, 270, 360], labels=[0, 1, 2, 3])

    return peaks
train_peaks = process_measurement(train_df, meta_train_df)

train_peaks = pd.merge(train_peaks, meta_train_df[['signal_id', 'id_measurement', 'target']], on='signal_id', how='left')



del train_df

gc.collect()
train_peaks.shape
train_peaks.tail()
def calculate_features(peaks_df, meta_df):

    results = pd.DataFrame(index=meta_df['id_measurement'].unique())

    results.index.rename('id_measurement', inplace=True)





    # Count total peaks for each measurement id

    res = peaks_df.groupby('id_measurement').agg({

        'px': 'count',

    })

    res.columns = ["peak_count_total"]

    results = pd.merge(results, res, on='id_measurement', how='left')





    # Count peaks in phase resolved quarters 0 and 2

    p = peaks_df[peaks_df['Q'].isin([0, 2])].copy()

    res = p.groupby('id_measurement').agg({

        'px': 'count',

    })

    res.columns = ["peak_count_Q02"]

    results = pd.merge(results, res, on='id_measurement', how='left')





    # Count peaks in phase resolved quarters 1 and 3

    p = peaks_df[peaks_df['Q'].isin([1, 3])].copy()

    res = p.groupby('id_measurement').agg({

        'px': 'count',

    })

    res.columns = ['peak_count_Q13']

    results = pd.merge(results, res, on='id_measurement', how='left')





    # Calculate height properties using phase resolved quarters 0 and 2

    p = peaks_df[peaks_df['Q'].isin([0, 2])].copy()

    res = p.groupby('id_measurement').agg({

        'height': ['mean', 'std'],

    })

    res.columns = ["_".join(f) + '_Q02' for f in res.columns]     

    results = pd.merge(results, res, on='id_measurement', how='left')

    

    return results
X_train = calculate_features(train_peaks, meta_train_df)
X_train.head()
X_train[y_train==0].peak_count_total.describe()
X_train[y_train==1].peak_count_total.describe()
y_train = meta_train_df.groupby('id_measurement')['target'].sum() > 0

y_train = y_train.astype(np.float)
X_train.shape
num_folds = 5



np.random.seed(13)



splits = np.zeros(X_train.shape[0], dtype=np.int)

m = y_train == 1

splits[m] = np.random.randint(0, num_folds, size=m.sum())

m = y_train == 0

splits[m] = np.random.randint(0, num_folds, size=m.sum())
pd.Series(splits).value_counts()
params = {

    'objective': 'binary',

    'metric': 'binary_logloss',

    

    'boosting': 'gbdt',

    'learning_rate': 0.01,

    'num_leaves': 80,

    'num_boost_round': 10000,

    

    'feature_fraction': 0.8,

    'bagging_fraction': 0.8,



    'num_threads': 4,

    'seed': 23974,

}
models = []

cv_scores = []

val_cv_scores = []

feature_names = X_train.columns.tolist()



yp_train = np.zeros(X_train.shape[0])

yp_val = np.zeros(X_train.shape[0])

yp_test = np.zeros(X_train.shape[0])



for fold in range(num_folds):

    val_fold = fold

    test_fold = (fold + 1) % num_folds

    train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]



    train_indices = np.where(np.isin(splits, train_folds))[0]

    val_indices = np.where(splits == val_fold)[0]

    test_indices = np.where(splits == test_fold)[0]



    trn = lgb.Dataset(

        X_train.values[train_indices],

        y_train[train_indices],

        feature_name=feature_names,

    )

    val = lgb.Dataset(

        X_train.values[val_indices],

        y_train[val_indices],

        feature_name=feature_names,

    )

    test = lgb.Dataset(

        X_train.values[test_indices],

        y_train[test_indices],

        feature_name=feature_names,

    )



    # train model

    model = lgb.train(

        params, 

        trn, 

        valid_sets=(trn, test, val), 

        valid_names=("train", "test", "validation"), 

        early_stopping_rounds=10,

        verbose_eval=50

    )



    # predict

    yp = model.predict(X_train.values[train_indices])

    yp_train[train_indices] += yp

    yp_val_fold = model.predict(X_train.values[val_indices])

    yp_val[val_indices] += yp_val_fold

    yp_test_fold = model.predict(X_train.values[test_indices])

    yp_test[test_indices] += yp_test_fold

    

    # save 

    models.append(model)

    cv_scores.append(model.best_score['test']['binary_logloss'])

    val_cv_scores.append(model.best_score['validation']['binary_logloss'])



yp_train /= (num_folds - 2)

cv_scores = np.asarray(cv_scores)

val_cv_scores = np.asarray(val_cv_scores)
print("CV Val Logloss: {:.4f} +/- {:.4f} ({:.4f})".format(val_cv_scores.mean(), val_cv_scores.std()/np.sqrt(val_cv_scores.shape[0]), val_cv_scores.std()))

print("CV Test Logloss: {:.4f} +/- {:.4f} ({:.4f})".format(cv_scores.mean(), cv_scores.std()/np.sqrt(cv_scores.shape[0]), cv_scores.std()))



print("Train  accuracy: {:.4f}".format(metrics.accuracy_score(y_train, yp_train > 0.5)))

print("CV Val accuracy: {:.4f}".format(metrics.accuracy_score(y_train, yp_val > 0.5)))

print("CV Test accuracy: {:.4f}".format(metrics.accuracy_score(y_train, yp_test > 0.5)))
thresholds = np.linspace(.01, .9, 90)



scores_train = []

scores_val = []

scores_test = []



for t in thresholds:

    s_train = metrics.f1_score(

        y_train.values.astype(np.float), 

        yp_train > t

    )

    s_val = metrics.f1_score(

        y_train.values.astype(np.float), 

        yp_val > t

    )

    s_test = metrics.f1_score(

        y_train.values.astype(np.float), 

        yp_test > t

    )

    

    scores_train.append(s_train)

    scores_val.append(s_val)

    scores_test.append(s_test)

    

plt.plot(thresholds, scores_train)

plt.plot(thresholds, scores_val)

plt.plot(thresholds, scores_test)

plt.axvline(thresholds[np.argmax(scores_val)], ls='--')

plt.xlabel('Threshold')

plt.ylabel('F1 Score')

plt.show()



print(round(np.max(scores_val), 4), thresholds[np.argmax(scores_val)])

best_thresh = thresholds[np.argmax(scores_val)]
best_thresh = 0.4
pred_problem = yp_test > best_thresh

pred_neg     = yp_test <= best_thresh



true_problem = y_train > 0.5

true_neg     = y_train <= 0.5
(true_problem & pred_problem).sum()
(pred_neg & true_neg).sum()
(pred_neg & true_problem).sum()
(pred_problem & true_neg).sum()
importances = pd.DataFrame()



for fold_ in range(len(models)):

    

    model = models[fold_]

    

    imp_df = pd.DataFrame()

    imp_df['feature'] = X_train.columns

    imp_df['gain'] = model.feature_importance('gain')

    imp_df['fold'] = fold_ + 1

    importances = pd.concat([importances, imp_df], axis=0, sort=False)

importances.groupby('feature').gain.mean().sort_values(ascending=True).plot(kind='barh');
sns.set_context("paper", font_scale=2)



important_features = importances[['gain', 'feature']].groupby('feature').mean().sort_values('gain').index.values[::-1]



for f in important_features:

    print(f)

    fig, ax = plt.subplots(1, 1, figsize=(8,6))

    sns.regplot(

        f,

        'target',

        pd.merge(X_train, y_train.to_frame(), on='id_measurement', how='left'),

        logistic=True,

        n_boot=100,

        y_jitter=.1,

        scatter_kws={'alpha':0.1, 'edgecolor':'none'},

        ax=ax

    )

    plt.show()
del X_train, train_peaks

gc.collect()
NUM_TEST_CHUNKS = 10



test_chunk_size = int(np.ceil((meta_test_df.shape[0]/3.)/float(NUM_TEST_CHUNKS))*3.)



test_peaks = []



for j in range(NUM_TEST_CHUNKS):



    j_start = j*test_chunk_size

    j_end = (j+1)*test_chunk_size



    signal_ids = meta_test_df['signal_id'].values[j_start:j_end]



    test_df = pd.read_parquet(

        data_dir + '/test.parquet',

        columns=[str(c) for c in signal_ids]

    )



    p = process_measurement(

        test_df, 

        meta_test_df.iloc[j_start:j_end], 

    )



    test_peaks.append(p)



    print(j)



    del test_df

    gc.collect()





test_peaks = pd.concat(test_peaks)
test_peaks = pd.merge(test_peaks, meta_test_df[['signal_id', 'id_measurement']], on='signal_id', how='left')

test_peaks.head()
X_test = calculate_features(test_peaks, meta_test_df)
X_test.head()
yp_test = np.zeros(X_test.shape[0])



for j in range(len(models)):

    model = models[j]

    yp_test += model.predict(X_test.values)/len(models)
test_submission = pd.DataFrame(

    yp_test,

    index=X_test.index,

    columns=['probability']

)

test_submission['target'] = (yp_test > best_thresh).astype(np.int)



test_submission = pd.merge(

    meta_test_df[['id_measurement', 'signal_id']],

    test_submission,

    on='id_measurement',

    how='left'

)
test_submission.head()
test_submission.target.value_counts()
submission = test_submission[['signal_id', 'target']]

submission.to_csv('submission.csv', index=False)