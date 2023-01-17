import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm



print(os.listdir("../input"))
df_train = pd.read_csv("../input/X_train.csv")

df_train.head()
df_test = pd.read_csv("../input/X_test.csv")

df_test.head()
df_train["series_id"].is_monotonic
df_train.groupby("series_id").count().head(20)
df_y = pd.read_csv("../input/y_train.csv")

df_y.head()
df_y.nunique()
df_y["surface"].value_counts().plot(kind="barh", figsize=(14,8), fontsize=12)
from sklearn.preprocessing import LabelEncoder



# encode class values as integers so they work as targets for the prediction algorithm

encoder = LabelEncoder()

y = encoder.fit_transform(df_y["surface"])

y_count = len(list(encoder.classes_))
label_mapping = {i: l for i, l in enumerate(encoder.classes_)}
y.shape
df_train.shape
3810*128 #this should match the two values above
df_train["target"] = y.repeat(128)
#a look at just a few columns to see target is there

df_train[["series_id", "orientation_X", "orientation_Y", "orientation_Z", "target"]].head()
def plot_robot_series(series_id):

    robot_series_data = df_train[df_train["series_id"] == series_id]

    orientation_data = robot_series_data[["orientation_X", "orientation_Y", "orientation_Z"]]

    angular_data = robot_series_data[["angular_velocity_X", "angular_velocity_Y", "angular_velocity_Z"]]

    linear_data = robot_series_data[["linear_acceleration_X", "linear_acceleration_Y", "linear_acceleration_Z"]]

    surface = robot_series_data["target"].iloc[0]

    surface = label_mapping[surface]



    fig, axs = plt.subplots(figsize=(15,3), nrows=1, ncols=3)

    axs[0].plot(orientation_data)

    axs[0].set_title(surface+": orientation XYZ")

    axs[0].legend(("X", "Y", "Z"), loc="upper left")

    axs[1].plot(angular_data)

    axs[1].set_title(surface+": angular velocity")

    axs[1].legend(("X", "Y", "Z"), loc="upper left")

    axs[2].plot(linear_data)

    axs[2].set_title(surface+": linear acceleration")

    axs[2].legend(("X", "Y", "Z"), loc="upper left")

    plt.show()
for key in label_mapping:

    rows = df_train[df_train["target"] == key]

    #find the first row with this surface type

    row = df_train.index.get_loc(rows.iloc[0].name)

    sid = df_train.iloc[row]["series_id"]

    plot_robot_series(sid)

    #print(row)
grouped = df_train.groupby("target")["linear_acceleration_Y"]

rowlength = int(grouped.ngroups/3)

fig, axs = plt.subplots(figsize=(15,15), 

                        nrows=3, ncols=rowlength)



targets = zip(grouped.groups.keys(), axs.flatten())

for i, (key, ax) in enumerate(targets):

    ax.plot(grouped.get_group(key))

    ax.set_title('a='+label_mapping[key])

ax.legend()

plt.show()
from tqdm import tqdm



def process_outliers(df, outlier_cols):

    def fz_raw(x):

        #TODO: negative outliers?

        zscore = (x - x.mean())/x.std()

        z_upper = x.mean()+3*x.std()

        z_upper = np.repeat(z_upper, 128)

        z_lower = x.mean()-3*x.std()

        z_lower = np.repeat(z_lower, 128)



        return np.array((zscore, z_upper, z_lower))



    cols_to_drop = []

    for col in tqdm(outlier_cols):

        a = df[col].values

        

        series_groups = df.groupby("series_id")

        xyz = series_groups[col].apply(fz_raw)

        xyz_c = np.concatenate(xyz, axis=1)

        

        z = xyz_c[0]

        mask = z >= 3

#        print(sum(mask))

        a[mask] = xyz_c[1][mask]

        mask = z <= -3

#        print(sum(mask))

        a[mask] = xyz_c[2][mask]
outlier_cols = [col_name for col_name in df_train.columns if col_name not in ["row_id", "series_id", "measurement_number"]]

outlier_cols.remove("orientation_X")

outlier_cols.remove("orientation_Y")

outlier_cols.remove("orientation_Z")

outlier_cols.remove("orientation_W")

outlier_cols.remove("target")

process_outliers(df_train, outlier_cols)
process_outliers(df_test, outlier_cols)
grouped = df_train.groupby("target")["linear_acceleration_Y"]

rowlength = int(grouped.ngroups/3)   # fix up if odd number of groups

fig, axs = plt.subplots(figsize=(12,12), 

                        nrows=3, ncols=rowlength)

#                        gridspec_kw=dict(hspace=0.4)) # Much control of gridspec



targets = zip(grouped.groups.keys(), axs.flatten())

for i, (key, ax) in enumerate(targets):

    ax.plot(grouped.get_group(key))

    ax.set_title('a='+label_mapping[key])

ax.legend()

plt.show()
# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    df = actual

    xyz = np.vectorize(quaternion_to_euler)(df['orientation_X'], df['orientation_Y'], df['orientation_Z'], df['orientation_W'])

    actual['euler_x'] = xyz[0]

    actual['euler_y'] = xyz[1]

    actual['euler_z'] = xyz[2]

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    

    series_groups = df.groupby("series_id")

    

    def f1(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    def f2(x):

        return np.mean(np.abs(np.diff(x)))

    

    def fx_raw(x):

        diff = np.diff(x)

        diff = np.concatenate([[np.nan], diff])

        abs_diff = np.abs(diff)

        abs_diff_diff = np.abs(np.diff(abs_diff))

        abs_diff_diff = np.concatenate([[np.nan], abs_diff_diff])

        

        raw_array = np.array((abs_diff, abs_diff_diff, diff))

        return raw_array

    

    def fx_sum(x):

        abs_max = np.max(np.abs(x))

        abs_min = np.min(np.abs(x))

        abs_diff = np.abs(np.diff(x))

        abs_diff_avg = np.mean(abs_diff)

        abs_diff_diff = np.abs(np.diff(abs_diff))

        abs_diff_diff_avg = np.mean(abs_diff_diff)

        sum_diff = np.sum(np.diff(x))

        sum_abs_diff = np.sum(abs_diff)

        count_diffs = np.count_nonzero(x)

        

        sum_array = np.array((abs_max, abs_min, abs_diff_avg, abs_diff_diff_avg, 

                              sum_diff, sum_abs_diff, count_diffs))

        return sum_array

    

    for col in tqdm(actual.columns):

        if col in ['row_id', 'series_id', 'measurement_number', 'target']:

            continue



        new[col + '_mean'] = series_groups[col].mean()

        new[col + '_min'] = series_groups[col].min()

        new[col + '_max'] = series_groups[col].max()

        new[col + '_std'] = series_groups[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']



        xyz = series_groups[col].apply(fx_sum)

        rows = len(series_groups)

        col_count = len(xyz[0]) # the count returned by fx_sum(x)

        xyz_sum = np.concatenate(xyz).reshape(rows, col_count)

        column_names = [col+"_abs_max", col+"_abs_min", 

                        col+"_abs_diff_avg", col+"_abs_diff_diff_avg",

                        col+"_sum_diff", col+"_sum_abs_diff",

                        col+"_count_diff"

                       ]

        df_xyz_sum = pd.DataFrame(xyz_sum, columns=column_names)

        new = pd.concat([new, df_xyz_sum], axis=1)

        

        xyz = series_groups[col].apply(fx_raw)

        xyz_c = np.concatenate(xyz, axis=1)

        df_xyz_raw = pd.DataFrame({col+"_abs_diff": xyz_c[0],

                                   col+"_abs_diff_diff": xyz_c[1],

                                   col+"_diff": xyz_c[2]

                                   })

        #this could become an infinite loop as this loop is over columns in "actual" and this adds more columns to it

        #but this does not happen, so i guess the column set is only read at start of loop

        actual = pd.concat([actual, df_xyz_raw], axis=1)



    return new, actual
%%time

df_train_sum, df_train = fe(df_train)
%%time

df_test_sum, df_test = fe(df_test)
df_train_sum.describe()
max_x_id = df_train_sum["orientation_X_sum_diff"].idxmax()

max_x_id
plot_robot_series(1904)
import matplotlib.pyplot as plt



matfig = plt.figure(figsize=(12,12))

plt.matshow(df_train.corr(), fignum=matfig.number)
#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas



corr_matrix = df_train.corr().abs()



#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

top_df = pd.DataFrame(sol).reset_index()
top_df.columns = ["var1", "var2", "corr"]

top_df.head(10)
cor = df_train.corr()

#just print one to see it works

cor["orientation_X"]["orientation_W"]
for x in range(len(top_df)):

    var1 = top_df.iloc[x]["var1"]

    var2 = top_df.iloc[x]["var2"]

    corr = cor[var1][var2]

    top_df.at[x, "corr2"] = corr

top_df.head(20)
#https://stackoverflow.com/questions/21137150/format-suppress-scientific-notation-from-python-pandas-aggregation-results

pd.set_option('display.float_format', lambda x: '%.5f' % x)

top_df[top_df["var2"] == "target"].sort_values(by="corr", ascending=False)
df_train.groupby("target").mean()
mean_group = df_train.groupby("target").mean()



def scatterplot_variable(var_x, var_y):

    ax = mean_group.plot.scatter(x=var_x, y=var_y, figsize=(8, 5))

    for i in mean_group.index:

        label = label_mapping[i]

        row = mean_group.loc[i]

        ax.annotate(label, (row[var_x], row[var_y]))
scatterplot_variable("orientation_X", "orientation_Y")
scatterplot_variable("angular_velocity_X", "angular_velocity_Y")
scatterplot_variable("linear_acceleration_X", "linear_acceleration_Y")
scatterplot_variable("total_angular_velocity", "total_linear_acceleration")
scatterplot_variable("euler_x", "euler_y")
def plot_average_diff(col_name):

    labels = []

    rows = []

    for x in range(9):

        label = label_mapping[x]

        labels.append(label)

        ser = df_train[df_train["target"] == x]

        #it is called "ox" because I started with orientation_X.

        ox = ser[col_name+"_diff"].values.reshape(-1, 128)

        # this was something I just used to get a bigger scale

        #ox *= 100

        omx = ox.mean(0)

        rows.append(omx)

    df = pd.DataFrame(rows)

    df.index = labels

    df.T.plot(figsize=(14,8))
plot_average_diff("orientation_X")
plot_average_diff("orientation_Y")
plot_average_diff("orientation_Z")
plot_average_diff("angular_velocity_X")
plot_average_diff("angular_velocity_Y")
plot_average_diff("angular_velocity_Z")
feature_cols = list(df_train.columns)

feature_cols.remove('row_id')

feature_cols.remove('series_id')

feature_cols.remove('measurement_number')

feature_cols.remove('target')

#feature_cols
df_train.to_csv("features_train_raw.csv")

df_test.to_csv("features_test_raw.csv")

df_train_sum.to_csv("features_train_sum.csv")

df_test_sum.to_csv("features_test_sum.csv")
#function to scale two dataframes. fit and transform the first (e.g., training set), 

#and use the same scaler to transform the seconds one (e.g., test set)

def scale_df(df1, df2, feature_cols):

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(df1[feature_cols])

    #df_X_train = pd.DataFrame(scaled_features, index=df_X_train.index, columns=df_X_train.columns)

    df1[feature_cols] = scaled_features

    scaled_test_features = scaler.transform(df2[feature_cols])

    df2[feature_cols] = scaled_test_features

    return df1, df2
df_train, df_test = scale_df(df_train, df_test, feature_cols)
df_train.describe()
#appears we stripped the columns in this summary dataframe already, so can just use those columns as is

df_train_sum, df_test_sum = scale_df(df_train_sum, df_test_sum, df_train_sum.columns)
df_train.to_csv("features_train_scaled_raw.csv")

df_test.to_csv("features_test_scaled_raw.csv")

df_train_sum.to_csv("features_train_scaled_sum.csv")

df_test_sum.to_csv("features_test_scaled_sum.csv")
df_train.head()
df_features = df_train[feature_cols]
df_train_sum.shape
from sklearn.metrics import accuracy_score, confusion_matrix

import itertools

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

import collections



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

sub_preds_rf = np.zeros((df_test_sum.shape[0], 9))

oof_preds_rf = np.zeros((df_train_sum.shape[0]))

score = 0

misclassified_indices = []

misclassified_tuples_all = []

for i, (train_index, test_index) in enumerate(folds.split(df_train_sum, y)):

    print('-'*20, i, '-'*20)

    

    clf =  RandomForestClassifier(n_estimators = 200, n_jobs = -1)

    clf.fit(df_train_sum.iloc[train_index], y[train_index])

    oof_preds_rf[test_index] = clf.predict(df_train_sum.iloc[test_index])

    sub_preds_rf += clf.predict_proba(df_test_sum) / folds.n_splits

    score += clf.score(df_train_sum.iloc[test_index], y[test_index])

    print('score ', clf.score(df_train_sum.iloc[test_index], y[test_index]))

    importances = clf.feature_importances_

    features = df_train_sum.columns



    feat_importances = pd.Series(importances, index=features)

    feat_importances.nlargest(30).sort_values().plot(kind='barh', color='#86bf91', figsize=(10,8))

    plt.show()

    

    missed = y[test_index] != oof_preds_rf[test_index]

    misclassified_indices.append(test_index[missed])

    misclassified_samples1 = y[test_index][missed]

    misclassified_samples2 = oof_preds_rf[test_index][missed].astype("int")

    m1 = encoder.inverse_transform(misclassified_samples1)

    m2 = encoder.inverse_transform(misclassified_samples2)

    misclassified_tuples = [(a, b) for a, b in zip(m1, m2)]

    misclassified_tuples_all.append(misclassified_tuples)



print('Avg Accuracy', score / folds.n_splits)

ss = pd.read_csv('../input/sample_submission.csv')

ss['surface'] = encoder.inverse_transform(sub_preds_rf.argmax(axis=1))

ss.to_csv('rf.csv', index=False)

ss.head(10)
# https://www.kaggle.com/artgor/where-do-the-robots-drive



def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(y, oof_preds_rf, encoder.classes_)
#misclassified_indices
#misclassified_tuples_all
df_train_sum.iloc[misclassified_indices[0]].head()
df_missed_all = None

total_counter = collections.Counter()



def counter_to_df(df_missed_all, counter, new_col_name):

    sorted_freqs = counter.most_common()

    #single item in sorted_freqs is like this: ('fine_concrete', 'wood'), 7)

    rows = [[row[0][0], row[0][1], row[1]] for row in sorted_freqs]

    df_missed = pd.DataFrame(rows, columns=["expected", "actual", "count"])

    if df_missed_all is None:

        df_missed_all = df_missed

    else:

        df_missed_all = df_missed_all.merge(df_missed, how='left', on=["expected", "actual"])

    created_col = df_missed_all.columns[-1]

    print("renaming: "+new_col_name)

    df_missed_all.rename(columns={created_col: new_col_name}, inplace=True)

    df_missed_all[new_col_name].fillna(0, inplace=True)

    df_missed_all[new_col_name] = df_missed_all[new_col_name].astype(int)

    return df_missed_all



i = 1

for misclassified_tuples in misclassified_tuples_all:

    counter = collections.Counter(misclassified_tuples)

    total_counter.update(misclassified_tuples)

    df_missed_all = counter_to_df(df_missed_all, counter, "count"+str(i))

    i += 1



df_missed_all = counter_to_df(df_missed_all, total_counter, "all")

#df_missed_all
new_order = [0, 1, 7, 2, 3, 4, 5, 6]

columns = [df_missed_all.columns[i] for i in new_order]

df_missed_all = df_missed_all[columns]
df_missed_all.sort_values(by="all", ascending=False)
y
miss_map = {}

hit_map = {}

ids, counts = np.unique(y, return_counts=True)

totals = dict(zip(ids, counts))

total_misses = 0

total_hits = 0



for label_id in label_mapping.keys():

    label_name = label_mapping[label_id]

    misses = df_missed_all[df_missed_all["expected"] == label_name]["all"].sum()

    total_misses += misses

    miss_map[label_name] = misses

    total = totals[label_id]

    hits = total - misses

    total_hits += hits

    hit_map[label_name] = hits

print(miss_map)

print(hit_map)

print(total_misses)

print(total_hits)

print(total_misses+total_hits)
df_hit_miss = pd.DataFrame.from_dict(hit_map, orient="index")

df_miss = pd.DataFrame.from_dict(miss_map, orient="index")

df_hit_miss = df_hit_miss.join(df_miss, how='outer', lsuffix='_left', rsuffix='_right')

df_hit_miss.columns = ["hits", "misses"]

df_hit_miss.sort_values(by="hits", ascending=False)
idx_counter = collections.Counter()

for missed_idx in misclassified_indices:

    idx_counter.update(missed_idx)
len(idx_counter.most_common())