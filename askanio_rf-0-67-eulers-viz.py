import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
X_train = pd.read_csv('../input/X_train.csv')

Y_train = pd.read_csv('../input/y_train.csv')

X_pred = pd.read_csv('../input/X_test.csv')
X_train.head()
Y_train.head()
X_pred.head()
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
def fe_step0 (actual):

    

    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual
X_train = fe_step0(X_train)

X_pred = fe_step0(X_pred)

print(X_train.shape)

X_train.head()
def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual
X_train = fe_step1(X_train)

X_pred = fe_step1(X_pred)

print(X_train.shape)

X_train.head()
def mean_abs_change(x):

    return np.mean(np.abs(np.diff(x)))
def get_features(df):

    result_df = pd.DataFrame()

    for col in df.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        result_df['{}_mean'.format(col)] = df.groupby(['series_id'])[col].mean()

        result_df['{}_max'.format(col)] = df.groupby(['series_id'])[col].max()

        result_df['{}_min'.format(col)] = df.groupby(['series_id'])[col].min()

        result_df['{}_sum'.format(col)] = df.groupby(['series_id'])[col].sum()

        result_df['{}_mean_abs_change'.format(col)] = df.groupby(['series_id'])[col].apply(mean_abs_change)

    return result_df
train_df = get_features(X_train)

pred_df = get_features(X_pred)
train_df = train_df.merge(Y_train, on='series_id', how='inner')
train_df.head()
surfaces = (train_df['surface'].value_counts()).index
features = []

for feature in train_df.columns.values[1:-2]:

    if 'euler' in feature:

        features.append(feature)
features
def plot_features_and_surfaces(features, surfaces, train_df):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(5,3,figsize=(16,24))



    for feature in features:

        i += 1

        plt.subplot(5,3,i)

        for surface in surfaces:

            ttc = train_df[train_df['surface']==surface]

            sns.kdeplot(ttc[feature], bw=0.5,label=surface)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
plot_features_and_surfaces(features, surfaces, train_df)
label_encoder = LabelEncoder()

train_df['surface'] = label_encoder.fit_transform(train_df['surface'])
train_df.drop(['series_id', 'group_id'], axis=1, inplace=True)
X = train_df.drop('surface', axis=1)

y = train_df['surface']
rf = RandomForestClassifier(random_state=42, n_estimators=100)
def evaluate_model(model, X, y):

    score = cross_val_score(rf, X, y, cv=6)

    print(score)

    print(score.mean())
evaluate_model(rf, X, y)
def predict_and_submit(model, X, y, x_pred, label_encoder, file_name):

    model.fit(X, y)

    predictions = label_encoder.inverse_transform(model.predict(x_pred))

    submission = pd.read_csv('../input/sample_submission.csv')

    submission['surface'] = predictions

    submission.to_csv(file_name, index=False)
predict_and_submit(rf, X, y, pred_df, label_encoder, 'submitWithEulers01.csv')