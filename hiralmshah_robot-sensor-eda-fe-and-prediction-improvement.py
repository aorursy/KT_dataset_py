# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style 

style.use('ggplot')

import warnings

warnings.filterwarnings('ignore')



import plotly.offline as py 

from plotly.offline import init_notebook_mode, iplot

py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version

import plotly.graph_objs as go # it's like "plt" of matplot



from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix

import gc





# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/X_train.csv')

X_train.head(3)
y_train = pd.read_csv('../input/y_train.csv')

y_train.head(3)
X_test = pd.read_csv('../input/X_test.csv')

X_test.head(3)
print('Size of Train Data')

print('Number of samples are: {0}\nNumber of features are: {1}'.format(X_train.shape[0], X_train.shape[1]))



print('\nSize of Test Data')

print('Number of samples are: {0}\nNumber of features are: {1}'.format(X_test.shape[0], X_test.shape[1]))



print('\nSize of Target Data')

print('Number of samples are: {0}\nNumber of features are: {1}'.format(y_train.shape[0], y_train.shape[1]))
X_train.describe()
target = y_train['surface'].value_counts().reset_index().rename(columns = {'index' : 'target'})

target
#sns.countplot(y='surface',data = y_train)

trace0 = go.Bar(

    x = y_train['surface'].value_counts().index,

    y = y_train['surface'].value_counts().values

    )



trace1 = go.Pie(

    labels = y_train['surface'].value_counts().index,

    values = y_train['surface'].value_counts().values,

    domain = {'x':[0.55,1]})



data = [trace0, trace1]

layout = go.Layout(

    title = 'Frequency Distribution for surface/target data',

    xaxis = dict(domain = [0,.50]))



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)

X_train.isnull().sum()
X_train['is_duplicate'] = X_train.duplicated()

X_train['is_duplicate'].value_counts()
X_train = X_train.drop(['is_duplicate'], axis = 1)
X_train_sort = X_train.sort_values(by = ['series_id', 'measurement_number'], ascending = True)

X_train_sort.head()
corr = X_train.corr()

corr
fig, ax = plt.subplots(1,1, figsize = (15,6))



hm = sns.heatmap(X_train.iloc[:,3:].corr(),

                ax = ax,

                cmap = 'coolwarm',

                annot = True,

                fmt = '.2f',

                linewidths = 0.05)

fig.subplots_adjust(top=0.93)

fig.suptitle('Orientation, Angular_velocity and Linear_accelaration Correlation Heatmap for Train dataset', 

              fontsize=14, 

              fontweight='bold')
fig, ax = plt.subplots(1,1, figsize = (15,6))



hm = sns.heatmap(X_test.iloc[:,3:].corr(),

                ax = ax,

                cmap = 'coolwarm',

                annot = True,

                fmt = '.2f',

                linewidths = 0.05)

fig.subplots_adjust(top=0.93)

fig.suptitle('Orientation, Angular_velocity and Linear_accelaration Correlation Heatmap for Test dataset', 

              fontsize=14, 

              fontweight='bold')
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(311)

ax.set_title('Distribution of Orientation_X,Y,Z,W',

             fontsize=14, 

             fontweight='bold')

X_train.iloc[:,3:7].boxplot()

ax = fig.add_subplot(312)

ax.set_title('Distribution of Angular_Velocity_X,Y,Z',fontsize=14, 

             fontweight='bold')

X_train.iloc[:,7:10].boxplot()

ax = fig.add_subplot(313)

ax.set_title('Distribution of linear_accelaration_X,Y,Z',fontsize=14, 

             fontweight='bold')

X_train.iloc[:,10:13].boxplot()
plt.figure(figsize=(26, 16))

for i, col in enumerate(X_train.columns[3:]):

    ax = plt.subplot(3, 4, i + 1)

    sns.distplot(X_train[col], bins=100, label='train')

    sns.distplot(X_test[col], bins=100, label='test')

    ax.legend()   
df = X_train.merge(y_train, on = 'series_id', how = 'inner')

targets = (y_train['surface'].value_counts()).index
df.head(3)
plt.figure(figsize=(26, 16))

for i,col in enumerate(df.columns[3:13]):

    ax = plt.subplot(3,4,i+1)

    ax = plt.title(col)

    for surface in targets:

        surface_feature = df[df['surface'] == surface]

        sns.kdeplot(surface_feature[col], label = surface)
series_dict = {}

for series in (X_train['series_id'].unique()):

    series_dict[series] = X_train[X_train['series_id'] == series] 
# From: Code Snippet For Visualizing Series Id by @shaz13

def plotSeries(series_id):

    style.use('ggplot')

    plt.figure(figsize=(28, 16))

    print(y_train[y_train['series_id'] == series_id]['surface'].values[0].title())

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        if col.startswith("o"):

            color = 'red'

        elif col.startswith("a"):

            color = 'green'

        else:

            color = 'blue'

        if i >= 7:

            i+=1

        plt.subplot(3, 4, i + 1)

        plt.plot(series_dict[series_id][col], color=color, linewidth=3)

        plt.title(col)
plotSeries(1)
# from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

def filter_signal(signal, threshold=1e3):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)
# denoise train and test angular_velocity and linear_acceleration data

X_train_denoised = X_train.copy()

X_test_denoised = X_test.copy()
X_train.head(3)
from numpy.fft import *



# train

for col in X_train.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = X_train.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_train

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        X_train_denoised[col] = list_denoised_data

        

# test

for col in X_test.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = X_test.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_train

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        X_test_denoised[col] = list_denoised_data

        
series_dict = {}

for series in (X_train_denoised['series_id'].unique()):

    series_dict[series] = X_train_denoised[X_train_denoised['series_id'] == series] 
plotSeries(1)
plt.figure(figsize=(24, 8))

plt.title('linear_acceleration_X')

plt.plot(X_train.angular_velocity_Z[128:256], label="original");

plt.plot(X_train_denoised.angular_velocity_Z[128:256], label="denoised");

plt.legend()

plt.show()
#https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

#quaternion to eular

def quaternion_to_euler(qx,qy,qz,qw):

    import math

    # roll (x-axis rotation)

    sinr_cosp = +2.0 * (qw * qx + qy + qz)

    cosr_cosp = +1.0 - 2.0 * (qx * qx + qy * qy)

    roll = math.atan2(sinr_cosp, cosr_cosp)

    

    # pitch (y-axis rotation)

    sinp = +2.0 * (qw * qy - qz * qx)

    if(math.fabs(sinp) >= 1):

        pitch = copysign(M_PI/2, sinp)

    else:

        pitch = math.asin(sinp)

        

    # yaw (z-axis rotation)

    siny_cosp = +2.0 * (qw * qz + qx * qy)

    cosy_cosp = +1.0 - 2.0 * (qy * qy + qz * qz)

    yaw = math.atan2(siny_cosp, cosy_cosp)

    

    return roll, pitch, yaw
def eular_angle(data):

    x, y, z, w = data['orientation_X'].tolist(), data['orientation_Y'].tolist(), data['orientation_Z'].tolist(), data['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    data['euler_x'] = nx

    data['euler_y'] = ny

    data['euler_z'] = nz

    

    return data
data = eular_angle(X_train_denoised)

test = eular_angle(X_test_denoised)

print(data.shape, test.shape)
data.head(3)
def fe_eng1(data):

    data['total_angular_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['total_linear_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['total_orientation'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['total_linear_acc'] / data['total_angular_vel']

    data['total_angle'] = (data['euler_x'] ** 2 + data['euler_y'] ** 2 + data['euler_z'] ** 2) ** 5

    data['angle_vs_acc'] = data['total_angle'] / data['total_linear_acc']

    data['angle_vs_vel'] = data['total_angle'] / data['total_angular_vel']

    return data
data = fe_eng1(data)

test = fe_eng1(test)

print(data.shape, test.shape)
def fe_eng2(data):

    df = pd.DataFrame()

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        #in statistics, the median absolute deviation (MAD) is a robust measure of the variablility of a univariate sample of quantitative data.

        df[col + '_mad'] = data.groupby(['series_id'])[col].apply(lambda x: np.median(np.abs(np.diff(x))))

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
%%time

data = fe_eng2(data)

test = fe_eng2(test)

print(data.shape, test.shape)
data.head(3)
data.fillna(0, inplace = True)

data.replace(-np.inf, 0, inplace = True)

data.replace(np.inf, 0, inplace = True)

test.fillna(0, inplace = True)

test.replace(-np.inf, 0, inplace = True)

test.replace(np.inf, 0, inplace = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
y_train.head()
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=60)

predicted = np.zeros((test.shape[0],9))

measured= np.zeros((data.shape[0]))

score = 0
for times, (trn_idx, val_idx) in enumerate(folds.split(data.values,y_train['surface'].values)):

    model = RandomForestClassifier(n_estimators=700, n_jobs = -1)

    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

    model.fit(data.iloc[trn_idx],y_train['surface'][trn_idx])

    measured[val_idx] = model.predict(data.iloc[val_idx])

    predicted += model.predict_proba(test)/folds.n_splits

    score += model.score(data.iloc[val_idx],y_train['surface'][val_idx])

    print("Fold: {} score: {}".format(times,model.score(data.iloc[val_idx],y_train['surface'][val_idx])))

    

    gc.collect()
print('Average score', score / folds.n_splits)
confusion_matrix(measured,y_train['surface'])
fig, ax = plt.subplots(1,1,figsize=(12,5))

sns.heatmap(pd.DataFrame(confusion_matrix(measured,y_train['surface'])),

            ax = ax,

            cmap = 'coolwarm',

            annot = True,

            fmt = '.2f',

            linewidths = 0.05)

fig.subplots_adjust(top=0.93)

fig.suptitle('Confusion matrix, Actual vs Predicted label Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)

indices = np.argsort(importances)[::-1]
feature_importances = pd.DataFrame(importances, index = data.columns, columns = ['importance'])

feature_importances.sort_values('importance', ascending = False)

feature_importances.head(20)
feature_importances.sort_values('importance', ascending = False).plot(kind = 'bar', 

                         figsize = (35,8), 

                         color = 'r', 

                         yerr=std[indices], 

                        align = 'center')

plt.xticks(rotation=90)

plt.show()
feature_importances.sort_values('importance', ascending = False)[:100].plot(kind = 'bar',

                                                                            figsize = (30,5),

                                                                            color = 'g', 

                                                                            yerr=std[indices[:100]], 

                                                                            align = 'center')

plt.xticks(rotation=90)

plt.show()
less_important_features = feature_importances.loc[feature_importances['importance'] < 0.0025]

print('There are {0} features their importance value is less then 0.0025'.format(less_important_features.shape[0]))
#Remove less important features from train and test set.

for i, col in enumerate(less_important_features.index):

    data = data.drop(columns = [col], axis = 1)

    test = test.drop(columns = [col], axis = 1)

    

data.shape, test.shape
predicted = np.zeros((test.shape[0],9))

measured= np.zeros((data.shape[0]))

score = 0

for times, (trn_idx, val_idx) in enumerate(folds.split(data.values,y_train['surface'].values)):

    model = RandomForestClassifier(n_estimators=700, n_jobs = -1)

    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

    model.fit(data.iloc[trn_idx],y_train['surface'][trn_idx])

    measured[val_idx] = model.predict(data.iloc[val_idx])

    predicted += model.predict_proba(test)/folds.n_splits

    score += model.score(data.iloc[val_idx],y_train['surface'][val_idx])

    print("Fold: {} score: {}".format(times,model.score(data.iloc[val_idx],y_train['surface'][val_idx])))

    

    gc.collect()
print('Average score', score / folds.n_splits)
submission = pd.read_csv('../input/sample_submission.csv')

submission['surface'] = le.inverse_transform(predicted.argmax(axis=1))

submission.to_csv('rs_surface_submission6.csv', index=False)

submission.head(10)