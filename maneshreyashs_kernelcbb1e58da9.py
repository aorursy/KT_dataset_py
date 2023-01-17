import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
# pandas doesn't show us all the decimals

pd.options.display.precision = 15

# much better!

train.head()
from sklearn.linear_model import LinearRegression

def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



from scipy import stats
rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01',

                               'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend', 'iqr', 

                                'q999','q001','ave10'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()

    X_train.loc[segment, 'q95'] = np.quantile(x,0.95)

    X_train.loc[segment, 'q99'] = np.quantile(x,0.99)

    X_train.loc[segment, 'q05'] = np.quantile(x,0.05)

    X_train.loc[segment, 'q01'] = np.quantile(x,0.01)

    

    X_train.loc[segment, 'abs_max'] = np.abs(x).max()

    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()

    X_train.loc[segment, 'abs_std'] = np.abs(x).std()

    X_train.loc[segment, 'trend'] = add_trend_feature(x)

    X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    

    X_train.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_train.loc[segment, 'q999'] = np.quantile(x,0.999)

    X_train.loc[segment, 'q001'] = np.quantile(x,0.001)

    X_train.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)
X_train.head()




scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

parameters = [{ 'gamma': [0.6, 0.7, 0.8],

               'C': [2.35, 2.4, 2.45, 2.5],

              'nu': [0.85, 0.9, 0.95]}]



reg1 = GridSearchCV(NuSVR(kernel='rbf', tol=0.01), parameters, cv = 3, scoring=scorer)

reg1.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = reg1.predict(X_train_scaled)



print(reg1.best_params_)

print(reg1.best_score_)
parameters = [{ 'gamma': [0.06, 0.1, 0.08, 0.09], #np.logspace(-2, 2, 5)

               'alpha': [0.005, 0.01, 0.05]}]



reg2 = GridSearchCV(KernelRidge(kernel='rbf'), parameters, cv = 3, scoring=scorer)

reg2.fit(X_train_scaled, y_train.values.flatten())

y_pred2 = reg2.predict(X_train_scaled)



print(reg2.best_params_)

print(reg2.best_score_)
plt.tight_layout()

f = plt.figure(figsize=(12, 6))

f.add_subplot(1,2, 1)

plt.scatter(y_train.values.flatten(), y_pred1)

plt.title('reg1', fontsize=20)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

f.add_subplot(1,2, 2)

plt.scatter(y_train.values.flatten(), y_pred2)

plt.title('reg2', fontsize=20)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show(block=True)
score1 = mean_absolute_error(y_train.values.flatten(), y_pred1)

print(f'Score1: {score1:0.3f}')

score2 = mean_absolute_error(y_train.values.flatten(), y_pred2)

print(f'Score2: {score2:0.3f}')

score3 = mean_absolute_error(y_train.values.flatten(), y_pred1*0.5+y_pred2*0.5)

print(f'Score3: {score3:0.3f}')
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)

    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)

    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)

    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)

    

    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()

    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()

    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()

    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)

    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    

    X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)

    X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)

    X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = reg1.predict(X_test_scaled)*0.5 + reg2.predict(X_test_scaled)*0.5

submission.to_csv('submission.csv')

print(os.getcwd())