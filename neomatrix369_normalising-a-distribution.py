import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["figure.figsize"] = (20,3)
# data = {'score': [234, 24, 14, 27,-74,45,73,-18,59,160] }

num_of_points=20

scale = 0.1

centre_of_distribution = 0.0

data = {'score': np.random.normal(centre_of_distribution, scale, num_of_points) }
df = pd.DataFrame(data)

df
df['score'].plot(kind='bar')
df['score'].plot()
df['score'].describe()
df['score_min_removed'] = df['score'] - df['score'].min()
df[['score','score_min_removed']].plot()
df[['score','score_min_removed']].plot(kind='bar')
df['score_abs_min_removed'] = df['score'] - abs(df['score'].min())
df[['score','score_abs_min_removed']].plot()
df[['score','score_abs_min_removed']].plot(kind='bar')
df['score_max_removed'] = df['score'] - df['score'].max()
df[['score','score_max_removed']].plot()
df[['score','score_max_removed']].plot(kind='bar')
df['score_mean_removed'] = df['score'] - df['score'].mean()
df[['score','score_mean_removed']].plot()
df[['score','score_mean_removed']].plot(kind='bar')
df['score_median_removed'] = df['score'] - df['score'].median()
df[['score','score_median_removed']].plot()
df[['score','score_median_removed']].plot(kind='bar')
def trimean(values):

    return (np.quantile(values, 0.25) + (2 * np.quantile(values, 0.50)) + np.quantile(values, 0.75))/4
df['score_trimean_removed'] = df['score'] - trimean(df['score'])
df[['score','score_trimean_removed']].plot()
df[['score','score_trimean_removed']].plot(kind='bar')
df[['score','score_min_removed', 'score_abs_min_removed','score_max_removed', 'score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot()
df[['score','score_min_removed', 'score_abs_min_removed', 'score_max_removed', 'score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot(kind='bar')
df[['score','score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot()
df[['score','score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot(kind='bar')
from sklearn.preprocessing import normalize
values = normalize(np.array(df['score']).reshape(1,-1))

print(values[0])

df['score_sklearn_normalize'] = values[0]
df['score_sklearn_normalize'].plot()
def normalise_mean(data):

    return (data - data.mean()) / data.std()
df['score_normalise_mean'] = normalise_mean(df['score'])

df['score_normalise_mean']
df['score_normalise_mean'].plot()
def normalise_min_max(data):

    return (data - data.max()) / (data.max() - data.min())
df['score_normalise_min_max'] = normalise_min_max(df['score'])

df['score_normalise_min_max']
df['score_normalise_min_max'].plot()
import numpy as np
df['score_exp'] = df['score'].apply(np.exp)

df['score_exp']
df['score_exp'].plot()
df['score_log_base_e'] = df['score'].apply(np.log)

df['score_log_base_e']
df['score_log_base_e'].plot()
df['score_log_base_10'] = df['score'].apply(np.log10)

df['score_log_base_10']
df['score_log_base_10'].plot()
df.columns
columns_to_show = ['score_sklearn_normalize', 'score_normalise_mean', 'score_normalise_min_max', 

                   'score_exp', 'score_log_base_e', 'score_log_base_10']

plt.plot(df[columns_to_show])

plt.legend(columns_to_show)