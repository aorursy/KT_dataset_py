import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv("../input/winequalityN.csv")
df.head()
white_df = df[(df['type'] == 'white')]
white_df.describe()
def remove_garbage(df):
    df.dropna(inplace=True)
    df = df.loc[:, [i for i in df.columns]]
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

white_df = white_df.drop(['type'], axis=1)
white_df = remove_garbage(pd.DataFrame(data = white_df, columns = list(white_df.columns.values)))
white_df_x = white_df.iloc[:,[i for i in range(len(white_df.columns) - 1)]]
white_df_x.describe()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.pairplot(white_df.iloc[:,[i for i in range(len(white_df_x.columns))]], size = 2.5)
plt.show();
from scipy import stats
def is_normal(x, treshhold = 0.05):
    k2,p = stats.normaltest(x)
    print(p)
    print(p > treshhold)
    print('\n')
    return p > treshhold

for name in list(white_df_x):
    is_normal(np.array(white_df_x[name]))
white_df_x['alcohol'].map(type).unique()
white_df_x.alcohol = white_df_x.alcohol.apply(float)
from sklearn import preprocessing
standrd_scaler = preprocessing.StandardScaler()
np_scaled = standrd_scaler.fit_transform(white_df_x)
scaled_white_df = pd.DataFrame(np_scaled, columns = [name for name in list(white_df_x)])
scaled_white_df.head()
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (20,10))
scaled_white_df.boxplot(column=[name for name in list(scaled_white_df)], grid=False)
def fit_and_plot(name, dist, data):
    upper_bound = len(data)
    params = dist.fit(data) #return (mean, std) tuple
    arg = params[:-2] #The skewness reduces as the value of alpha increases. (for gamma distribution)
    #gamma is class of continue distributions
    loc = params[0]
    scale = params[1]
    x = np.linspace(min(data), max(data))
    _, ax = plt.subplots(figsize=(30, 10))
    plt.scatter(data, np.linspace(1, 10, upper_bound))
    ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
    plt.plot(x, dist.pdf(x, loc=0, scale=1), '-', color = "r")
    plt.show()
    print(name)
    print('mean = ' + str(loc), 'std = ' + str(scale))
    print('\n')
    return dist, loc, scale
for name in list(scaled_white_df):
    fit_and_plot(name, stats.norm, scaled_white_df[name])
min_max_scaler = preprocessing.MinMaxScaler()
np_min_max_scaled = min_max_scaler.fit_transform(white_df_x)
min_max_scaled_white_df = pd.DataFrame(np_min_max_scaled, columns = [name for name in list(white_df_x)])
fig = plt.figure(figsize = (20,10))
min_max_scaled_white_df.boxplot(column=[name for name in list(min_max_scaled_white_df)], grid=False)
def outliers_detection(data):
    data = np.array(data)
    percentile_25 = np.percentile(data, 25)
    percentile_50 = np.percentile(data, 50)
    percentile_75 = np.percentile(data, 75)
    lower_bound = percentile_25 - 1.5 * (percentile_75 - percentile_25)
    upper_bound = percentile_75 + 1.5 * (percentile_75 - percentile_25)
    outliers = []
    for point in list(data):
        if point < lower_bound or point > upper_bound:
            outliers.append(point)
        else:
            outliers.append('not a outlier')
    
    return outliers
d_outliers_focused = {}
for name in list(white_df_x):
    d_outliers_focused.setdefault(name, outliers_detection(white_df_x[name]))
white_df_outliers_focused = pd.DataFrame(data=d_outliers_focused)
white_df_outliers_focused.head()
series_list = []
for index, row in white_df_outliers_focused.iterrows():
    for name in list(white_df_outliers_focused):
        if type(row[name]) == np.float64:
            series_list.append(row)
            break
            
white_df_outliers = pd.DataFrame(series_list, columns=list(white_df_outliers_focused))
white_df_outliers.describe()
white_df_outliers.head(50)
outliers_quality = pd.concat((white_df_outliers, white_df['quality']), axis=1).dropna()
quality = list(outliers_quality['quality'])
max(set(quality), key=quality.count)
outliers_indices = white_df_outliers.index.tolist()
print(outliers_indices)
white_df_x.drop(white_df_x.index[outliers_indices], inplace=True)
white_df_x.describe()
for name in list(white_df_x):
    is_normal(np.array(white_df_x[name]))
np_scaled = standrd_scaler.fit_transform(white_df_x)
scaled_white_df = pd.DataFrame(np_scaled, columns = [name for name in list(white_df_x)])
scaled_white_df.head()
fit_and_plot('chlorides', stats.norm, scaled_white_df['chlorides'])