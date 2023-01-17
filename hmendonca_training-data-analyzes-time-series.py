import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
import seaborn as sns
train = pd.read_csv('../input/train.csv')
print("shape of Training  dataset", train.shape)
train.describe()
train.isnull().sum().sort_values(ascending=False).sum()
plt.figure(figsize=(10,6))
xflat = train.iloc[:,2:].values.flatten() # remove id and target
xflat = pd.DataFrame(np.log1p(xflat[xflat>0])) # remove zeros
hist = np.histogram(xflat, 30)
sns.distplot(xflat, bins=hist[1], kde=False).set_title('Log histogram of Training Features (X)')
plt.xlabel('log(x)')
plt.ylabel('count')
print('Train mean: {}, std: {}'.format(xflat.values.mean(), xflat.values.std()))
plt.figure(figsize=(10,6))
target = pd.DataFrame(np.log1p(train.target))
sns.distplot(target, bins=hist[1], kde=False).set_title('Log histogram of Training Features (X)')
plt.xlabel('log target')
plt.ylabel('count')
print('Target mean: {}, std: {}'.format(target.values.mean(), target.values.std()))
# now let's check the test dataset
test = pd.read_csv('../input/test.csv')
del test['ID']
test = test.values
plt.figure(figsize=(10,6))
xflat_test = test.flatten() # remove id
xflat_test = pd.DataFrame(np.log1p(xflat_test[xflat_test>0])) # remove zeros
sns.distplot(xflat_test, bins=hist[1], kde=False).set_title('Log histogram of Test Features')
plt.xlabel('log(x)')
plt.ylabel('count')
print('Test mean: {}, std: {}'.format(xflat_test.values.mean(), xflat_test.values.std()))
print('Train mean: {}, std: {}'.format(xflat.values.mean(), xflat.values.std()))
print('Test mean: {}, std: {}'.format(xflat_test.values.mean(), xflat_test.values.std()))
print('Target mean: {}, std: {}'.format(target.values.mean(), target.values.std()))

plt.figure(figsize=(10,6))
plt.hist(xflat_test.values, alpha=.8, label='test', bins=hists[1], density=True,  histtype='bar')
plt.hist(xflat.values, alpha=.5, label='training', bins=hists[1], density=True,  histtype='bar')
plt.hist(target.values, alpha=.3, label='target', bins=hists[1], density=True,  histtype='bar')

plt.legend(prop={'size': 12})
plt.title('Normalized Log Histogram of Training Features, Test and Target');
plt.xlabel('log scale')
plt.ylabel('distribution')
plt.show()
df = np.log1p(train.drop(["ID"], axis=1))

# Drop columns with less than 20% non-zeros
df = df.loc[:, (df != 0).sum() > df.shape[0]*0.2]
df.shape
# remove crytpic column names 
cols = [str(n) for n in np.arange(-1, df.shape[1]-1)]
cols[0:1] = ['target']
df.columns = cols
df['max'] = np.log1p(train.iloc[:,2:].max(axis=1).values)    # without id and target
df['muLog'] = np.log1p(train.iloc[:,2:]).mean(axis=1).values # mean of the log
df['mu'] = np.log1p(train.iloc[:,2:].mean(axis=1).values)    # log of the mean
df.head()
def plot_corr(dframe):
    # Compute the correlation matrix
    corr = dframe.corr(method='pearson')
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
fix_cols = ['target','mu','muLog','max']
rand_cols = [str(n) for n in np.arange(0, df.shape[1]-3, 3)] # from non-zero columns

plot_corr(df[fix_cols + rand_cols])
# more pretty bad correlations
rand_cols = [str(n) for n in np.arange(0, df.shape[1]-3, 12)]
sns.pairplot(df[['target'] + rand_cols])
# somewhat better on averages?
sns.pairplot(df[fix_cols])
# assuming the columnss are actually days (or similar)
ts = train.iloc[:,2:] ## drop ID and target
ts.columns = np.arange(ts.shape[1])
quart_log = np.log1p(ts.T).rolling(365, min_periods=1, center=True, win_type='gaussian').mean(std=80).iloc[::91].T # quartal mean of the log
log_quart = np.log1p(ts.T.rolling(365, min_periods=1, center=True, win_type='gaussian').mean(std=80)).iloc[::91].T # log of the quartal mean
quart_log.columns = ['ql'+str(n) for n in quart_log.columns]
log_quart.columns = ['lq'+str(n) for n in log_quart.columns]
df = df.join(quart_log) # get target back
df = df.join(log_quart)
quart_log.head()
new_cols = ['target','ql0','ql91','ql4641','ql4732','lq0','lq91','lq4641','lq4732']
g = sns.pairplot(df[new_cols])
plot_corr(df[new_cols])
# How do they actually look like
log_quart.T.iloc[:,:5].plot(figsize=(20,6))
print(df.iloc[:5].target)
# Let's see some real data (first row [target + all columns])
g = np.log1p(train.iloc[:1, 1:].T).plot(figsize=(20,6))
# Median changes over "time"? (assuming the columns are sorted transaction times/days)
col_mean = pd.DataFrame((log_quart.median())).reset_index()
col_mean.columns = ['index', 'mean']
col_mean['index'] = np.arange(0,train.shape[1],91)
g = sns.lmplot(data=col_mean, x='index', y='mean')