import pandas as pd
import numpy as np
df_camera = pd.read_csv('../input/camera_dataset.csv', na_values=0.0)
df_camera.head()
df_camera.describe()
%matplotlib inline

import matplotlib.pyplot as plt
from matplotlib import rc

rc('image', cmap='CMRmap')
from pandas.plotting import scatter_matrix
plt.figure()
_ = scatter_matrix(df_camera.drop(columns=['Price','Release date']), figsize=(12,12))
from scipy.stats import spearmanr

df_camera_v2 = df_camera.drop(columns=['Model'])
cor_, p_ = spearmanr(df_camera_v2, nan_policy='omit')
cor_ = np.abs(cor_)
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

img = ax.imshow(cor_)
plt.colorbar(img, shrink=0.8)

ax.set_xticks([xpos for xpos in np.linspace(ax.get_xlim()[0]+0.5,ax.get_xlim()[1]-0.5,df_camera_v2.shape[1])])
ax.set_yticks([ypos for ypos in np.linspace(ax.get_ylim()[0]-0.5,ax.get_ylim()[1]+0.5,df_camera_v2.shape[1])])
ax.set_xticklabels(df_camera_v2.columns, rotation='vertical')
ax.set_yticklabels(df_camera_v2.columns[::-1])
ax.tick_params(left=False, bottom=False, labeltop=True, labelbottom=False)
_ = ax.set_title('Correlation between properties of cameras', fontdict={'fontweight':800, 'fontsize':20})
years = np.sort(df_camera['Release date'].unique())

cols = [
    'Weight (inc. batteries)', 'Effective pixels', 'Max resolution', 'Price'
]
rel_dates = [ (df_camera[df_camera['Release date']==year]['Weight (inc. batteries)'].dropna()) 
     for year in years]
data_peryear = {}
for col in cols:
    data_peryear[col] = [ (df_camera[df_camera['Release date']==year][col].dropna()) 
                         for year in years]
data_peryear['log_Price'] = [np.log(x) for x in data_peryear.pop('Price')]

fig, axs= plt.subplots(2,2, sharex=True)
axs = axs.flatten()

fig.set_size_inches(12,12)

for ax, varname in zip(axs, data_peryear):
    data = data_peryear[varname]
    ax.boxplot(data)
    ax.set_xticklabels(years, rotation='vertical')
    ax.set_title(varname)
year_2007 = df_camera[df_camera['Release date']==2007].drop(columns=['Model', 'Release date'])
rest_data = df_camera[df_camera['Release date']!=2007].drop(columns=['Model', 'Release date'])
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

imp = Imputer(strategy='median')
rest_data = imp.fit_transform(rest_data)
year_2007 = imp.transform(year_2007)

X = year_2007[:,:-1]
y = year_2007[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train)

param_grid = [
    {'alpha':np.linspace(0.01,10,10)},
]
gcv = GridSearchCV(Ridge(), param_grid)
gcv.fit(X_train_mm, y_train)

X_test_mm = mm.transform(X_test)
gcv.score(X_test, y_test)
