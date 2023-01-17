import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

from geopy import distance

from scipy.stats import skew

from statsmodels.stats.outliers_influence import variance_inflation_factor



plt.style.use('seaborn-whitegrid')

sns.set_style('whitegrid')

plt.rcParams["axes.labelsize"] = 15



from bokeh.plotting import figure, show, output_notebook, output_file

from bokeh.tile_providers import CARTODBPOSITRON

from bokeh.transform import log_cmap

from bokeh.models import ColumnDataSource, LogTicker, ColorBar #, HoverTool, CategoricalColorMapper, LogColorMapper

from bokeh.models.formatters import BasicTickFormatter, NumeralTickFormatter

import bokeh.palettes as bp



output_notebook()

%matplotlib inline
df = pd.read_csv('../input/kc_house_data.csv')

df.head()
df.info()
df.drop(['id', 'date'], axis=1, inplace=True)

df.columns
fig, axes = plt.subplots(3,2, figsize=(18,16))

for xcol, ax in zip(['floors', 'waterfront', 'view', 'condition',

                     'grade', 'bedrooms'], axes.flatten()):

    sns.boxplot(xcol, 'price', data=df, ax=ax)

    



fig = plt.figure(figsize=(16, 8))

sns.boxplot('bathrooms', 'price', data=df)



plt.tight_layout()
features_cont = ['sqft_living', 'sqft_lot', 'sqft_above','sqft_basement', 

                'sqft_living15', 'sqft_lot15']



fig, axes = plt.subplots(3,2, figsize=(14,14))



for xcol, ax in zip(features_cont, axes.flatten()):

    sns.scatterplot(xcol, 'price', data=df, ax=ax)



plt.tight_layout()
def lgn2x(a):

    return a * (np.pi/180) * 6378137



def lat2y(a):

    return np.log(np.tan(a * (np.pi/180)/2 + np.pi/4)) * 6378137





# project coordinates

df['x_coor'] = df['long'].apply(lambda row: lgn2x(row))

df['y_coor'] = df['lat'].apply(lambda row: lat2y(row))



# creating the map

output_file("tile.html")

xmin, xmax =  df['x_coor'].min(), df['x_coor'].max() 

ymin, ymax =  df['y_coor'].min(), df['y_coor'].max() 



# range bounds supplied in web mercator coordinates

map_kc = figure(x_range=(xmin, xmax), y_range=(ymin, ymax),

           x_axis_type="mercator", y_axis_type="mercator", title="House Price on King County, USA",

           plot_width=700, plot_height=500,)



map_kc.title.text_font_size = '16pt'

map_kc.add_tile(CARTODBPOSITRON)



source = ColumnDataSource({'x':df['x_coor'], 'y':df['y_coor'], 'z':df['price']})

colormapper = log_cmap('z', palette=bp.Inferno256, low=df['price'].min(), high=df['price'].max())



map_kc.circle(x ='x', y='y', source=source, color=colormapper)



color_bar = ColorBar(color_mapper=colormapper['transform'], width=18, location=(0,0), 

                     ticker=LogTicker(), label_standoff=12)



color_bar.formatter = NumeralTickFormatter(format='0,0')

# color_bar.formatter = BasicTickFormatter(precision=3)



map_kc.add_layout(color_bar, 'right')



show(map_kc)
location = tuple(map(tuple, df[['lat', 'long']].values))

# the distance of every house from downtowm seattle

seattle_dt = (47.6050, -122.3344)



df['distance'] = [distance.distance(seattle_dt, loc).km for loc in location]



# df.drop(['lat', 'long', 'x_coor', 'y_coor'], axis=1, inplace=True)

df.drop(['x_coor', 'y_coor'], axis=1, inplace=True)



df.head()
def get_vif(data):

    

    X = data.iloc[:,1:]

    vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif["features"] = X.columns



    return vif.round(1)



get_vif(df)
plt.figure(figsize=(16,12))

sns.heatmap(df.corr(), annot=True)
df['sqft'] = df['sqft_living'] + df['sqft_above'] + df['sqft_living15']

df.drop(['sqft_living', 'sqft_above', 'sqft_living15'], axis=1, inplace=True)



df['sqft_lot_comb'] = df['sqft_lot'] + df['sqft_lot15']

df.drop(['sqft_lot', 'sqft_lot15'], axis=1, inplace=True)



df.head()
get_vif(df)
fig = plt.figure(figsize=(11,5))

fig = sns.distplot(df['price'])

fig.set(yticks=[]);
# computing skewness factor

skewness = df.apply(lambda x: skew(x))

skewness
# converting longtitude to positive values to enable us using the log function on all data

# this operation doesn't affect results, as all the whole 'long' column is negative

df['long'] = abs(df['long'])



skewed = skewness[skewness > 0.75].index



df[skewed] = np.log1p(df[skewed])



# plot the new target ditribution

fig = plt.figure(figsize=(11,5))

fig = sns.distplot(df['price'])

fig.set(yticks=[]);
fig, axes = plt.subplots(2,2, figsize=(16,10))



sns.scatterplot('sqft', 'price', data=df, ax=axes[0,0])

sns.scatterplot('sqft_lot_comb', 'price', data=df, ax=axes[0,1])

sns.boxplot('bedrooms', 'price', data=df, ax=axes[1,0])

sns.boxplot('grade', 'price', data=df, ax=axes[1,1])

axes[1,0].set_xticks([])

axes[1,1].set_xticks([])



plt.tight_layout()
get_vif(df)
# Standardizing the data

df = (df - df.mean()) / df.std()



get_vif(df)
def split_kfold(folds, i):    

    train = folds.copy() 

    test = folds[i]

    del train[i]

    train = np.concatenate(train, axis=0)

    d = train.shape[1]-1

    x_train, y_train = train[:, :d], train[:, d]

    x_test, y_test = test[:, :d], test[:, d]

    

    return x_train, x_test, y_train, y_test





def get_error(Y, Yhat):

    N = len(Y)   

    d1 = Y - Yhat

    d2 = Y - Y.mean()

    r2 = 1 - (d1.dot(d1) / d2.dot(d2))

    r2_adj = 1 - (1 - r2)*((N - 1) / (N - D - 1))

    mse = d1.dot(d1) / N

    return r2_adj, mse





def fit_kfold(X, Y, X_test, Y_test):

    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

    Yhat = X.dot(w)

    Yhat_test =  X_test.dot(w)

    r2_test, mse_test = get_error(Y_test, Yhat_test)

    

    return r2_test, w

# df_array = df[features].values

X = df.iloc[:,1:]

Y = df.iloc[:,0]



df_array = np.c_[X.values, Y.values]

k = 7

D = X.shape[1]

folds = np.array_split(df_array, k)



r2_test = []

coef = []



for i in range(k):

    x_train, x_test, y_train, y_test = split_kfold(folds, i)

    # prepare the array

    x_train = np.c_[np.ones(x_train.shape[0]), x_train]

    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    

    r2_test_temp, w = fit_kfold(x_train, y_train, x_test, y_test)

    r2_test.append(r2_test_temp)

    coef.append(w)

    

r2_test_kfold = sum(r2_test) / len(r2_test)

coef = np.sum(coef, axis=0) / len(coef)



indx = list(df.columns)

indx[0] = 'bias'

coef = pd.DataFrame(coef, index=indx, columns=['coef'])



print('Using  k-fold cross-validation where k = ', k,':')

print('R2_adjusted of the test data, using a simple linear regression, is: ', r2_test_kfold)
coef.reindex(coef['coef'].abs().sort_values(ascending=False).index)