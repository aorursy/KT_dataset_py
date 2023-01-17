import numpy as np 
import pandas as pd 
import scipy
import random
random.seed(10)
np.random.seed(11)


from scipy import stats
from scipy.stats import norm
import missingno as msno
import datetime

#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

# Ploting libs

from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "notebook" 
# As after installing vscode, renderer changed to vscode, 
# which made graphs no more showed in jupyter.

from yellowbrick.regressor import ResidualsPlot


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_palette('RdBu')
df = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
print('Observations                 : ', df.shape[0])
print('Features -- exclude the Price: ', df.shape[1] - 1)
# Datatypes
df.info()
df.head(5)
# zero values
(df==0).sum().sort_values(ascending=False).head(6)
# Zeroes to Missing in Landsize and BuildingArea
df['Landsize'].replace(0, np.nan, inplace=True)
df['BuildingArea'].replace(0, np.nan, inplace=True)
# Extract Month & Year from Date, then drop Date

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.drop('Date', axis=1, inplace=True)
# Drop: Texts
df.drop(['Suburb', 'Address', 'SellerG'], axis=1, inplace=True)
# A Brief of Missing data
total_miss   = df.isnull().sum().sort_values(ascending=False)

percent      = total_miss / df.shape[0]

table = pd.concat([total_miss, percent], axis=1, keys=['Numbers', 'Percent'])
print(table.head(15))
# Drop: Missing > 40%
df.drop(['BuildingArea', 'YearBuilt', 'Landsize'], axis=1, inplace=True)
# Drop: Missing in Price
df.dropna(subset=['Price'], axis=0, inplace=True)
# Drop: Minorities
df.dropna(subset=['Propertycount', 'Regionname', 'CouncilArea', 'Postcode', 'Distance'],
          axis=0, inplace=True)
df.describe(percentiles=[0.01, 0.25, 0.75, 0.99])
# Texts with too many of uniques
df.drop(['Postcode', 'Propertycount'], axis=1, inplace=True)
df.describe(include='O').sort_values(axis=1, by=['unique'], ascending=False)
df.drop('CouncilArea', axis=1, inplace=True)
#Classify features based on Datatypes, helpful for EDA.

continuous_features = ['Price',      'Distance']

discrete_features  = ['Bathroom',    'Bedroom2',       'Car',        'Rooms']

category_features  = ['Type',        'Method',         'Regionname']
                     
sns.distplot(df['Price'], fit=norm);
df[continuous_features].hist(bins=40, figsize=(18,9))
plt.show()
df[discrete_features].hist(bins=40, figsize=(20,20))
plt.show()
# First try for Total sales per Region

# plotly.offline.init_notebook_mode(connected=True)

regions = df.Regionname.unique()
total_values_per_region = [df['Price'][df.Regionname==region].sum() for region in regions]

fig = px.bar(y=regions, x=total_values_per_region,
             title='Total Sales per Regions', orientation='h',
             template='plotly_white')

fig.update_layout(xaxis={'title':'Price'},
                  yaxis={'title':'Regions'})

fig.show()
fig = px.box(df, x='Regionname', y='Price', template='simple_white')
fig.update_layout(title='Price by Regions')
# IQR score
def IQR_outlier_detect(data=df, features=[]):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        outside_IQR = (data[feature]<=(Q1-1.5*IQR)) | ((Q3+1.5*IQR)<=data[feature])  
        outside_IQR = outside_IQR.sum()        
        
        print('Outside of IQR: %s -- Total: %d -- percent %2.2f'% (feature, outside_IQR, outside_IQR/df.shape[0]))
    return

IQR_outlier_detect(df, features=['Price'])
fig = px.scatter(df, x='Longtitude', y='Lattitude', color='Price')
fig.update_layout(title='Price by Locations')
# Price vs Continuous Features

corr_matrix = df[continuous_features].corr()

figure = plt.figure(figsize=(16,12))

mask = np.triu(corr_matrix) # Hide the upper part.
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)

plt.show()
# Price vs Discrete Features

corr_matrix = df[discrete_features + ['Price']].corr()

figure = plt.figure(figsize=(16,12))

mask = np.triu(corr_matrix) # Hide the upper part.
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)

plt.show()

df.drop('Bedroom2', axis=1, inplace=True)

discrete_features.remove('Bedroom2')
# First, detect Outliers
features = continuous_features + discrete_features
IQR_outlier_detect(df, features)
# Remove Outliers
def IQR_outlier_remove(data=df, features=[]):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # the core: the ~ is a must to avoid removing NaN.
        outside_IQR = (data[feature]<=(Q1-1.5*IQR)) | ((Q3+1.5*IQR)<=data[feature])
        data = data[~outside_IQR]
        print('Cleaning: ', feature)
        print('Q1: %2.2f', Q1)
        print('Q2: %2.2f', Q3)
        print('After cleaning, data left: %d \n' % (data.shape[0]))
        
        # debug
        #inside_IQR = ((Q1-1.5*IQR)<= data[feature]) & (data[feature]<=(Q3+1.5*IQR))
        
    return data

# Driving code
features = continuous_features + discrete_features
df = IQR_outlier_remove(df, features)
# How much observations left?
df.shape
df.dtypes
features_to_scaler = ['Rooms', 'Distance', 'Bathroom', 'Car',
                        'Lattitude', 'Longtitude',
                        'Month', 'Year']
df_std = df
scaler = StandardScaler()

for feature in features_to_scaler:
    df_std[feature] = scaler.fit_transform(df_std[feature].values.reshape(-1, 1))
df_std.head()
df_std.head()
df_encode = pd.get_dummies(df_std)
df_encode.dtypes
# A Brief of Missing data

total_miss   = df.isnull().sum().sort_values(ascending=False)

percent      = total_miss / df.shape[0]

table = pd.concat([total_miss, percent], axis=1, keys=['Numbers', 'Percent'])
print(table.head(8))
msno.heatmap(df)
df_encode.dtypes
# K-nn imputation
neighbors = 10
imputer = KNNImputer(n_neighbors=neighbors)

df_filled = imputer.fit_transform(df_encode)

# to Dataframe
df_filled = pd.DataFrame(df_filled)
df_filled.head()
y = df_filled[1]

X = df_filled.drop(labels=1, axis=1)
# Normality of y
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)
y = np.log(y)

# Check again
# Normality of y
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
A = Ridge(alpha=0)

A.fit(X_train, y_train)
print("A's score: %2.4f" % A.score(X_test, y_test))
# B is the same as A but with CV

B = RidgeCV(alphas=[0], cv=5, scoring='r2')

B.fit(X_train, y_train)
print("A's score: %2.4f" % B.score(X_test, y_test))
# Finding the best k-folds

B_score = []
cv = []

for i in range(2, 11):
    model = Ridge(alpha=0, normalize=True)
    score = cross_val_score(model, X_train, y_train, cv=i).mean()
    if score<0 : score = 0
    B_score.append(round(score, 5))
    cv.append(i)
    
    print("cv: %d --- score: %2.5f" % (i, score))
    
B_score = [0 if score<0 else score for score in B_score]
print(B_score)

px.line(x=cv, y=B_score, 
        template='simple_white', 
        title='<b>K-fold vs R2</b>',
        labels={'x':'K-fold', 'y':'R2'})

cv = 8
params = {'alpha':[100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'normalize': (True, False)}

model = Ridge()
gsc = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)
# With those best params, plot: Residuals vs Prediction

B = gsc.best_estimator_
B.fit(X_train, y_train)
print("B's score: %2.4f" % B.score(X_test, y_test))

visualizer = ResidualsPlot(B)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 
from yellowbrick.regressor import PredictionError
from sklearn.linear_model import Lasso

model = PredictionError(B)
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.show()
pca = PCA()
pca.fit(X_train)

cumsum = pca.explained_variance_ratio_.cumsum() // 0.01
n_comp = [i for i in range(1, len(cumsum)+1, 1)]

print(cumsum)
px.bar(y=cumsum, x=n_comp, text=cumsum)
pipe = Pipeline([
                ('PCA', PCA(n_components=10)),
                ('Linear Regression', Ridge(alpha=0, normalize=True))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# D
step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , RidgeCV(alphas=[0], cv=7) )]

D = Pipeline(step)
D.fit(X_train, y_train)
score = D.score(X_test, y_test)
print("D's score: %2.4f" % score)
step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , Ridge() )]
pipe = Pipeline(step)

params = {'PCA__n_components' : range(1,24),
          'Lin_Reg__alpha'    : [100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'Lin_Reg__normalize': [True, False]}

gsc = GridSearchCV(pipe, param_grid=params, cv=7)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)
D = gsc.best_estimator_
D.fit(X_train, y_train)
print("B's score: %2.4f" % D.score(X_test, y_test))

visualizer = ResidualsPlot(D)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 