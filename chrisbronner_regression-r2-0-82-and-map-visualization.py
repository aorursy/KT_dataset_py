from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import validation_curve
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
df = pd.read_csv('../input/kc_house_data.csv')
# Convert date to timestamp
df['date'] = pd.to_datetime(df['date'])
df.head()
df.shape
df.columns
df.dtypes
df.corr()['price'].sort_values(ascending=False)
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),vmax=1.0,vmin=-1.0, square=True, fmt='.2f',
            annot=True, cbar_kws={"shrink": .75}, cmap='coolwarm')
plt.show()
plt.hist(df['price'], bins=200, alpha=0.3)
plt.grid()
plt.xlabel('House price ($)')
plt.ylabel('Transactions')
plt.xlim(0,1200000)
plt.show()
print('Maximum house price: \t${:0,.2f}'.format(df['price'].max()))
print('Minimum house price: \t${:0,.2f}'.format(df['price'].min()))
print('Mean house price: \t${:0,.2f}'.format(df['price'].mean()))
print('Median house price: \t${:0,.2f}'.format(df['price'].median()))
df_week = df.copy()
df_week['day_of_week'] = df_week['date'].dt.dayofweek
df_week = df_week.groupby('day_of_week', as_index=False).count()
# Group transactions by date
df_date = df.groupby('date', as_index=False).count()

# Plot transactions throughout a year
plt.figure(figsize=(12,6))
plt.plot(df_date['date'], df_date['id'])
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.grid()
plt.show()
plt.bar(df_week['day_of_week'], df_week['id'])
plt.xlabel('Day of Week')
plt.ylabel('Transactions')
plt.xticks(np.arange(7), ('Mon', 'Tue', 'Wed', 'Thu', 'Fri','Sat','Sun'))
plt.show()
plt.hexbin(df['sqft_living'], df['price'], gridsize=150, cmap='Blues')
plt.xlim(0,4000)
plt.ylim(0,1500000)
plt.xlabel('Square Footage of Living Space')
plt.ylabel('Price (USD)')
plt.show()
# Houses sold more than once
df_id = df.groupby('id', as_index=False).count()[['id','date']]
df_id = df_id[df_id['date']>1][['id']]
df_twice = df_id.merge(df, on='id', how='inner')
df_twice['price'].hist(bins=50)
plt.xticks(rotation=60)
plt.xlabel('Price (USD)')
plt.ylabel('Transactions')
plt.show()
df_twice[df_twice['id']==7200179]
# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# Transform date from single timestamp feature to multiple numerical 
# features
X['date_day'] = X['date'].dt.day
X['date_month'] = X['date'].dt.month
X['date_year'] = X['date'].dt.year
X['date_DoW'] = X['date'].dt.dayofweek
X = X.drop('date', axis=1)

# Split data set into training and test sets
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=0)
    
# Create standardized feature matrix
X_std = StandardScaler().fit_transform(X_train)
# Linear Regression
lr = LinearRegression()

# Evaluate model with cross-validation
cvs = cross_val_score(estimator=lr, X=X_train, 
                                    y=y_train, 
                                    cv=10, scoring='r2')
print('CV score: %.3f ± %.3f' % (cvs.mean(), cvs.std()))
lr.fit(X_train, y_train)
coef_list = list(lr.coef_)
name_list = list(X_train.columns)
pd.Series(coef_list, index=name_list)
def eval_model(estimator, X=X_train, y=y_train, out=True):
    '''
    Evaluates a model provided in 'estimator' and returns the 
    cross-validation score (as a list of 10 results)
    X: Feature matrix
    y: Targets
    '''
    cvs = cross_val_score(estimator=estimator, X=X, y=y, 
                          cv=10, scoring='r2')
    
    if out == True:
        print('CV score: %.3f ± %.3f' % (cvs.mean(), cvs.std()))
    
    return cvs
# Calculate correlation or each feature and create sorted 
# pandas Series of correlation factors
corr_ranking = df.corr()['price'].sort_values(ascending=False)[1:]
corr_ranking
feat_list = []       # List of features added to the model 
score_list = []      # List of CV scores obtained after adding features

# Loop over features in order of correlation with price
for feat in list(corr_ranking.index):
    
    feat_list.append(feat)      # Add feature name to feat_list
    
    # Calculate CV score
    cvs = eval_model(lr, X=X_train[feat_list], out=False)     
    
    score_list.append(cvs.mean())   # Add score to score_list
fig, ax = plt.subplots(1,1, figsize=(12,6))

# Accumulated CV score
ax.plot(score_list)
ax.set_xticks(list(range(len(feat_list))))
ax.set_xticklabels(feat_list, rotation=60)
ax.set_xlabel('Added feature')
ax.set_ylabel('CV score (R2) (blue)')
ax.grid()

# Derivative
axR = ax.twinx()
axR.set_ylabel('Individual feature contribution (orange)')
axR.bar(left=list(range(19)), height=np.diff([0]+score_list),
        alpha=0.4, color='orange')

plt.show()
features_range = range(1,24)
scores_list = []

for n in features_range:
    rfe = RFE(lr, n, step=1)
    rfe.fit(X_train, y_train)
    scores_list.append(rfe.score(X_train, y_train))
fig, ax = plt.subplots(1,1, figsize=(12,6))

# Regular sized plot
ax.plot(features_range, scores_list)
ax.set_xlabel('Number of features')
ax.set_ylabel('CV score (R2)')
ax.set_xticks(features_range)
ax.grid()

# Vertical magnification
axR = ax.twinx()
axR.plot(features_range, scores_list/max(scores_list), linestyle='--')
axR.set_ylabel('% of maximum CV score (dashed)')
axR.set_ylim(0.94,1.01)

plt.show()
# Fit RFE model with n_features_to_select=13 to X_train
rfe = RFE(lr, 13, step=1)
rfe.fit(X_train, y_train)

# Print list of selected columns
list(X_train.columns[rfe.support_])
X_rfe = X_train[list(X_train.columns[rfe.support_])]
eval_model(lr, X=X_std);
ridge = linear_model.Ridge()
eval_model(ridge, X=X_std);
# Trying different alpha parameters for Ridge regression

test_int = np.logspace(-10, 6, 17)

train_scores, valid_scores = \
            validation_curve(ridge, X_std, y_train, 'alpha',
                                test_int, cv=5)

# Plot validation curve
plt.figure(figsize=(12,6))
plt.title('Validation curve')
plt.semilogx(test_int, np.mean(train_scores, axis=1), 
             marker='o', label='Training data')
plt.semilogx(test_int, np.mean(valid_scores, axis=1), 
             marker='o', label='Validation data')
plt.legend()
plt.ylabel('CV score (R2)')
plt.xlabel('alpha parameter')
plt.grid()
plt.show()
lasso = linear_model.Lasso()
eval_model(lasso, X=X_std);
# Trying different alpha parameters for Lasso regression

test_int = np.logspace(-10, 6, 17)

train_scores, valid_scores = \
            validation_curve(lasso, X_std, y_train, 'alpha',
                                test_int, cv=5)

# Plot validation curve
plt.figure(figsize=(12,6))
plt.title('Validation curve')
plt.semilogx(test_int, np.mean(train_scores, axis=1), 
             marker='o', label='Training data')
plt.semilogx(test_int, np.mean(valid_scores, axis=1), 
             marker='o', label='Validation data')
plt.legend()
plt.ylabel('CV score (R2)')
plt.xlabel('alpha parameter')
plt.grid()
plt.show()
svr = SVR(kernel='linear', C=1)
eval_model(svr, X=X_std);
forest = RandomForestRegressor(max_depth=4, random_state=0,
                              n_estimators=200)
eval_model(forest, X=X_std);
%timeit eval_model(lr, X=X_std, out=False)
%timeit eval_model(forest, X=X_std, out=False)
%timeit eval_model(forest, X=X_rfe, out=False)
# Second order (quadratic) polynomials with linear regression
pipe = Pipeline([('poly', PolynomialFeatures(2)),
                 ('lr', lr)])
eval_model(pipe, X=X_train); # quadratic terms, unstandardized features
eval_model(pipe, X=X_std);   # quadratic terms, standardized features
# Third order (cubic) polynomials with linear regression
pipe = Pipeline([('poly', PolynomialFeatures(3)),
                 ('lr', lr)])
eval_model(pipe, X=X_train); # cubic terms, unstandardized features
pipe = Pipeline([('rfe', RFE(lr, 13, step=1)),
                 ('poly', PolynomialFeatures(3)),
                 ('lr', lr)])
eval_model(pipe, X=X_std);
pipe = Pipeline([('pca', PCA(n_components=13)),
                 ('lr', lr)])
eval_model(pipe, X=X_std);
pipe = Pipeline([('pca', PCA(n_components=13)),
                 ('poly', PolynomialFeatures(3)),
                 ('lr', lr)])
eval_model(pipe, X=X_std);
scores_list = []  # List in which scores will be saved

# Loop over values for n_components
for n_components in range(7,17):
    
    # Build pipe
    pipe = Pipeline([('pca', PCA(n_components=n_components)),
                     ('poly', PolynomialFeatures(3)),
                     ('lr', lr)])
    
    # Evaluate model, print, and save score in scores_list
    cvs = eval_model(pipe, X=X_std, out=False);
    print('n_components = %d :  %.3f pm %.3f' % \
          (n_components, cvs.mean(), cvs.std()))
    scores_list.append(cvs.mean())
plt.figure(figsize=(12,6))
plt.xlabel('n_components')
plt.ylabel('CV score (R2)')
plt.plot(range(7,17), scores_list, marker='o')
plt.grid()
plt.show()
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import bwr # import color map

plt.figure(figsize=(12, 12))

# Create map with basemap
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat = df['lat'].min()+0.1, 
            llcrnrlon = df['long'].min()-0.1,
            urcrnrlat = df['lat'].max(), 
            urcrnrlon = df['long'].max()-0.6)

# Load satellite image (deactivated for Kaggle)
#m.arcgisimage(service='ESRI_Imagery_World_2D', 
#              xpixels=1500, verbose=True)

# Reducing number of properties shown
plot_df = df[::10]

for index, house in plot_df.iterrows(): # Loop over houses
    
    # Get position on the map from geo coordinates
    x,y = m(house['long'], house['lat'])
    
    #  Get color from price
    price_min = 200000
    price_max = 800000
    price_norm = (house['price'] - price_min) / (price_max - price_min)

    rgb_exp = price_norm
    rgb=[0,0,0]
    for i in range(3): rgb[i] = int(bwr(rgb_exp)[i]*255)
    color_hex = "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])
    
    #Plot data point
    plt.plot(x,y, 'o', markersize=6, color=color_hex, alpha=0.6)
   
plt.show()