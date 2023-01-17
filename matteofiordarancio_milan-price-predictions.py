import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
%matplotlib inline
apartments = pd.read_csv("../input/milan-airbnb-open-data-only-entire-apartments/Airbnb_Milan.csv")
apartments.info()
apartments.head()
apartments.head()
apartments.drop(apartments.columns[[0, 1, 2]],axis=1,inplace=True)
apartments.drop(columns=["zipcode"], inplace=True)
weekly_price = apartments.cleaning_fee + apartments.daily_price * 7
apartments["weekly_price"] = weekly_price.values
apartments.drop(columns=["cleaning_fee", "daily_price"], inplace=True)

apartments.room_type.value_counts()
apartments.drop(columns=["room_type"], inplace=True)
import urllib

cmap = cm.jet
m = cm.ScalarMappable(cmap=cmap)
quartiere_colors = m.to_rgba(apartments.neighbourhood_cleansed)
quant_minimum = apartments.weekly_price.quantile(0.1)
quant_maximum = apartments.weekly_price.quantile(0.9)
price = ((apartments.weekly_price - quant_minimum) / (quant_maximum - quant_minimum)) * 30

#initializing the figure size
plt.figure(figsize=(20,20))
#loading the png milan image found on open street map and saving to my local folder along with the project
i=urllib.request.urlopen('https://i.ibb.co/s1Jf5k7/map-2.png')
mil_img=plt.imread(i)
#scaling the image based on the latitude and longitude max and mins for proper output
plt.imshow(mil_img, zorder=0, extent=[
                                      apartments.longitude.min(), 
                                      apartments.longitude.max(), 
                                      apartments.latitude.min(), 
                                      apartments.latitude.max(),
                                    ]
           )

ax=plt.gca()
# plot the points
apartments.plot(kind='scatter', x='longitude', y='latitude', label='price', c=quartiere_colors, s=price, ax=ax, zorder=5, edgecolors='black')

patch = []
for a in range(1, 9):
  patch.append(mpatches.Patch(color=m.to_rgba(a), label='Neighborhood ' + str(a)))

plt.legend(handles=patch)

plt.show()
apartments.neighbourhood_cleansed.value_counts()
def draw_plot(apartments):
    cmap = cm.jet
    m = cm.ScalarMappable(cmap=cmap)
    quartiere_colors = m.to_rgba(apartments.neighbourhood_cleansed)
    quant_minimum = apartments.weekly_price.quantile(0.1)
    quant_maximum = apartments.weekly_price.quantile(0.9)
    price = ((apartments.weekly_price - quant_minimum) / (quant_maximum - quant_minimum)) * 30

    #initializing the figure size
    plt.figure(figsize=(20,20))
    #loading the png milan image found on open street map and saving to my local folder along with the project
    i=urllib.request.urlopen('https://i.ibb.co/s1Jf5k7/map-2.png')
    mil_img=plt.imread(i)
    #scaling the image based on the latitude and longitude max and mins for proper output
    plt.imshow(mil_img, zorder=0, extent=[
                                          apartments.longitude.min(), 
                                          apartments.longitude.max(), 
                                          apartments.latitude.min(), 
                                          apartments.latitude.max(),
                                        ]
              )

    ax=plt.gca()
    # plot the points
    apartments.plot(kind='scatter', x='longitude', y='latitude', label='price', c=quartiere_colors, s=price, ax=ax, zorder=5, edgecolors='black')
    patch = []
    for a in range(1, 9):
      patch.append(mpatches.Patch(color=m.to_rgba(a), label='Neighborhood ' + str(a)))

    plt.legend(handles=patch)
    plt.show()
draw_plot(apartments[apartments.weekly_price > 800])
draw_plot(apartments[apartments.weekly_price < 350])
apartments.number_of_reviews.describe()
draw_plot(apartments[apartments.number_of_reviews > 44])
draw_plot(apartments[apartments.number_of_reviews <= 4])
draw_plot(apartments[apartments.bedrooms > 2])
draw_plot(apartments[apartments.bedrooms == 1])
apartments.boxplot(by="neighbourhood_cleansed", column="weekly_price", figsize=(10,10), showmeans=True)
# taken from https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
f = plt.figure(figsize=(25, 20))
plt.matshow(apartments.corr(method='pearson'), fignum=f.number)
plt.xticks(range(apartments.shape[1]), apartments.columns, fontsize=14, rotation=90)
plt.yticks(range(apartments.shape[1]), apartments.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
from geopy.distance import great_circle

def distance_to_mid(lat, lon):
    milan_centre = (45.464664, 9.188540)
    accommodation = (lat, lon)
    return great_circle(milan_centre, accommodation).km

apartments["dist_from_center"] = apartments.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)
from sklearn.feature_selection import VarianceThreshold

def remove_almost_constant_columns(threshold=0):
    qconstant_filter = VarianceThreshold(threshold=threshold)
    qconstant_filter.fit(apartments)
    constant_columns = [column for column in apartments.columns
                        if column not in apartments.columns[qconstant_filter.get_support()]]
    print(constant_columns)
    apartments.drop(labels=constant_columns, axis=1, inplace=True)

remove_almost_constant_columns(threshold=0.03) # we remove data that is 97% of the time the same
import seaborn as sns
from scipy.stats import norm

def plot_price_distribution(prices):
  plt.figure(figsize=(10,10))
  sns.distplot(prices, fit=norm)
  plt.title("Price Distribution Plot", size=15, weight='bold')

plot_price_distribution(apartments.weekly_price)
log_weekly_price = np.log2(apartments.weekly_price)
plot_price_distribution(log_weekly_price)
apartments["log_weekly_price"] = log_weekly_price.values
from sklearn.model_selection import train_test_split

X = apartments.drop(columns=["log_weekly_price", "weekly_price"])
y = apartments.log_weekly_price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.model_selection import KFold
kf = KFold(5, shuffle=True, random_state=42)
from sklearn.model_selection import cross_validate

def train_and_validate(model, X, y, cv):
    cv_result = cross_validate(model, X, y, cv=kf, return_train_score=True)
    return pd.DataFrame(cv_result)
from statsmodels.stats.proportion import proportion_confint

def confidence_interval(n_elements, R2_score, confidence):    
    return proportion_confint(n_elements * R2_score, n_elements, 1-confidence/100, method='wilson')

def print_confidence_interval(n_elements, R2_score):
    lower, upper = confidence_interval(n_elements, R2_score, 95)
    print(f"Interval of confidence: {lower:.3f}, {upper:.3f}")

from sklearn.linear_model import LinearRegression

result = train_and_validate(LinearRegression(), X, y, kf)
print(result)
print(train_and_validate(LinearRegression(), X, apartments.weekly_price, kf))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scale",  StandardScaler()),   # <- aggiunto
    ("linreg", LinearRegression())
])
print(train_and_validate(model, X, y, kf))

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

model = Pipeline([
    ("scale",  StandardScaler()),   # <- aggiunto
    ("regr", ElasticNet())
])

grid = {
    "regr__l1_ratio": np.linspace(0, 1, 10),      # <- grado polinomio
    "regr__alpha":  [0.1, 1, 10] # <- regolarizzazione
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(X_train, y_train);

display(pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False))
print_confidence_interval(len(X_test), gs.score(X_test, y_test))

from sklearn.kernel_ridge import KernelRidge
# best param alpha = 50, coef0=4, degree = 3
model = Pipeline([
    ("scale", StandardScaler()),
    ("regr",  KernelRidge(alpha=20, kernel="poly", degree=3, coef0=2))
])
grid = {
    "regr__alpha":  np.linspace(50, 200, 3), # <- regolarizzazione
    "regr__coef0": [4,5,6,7,3],
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(X_train, y_train);
display(pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False))
print_confidence_interval(len(X_test), gs.score(X_test, y_test))
red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
apartments.log_weekly_price.plot(kind='box', xlim=(6, 15), vert=False, flierprops=red_square, figsize=(20,2));
apartments.drop(apartments[(apartments.log_weekly_price > 13) | (apartments.log_weekly_price < 7) ].index, axis=0, inplace=True)
apartments.log_weekly_price.plot(kind='box', xlim=(7, 13), vert=False, flierprops=red_square, figsize=(20,2));
X = apartments.drop(columns=["log_weekly_price", "weekly_price"])
y = apartments.log_weekly_price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()

grid = { 
            "n_estimators"      : [10,20,30,50,100],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            "max_depth": [1,20,100, None],
            "min_samples_leaf": [1, 5, 10]
}
gs = GridSearchCV(model, grid, cv=kf)
gs.fit(X_train, y_train);
display(pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False))
print_confidence_interval(len(X_test), gs.score(X_test, y_test))
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


booster = xgb.XGBRegressor()
# create Grid
param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2],
              'alpha': [0.0, 0.5, 1, 2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

data_dmatrix = xgb.DMatrix(data=X,label=y)
booster = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.6, learning_rate = 0.05,
                max_depth = 6, gamma = 0, alpha=1, n_estimators = 300)

booster.fit(X_train,y_train)

preds = booster.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
print(f"R2 score: {r2_score(y_test, preds)}")
print_confidence_interval(len(X_test), r2_score(y_test, preds))

lin = LinearRegression()
lin.fit(X_train, y_train)
preds = lin.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
print(f"R2 score: {r2_score(y_test, preds)}")
print_confidence_interval(len(X_test), r2_score(y_test, preds))


xg_train = xgb.DMatrix(data=X_train, label=y_train)
params = {'colsample_bytree': 0.6, 'gamma': 0, 'alpha': 1,  'learning_rate': 0.05, 'max_depth': 6}

cv_results = xgb.cv(dtrain=xg_train, params=params, nfold=4,
                    num_boost_round=400, early_stopping_rounds=10, 
                    metrics="rmse", as_pandas=True)
cv_results.tail()
train_and_validate(booster, X, y, kf)
# we put back all the outliers by re-executing code above
train_and_validate(booster, X, y, kf)
X_no_airbnb_data = X.drop(columns=[
                                   "number_of_reviews", 
                                   "review_scores_rating", 
                                   "review_scores_accuracy", 
                                   "review_scores_cleanliness", 
                                   "review_scores_checkin", 
                                   "review_scores_communication", 
                                   "review_scores_location", 
                                   "review_scores_value",
                                  ])
train_and_validate(booster, X_no_airbnb_data, y, kf)
y_train.plot.hist(bins=40, figsize=(12, 4));
np.random.seed(42)
random_preds = np.random.normal(
    y_train.mean(),   # centro (media)
    y_train.std(),    # scala (dev. standard)
    len(y)        # numero di campioni
)
plt.figure(figsize=(12, 4))
plt.hist(random_preds, bins=40);
scores = []
for i in range(1, 1000):
  np.random.seed(i)
  random_preds = np.random.normal(
      y_train.mean(),   # centro (media)
      y_train.std(),    # scala (dev. standard)
      len(y)        # numero di campioni
  )
  scores.append(r2_score(y, random_preds))

np.mean(scores)
import xgboost
import shap
# download shap with pip3 install https://github.com/slundberg/shap/archive/master.zip
# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.summary_plot(shap_values, X)