# Load Libraries:

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import plotly.offline as pyo 

import plotly.graph_objs as go

import plotly.figure_factory as ff

#

from scipy import stats

from scipy.stats import norm, skew

#

from sklearn.preprocessing import RobustScaler,StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.base import clone

#

import xgboost as xgb

#

import warnings

warnings.filterwarnings("ignore")

#

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

data.head()
# row

print("row count:",data.shape[0])

# columns

print("column count:",data.shape[1])
data.info()
data.describe().T
# But there are "?", (Missing Attribute Values: horsepower has 6 missing values)

#Just we did not see here

data.isnull().sum()
data.cov()
data.corr()
sns.pairplot(data,markers="*");

plt.show()
data["origin"].value_counts()
data["origin"].value_counts(normalize=True)
colors = ['#f4cb42', '#cd7f32', '#a1a8b5'] #gold,bronze,silver

#

origin_counts = data["origin"].value_counts(sort=True)

labels = ["USA", "Europe","Japan"]

values = origin_counts.values

#

pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))

layout = go.Layout(title='Origin Distribution')

fig = go.Figure(data=[pie], layout=layout)

pyo.iplot(fig)
trace0 = [go.Bar(x=data["model year"]+1900,y=data["mpg"],

                   marker=dict(color="rgb(17,77,117)"),

                   name="Total")]



layout = go.Layout(title="Consumption Gallon by Years",barmode="stack")

fig = go.Figure(data=trace0,layout=layout)   

pyo.iplot(fig) 
trace1 = [go.Scatter(x=data["horsepower"], y=data["weight"],

                   text=data["car name"],

                   mode="markers",

                   marker=dict(size=2*data["cylinders"],

                               color=data["cylinders"],

                               showscale=True))]

 

layout = go.Layout(title="Relation Horse Power & Weight",

                   xaxis=dict(title="Horse Power"),

                   yaxis=dict(title="Weight"),

                   hovermode="closest")

fig = go.Figure(data=trace1,layout=layout)

pyo.iplot(fig)
trace2 = [go.Histogram(x=data.mpg,

                         xbins=dict(start=0,end=50))]

layout = go.Layout(title="MPG")



fig = go.Figure(data=trace2,layout=layout)

pyo.iplot(fig)
trace3 = [go.Box(y=data["mpg"],name=data.columns[0]),

          go.Box(y=data["cylinders"],name=data.columns[1]),

          go.Box(y=data["displacement"],name=data.columns[2]),

          go.Box(y=data["horsepower"],name=data.columns[3]),

          go.Box(y=data["weight"],name=data.columns[4]),

          go.Box(y=data["acceleration"],name=data.columns[5]),

          go.Box(y=data["origin"],name=data.columns[7])]



pyo.iplot(trace3)
# I choice Germany instead of Euro

country_number = pd.DataFrame(index=["USA","DEU","JPN"],columns=["number","country"])

country_number["country"] = ["United States","Germany","Japan"]

country_number["number"] = [249,79,70]
country_number
worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',

                 z = country_number['number'], autocolorscale = True, reversescale = False, 

                 marker = dict(line = dict(color = 'rgb(180,180,180)', width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of athletes'))]



layout = dict(title = 'Distribution of Data', geo = dict(showframe = False, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

pyo.iplot(fig, validate=False)
data["horsepower"] = data["horsepower"].replace("?","100")

data["horsepower"] = data["horsepower"].astype(float)
threshoold       = 2

horsepower_desc  = data["horsepower"].describe()

q3_hp            = horsepower_desc[6]

q1_hp            = horsepower_desc[4]

IQR_hp           = q3_hp - q1_hp

top_limit_hp     = q3_hp + threshoold * IQR_hp

bottom_limit_hp  = q1_hp - threshoold * IQR_hp

filter_hp_bottom = bottom_limit_hp < data["horsepower"]

filter_hp_top    = data["horsepower"] < top_limit_hp

filter_hp        = filter_hp_bottom & filter_hp_top
data = data[filter_hp]

data.shape
data.columns
acceleration_desc  = data["acceleration"].describe()

q3_acc             = acceleration_desc[6]

q1_acc             = acceleration_desc[4]

IQR_acc            = q3_acc - q1_acc

top_limit_acc      = q3_acc + threshoold * IQR_acc

bottom_limit_acc   = q1_acc - threshoold * IQR_acc

filter_acc_bottom  = bottom_limit_acc < data["acceleration"]

filter_acc_top     = data["acceleration"] < top_limit_acc

filter_acc         = filter_acc_bottom & filter_acc_top
data = data[filter_acc]

data.shape
f,ax = plt.subplots(figsize = (20,7))

sns.distplot(data.mpg, fit=norm);

# we see that, data["mpg"] has positive skewness
(mu,sigma) = norm.fit(data["mpg"])

print("mu:{},sigma:{}".format(mu,sigma))
# qq plot:

plt.figure(figsize = (20,7))

stats.probplot(data["mpg"],plot=plt)

plt.show

print("We expect that our data points will be on red line for gaussian distributin. We see dist tails")
data["mpg"] = np.log1p(data["mpg"])
f,ax = plt.subplots(figsize = (20,7))

sns.distplot(data["mpg"], fit=norm);
# Let's other skewness of features

# if skew > 1  : positive skewness

# if skew > -1 : negative skewness



features = ['cylinders', 'displacement', 'horsepower', 'weight','acceleration','origin']

skew_list = []

for i in range(0,6):

    skew_list.append(skew(data.iloc[:,i]))

# So, features are good at skewness 

skew_list
data["origin"] = data["origin"].astype(str)

data["cylinders"] = data["cylinders"].astype(str)
data.drop(["car name"],axis=1,inplace=True)
# One Hot Encoding - 1

data = pd.get_dummies(data,drop_first=True)
# I decide to use model year as a fuaure, so i will apply one hot encoding

data["model year"] = data["model year"].astype(str)
# One Hot Encoding - 2

data = pd.get_dummies(data,drop_first=True)
data.info()
y = data["mpg"]

x = data.drop(["mpg"],axis=1)
# Creating Train and Test Datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)
# Scale the data to be between -1 and 1

# Mean= 0

# Std = 1

sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("LR Coef:",lr.intercept_)

print("LR Coef:",lr.coef_)

mse = mean_squared_error(y_test,y_pred)

print("MSE",mse)
ridge = Ridge(random_state=42, max_iter=10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = dict(alpha=alphas)
clf = GridSearchCV(ridge,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)

clf.fit(X_train,y_train)

scores = clf.cv_results_["mean_test_score"]

scores_std = clf.cv_results_["std_test_score"]
print("Ridge Coef:",clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator:",ridge)
y_pred_ridge = clf.predict(X_test)

mse_ridge = mean_squared_error(y_test,y_pred_ridge)

print("Ridge MSE:",mse_ridge)
lasso = Lasso(random_state=42, max_iter=10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = dict(alpha=alphas)
clf = GridSearchCV(lasso,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)

clf.fit(X_train,y_train)

scores = clf.cv_results_["mean_test_score"]

scores_std = clf.cv_results_["std_test_score"]
print("Lasso Coef:",clf.best_estimator_.coef_)

lasso = clf.best_estimator_

print("Lasso Best Estimator:",lasso)

print("Put Zero for redundat features:")
y_pred_lasso = clf.predict(X_test)

mse_lasso = mean_squared_error(y_test,y_pred_lasso)

print("Lasso MSE:",mse_lasso)
parameters = dict(alpha=alphas,l1_ratio=np.arange(0.0,1,0.05))

eNet = ElasticNet(random_state=42, max_iter=10000)

clf = GridSearchCV(eNet,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)

clf.fit(X_train,y_train)
print("Lasso Coef:",clf.best_estimator_.coef_)

eNet = clf.best_estimator_

print("Lasso Best Estimator:",eNet)
y_pred_eNet = clf.predict(X_test)

mse_eNet = mean_squared_error(y_test,y_pred_eNet)

print("Lasso MSE:",mse_eNet)
technices = ["Ridge","Lasso","ElasticNet"]

results   = [0.01822,0.01844,0.01813]
fig = go.Figure(data=[go.Bar(

            x=technices, y=results,

            text=results,

            textposition='auto',)])

fig.show()
# objective: aim

# n_estimator: number of trees

model_xgb = xgb.XGBRegressor(objective="reg:linear",max_depth=5,min_child_weight=4,subsample=0.7,n_estimator=1000,learning_rate=0.07)

model_xgb.fit(X_train,y_train)
y_pred_xgb = model_xgb.predict(X_test)

mse_xgb = mean_squared_error(y_test,y_pred_xgb)

print("XGBOOST MSE:",mse_xgb)
class AveragingModels():

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    # the reason of clone is avoiding affect the original base models

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]  

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)

        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)



averaged_models = AveragingModels(models = (model_xgb,lr))

averaged_models.fit(X_train,y_train)

y_pred_averaged_models = averaged_models.predict(X_test)

mse_averaged_models = mean_squared_error(y_test,y_pred_averaged_models)

print("Averaging Models MSE:",mse_averaged_models)