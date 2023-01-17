%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# load the first part of data
appStore = pd.read_csv("../input/AppleStore.csv")
appdes = pd.read_csv("../input/appleStore_description.csv")
# merge two dataset together
df = appStore.merge(appdes, on = ["id", "track_name", "size_bytes"], how = 'left')
df.info()
# currency
# we are first to see the description of currency
df.groupby("currency").size()
# all of the price unit are USD that means this feature is useless. so we dropped it.
df = df.drop("currency", axis = 1);
# change the unit of size_bytes from bytes to Mb
df["size_bytes"] = 1.0 * df["size_bytes"] / (1024 * 1024)
print("mean: %f, std: %f" %(df["size_bytes"].median(), df["size_bytes"].std()))
def draw_hist(dataframe, name, percent = 0.95):
    tmp = dataframe[name]

    num = int(percent * tmp.shape[0])
    tmp = tmp.sort_values(ascending = True).head(num)

    # draw histgram
    data = [go.Histogram( x = tmp)]

    # add title
    layout = go.Layout(title = 'Frequency of ' + name,
                      xaxis = dict(title = name))

    # draw fig
    fig = go.Figure(data = data, layout = layout)

    iplot(fig, filename = name)
draw_hist(df, "size_bytes", 0.95)
# There are two kind of version number: 
    #<main version number>.<subversion number>.<stage version number>
    #digital number
# here we firstly split version number based on the symbol '.'
df['ver_format'] = df['ver'].map(lambda x: 1 if '.' in x else 0) 
df.groupby('ver_format').size()
# we used regExp to get the first number
df['ver_main'] = df['ver'].str.extract('([0-9]+)\.{1}')

# fill na using other format
df.loc[df['ver_format'] == 0, 'ver_main'] = 0

# change data type
df['ver_main'] = df['ver_main'].astype(int)

# show data info
df.groupby('ver_main').size().index
draw_hist(df, "ver_main", 0.95)
# we reset the main version number (> 13 and 0) as 0 
df.loc[df['ver_main'] > 13, "track_name"] = 0
def draw_Pie(dataframe, name):
    tmp = dataframe.groupby(name).size()

    # draw histgram
    data = [go.Pie(values = tmp.values, 
               labels = tmp.index)]

    # add title
    layout = go.Layout(title = 'Pie of ' + name,
                      margin = dict(l = 80, r = 100, t = 40, b = 40))

    # draw fig
    fig = go.Figure(data = data, layout = layout)

    iplot(fig, filename = name)
draw_Pie(df, "cont_rating")
# recode this feature
df['cont_rating.num'] = df['cont_rating'].str.extract('([0-9]+)').astype(int)
draw_Pie(df, "prime_genre")
# get the cols' name
cols = df.groupby('prime_genre').size().sort_values(ascending = False).head(11).index

# recode 
for col in cols:
    df[col] = df['prime_genre'].map(lambda x: 1 if col == x else 0).astype(int)
# get the len of app_desc as the feature
df['desc_length'] = df['app_desc'].map(lambda x: len(x))
cols = ["id", "size_bytes", "price", "sup_devices.num", "ipadSc_urls.num", 
        "lang.num", "vpp_lic", "ver_format", "ver_main", 
        "Games", "Entertainment", "Education", "Photo & Video", "Utilities", 
        "Health & Fitness", "Productivity", "Social Networking", "Lifestyle", 
        "Music", "Shopping", "desc_length"]

X = df[cols].astype(float)
y = df["user_rating_ver"]
# split file
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)
# size_bytes
def plot_three(df_x, df_y, name):
    x = df_x[name]
    tmp = pd.concat([x, df_y], axis = 1)
    tmp.head()
    tmp.columns = ["x", "y"]
    
    fig, axs = plt.subplots(ncols = 3, figsize=(20,5))
    sns.distplot(tmp['x'], ax = axs[0])
    sns.scatterplot(x = "x", y = "y", data = tmp, ax = axs[1])
    sns.boxplot(x = 'y', y = 'x', data = tmp, ax = axs[2])
plot_three(X_train, y_train, "size_bytes")
# we separate this variable and create a interact variable
def add_binary(data, feature_name, para):
    # generate name
    cate = feature_name + "_Binary"
    inter = feature_name + "_Int"
    
    # get mean and std, generate threshold
    mean = data[feature_name].mean()
    std = data[feature_name].std()
    threshold = mean + 3 * std
    
    # get cate
    if para == "mean":
        data[cate] = data[feature_name].map(lambda x: 1 if x < threshold else 0).astype(float)
    if para == "binary":
        data[cate] = data[feature_name].map(lambda x: 1 if x > 0 else 0).astype(float)
    if para == "lan":
        data[cate] = data[feature_name].map(lambda x: 0 if x == 1 else 1).astype(float)
    if para == "ver":
        data[cate] = data[feature_name].map(lambda x: 1 if x < 200 else 0).astype(float)
    # generate int, first stand, then multiple
    data[inter] = data[feature_name].map(lambda x: (x - mean) / std).astype(float)
    data[inter] = data[inter] * data[cate]
    
    return(data)
X_train = add_binary(X_train, 'size_bytes', "mean")
X_test = add_binary(X_test, 'size_bytes', 'mean')
X_train = add_binary(X_train, 'price', 'binary')
X_test = add_binary(X_test, 'price', 'binary')
# language
plot_three(X_train, y_train, "lang.num")
X_train = add_binary(X_train, 'lang.num', 'lan')
X_test = add_binary(X_test, 'lang.num', 'lan')
# vpp_lic
plot_three(X_train.loc[X_train["ver_main"] < 100,], y_train.loc[X_train["ver_main"] < 100,], "ver_main") # "ver_format", "ver_main"
# recode main version number
X_train = add_binary(X_train, 'ver_main', 'ver')
X_test = add_binary(X_test, 'ver_main', 'ver')
# view the whole data correlation array
tmp = pd.concat([y_train, X_train], axis = 1).corr()
tmp["user_rating_ver"].sort_values(ascending = False)
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.svm import LinearSVR, NuSVR,SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
 
classifiers = [
    LinearSVR(),
    NuSVR(),
    SVR(),
    DecisionTreeRegressor(),
    RandomForestRegressor(), 
    AdaBoostRegressor(), 
    GradientBoostingRegressor(),
    XGBRegressor(),
    Lasso(),
    Ridge()]

log_cols = ["Regressor", "MSE"]
log = pd.DataFrame(columns=log_cols)

mse_dict = {}
for _ in range(0, 20):
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        MSE = mean_squared_error(y_hat, y_test)
        if name in mse_dict:
            mse_dict[name] += MSE
        else:
            mse_dict[name] = MSE

for clf in mse_dict:
    mse_dict[clf] = mse_dict[clf]
    log_entry = pd.DataFrame([[clf, mse_dict[clf]]], columns = log_cols)
    log =log.append(log_entry)

plt.xlabel("MSE")
plt.title("Regressor Mean Standard Error")

sns.set_color_codes("muted")
sns.barplot(x = "MSE", y = "Regressor", data = log, color = "b")
# we select XGBregressor to fit the model
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
y_hat = my_model.predict(X_test)
MSE = mean_squared_error(y_hat, y_test)
MSE
