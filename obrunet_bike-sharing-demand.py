import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt



pd.options.display.max_columns = 100



import warnings

warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb

import lightgbm as lgbm
df = pd.read_csv('../input/bike-sharing-demand/train.csv',header = 0)

df.head()
df.shape
df.info()
df.describe()
plt.figure(figsize=(8, 4))

sns.distplot(df.temp, bins=10, label='real temp.')

sns.distplot(df.atemp, bins=10, label='feels like temp.')

plt.legend()

plt.show()
plt.figure(figsize=(4, 4))

sns.scatterplot(df.atemp, df.temp)

plt.show()
df['Delta'] = df.temp - df.atemp

sns.lineplot(x=df.index, y=df.Delta)
df[df.Delta > 5].head()
df[(df.atemp >= 12) & (df.atemp <= 12.5)].head()
plt.figure(figsize=(8, 4))

sns.distplot(df.humidity, bins=10, label='humidity')

sns.distplot(df.windspeed, bins=10, label='windspeed')

plt.legend()

plt.show()
sns.pairplot(df[['temp', 'atemp', 'humidity', 'windspeed']])
df['casual_percentage'] = df['casual'] / df['count']

df['registered_percentage'] = df['registered'] / df['count']
def change_datetime(df):

    """ Modify the col datetime to create other cols: dow, month, week..."""

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["dow"] = df["datetime"].dt.dayofweek

    df["month"] = df["datetime"].dt.month

    df["week"] = df["datetime"].dt.week

    df["hour"] = df["datetime"].dt.hour

    df["year"] = df["datetime"].dt.year

    df["season"] = df.season.map({1: "Winter", 2 : "Spring", 3 : "Summer", 4 :"Fall" })

    df["month_str"] = df.month.map({1: "Jan ", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec" })

    df["dow_str"] = df.dow.map({5: "Sat", 6 : "Sun", 0 : "Mon", 1 :"Tue", 2 : "Wed", 3 : "Thu", 4: "Fri" })

    df["weather_str"] = df.weather.map({1: "Good", 2 : "Normal", 3 : "Bad", 4 :"Very Bad"})

    return df

    

    

df = change_datetime(df)

df.head()
sns.kdeplot(data=df['count'])
df['y_log'] = np.log(df['count'])

sns.kdeplot(data=df['y_log'])
plt.figure(figsize=(12, 6))

sns.pointplot(x=df["hour"], y=df["count"], hue=df["season"])

plt.xlabel("Hour Of The Day")

plt.ylabel("Users Count") 

plt.title("Rentals Across Hours")

plt.show()
plt.figure(figsize=(12, 6))

sns.lineplot(x="hour", y="count", hue="season", data=df)
# ---------------------------------------------------------

plt.figure(figsize=(6,3))

plt.stackplot(range(1,25),

              df.groupby(['hour'])['casual_percentage'].mean(), 

              df.groupby(['hour'])['registered_percentage'].mean(), 

              labels=['Casual','Registered'])

plt.legend(loc='upper left')

plt.margins(0,0)

plt.title("Evolution of casual /registered bikers' share over hours of the day")



# ---------------------------------------------------------

plt.figure(figsize=(6,6))

df_hours = pd.DataFrame(

    {"casual" : df.groupby(['hour'])['casual'].mean().values,

    "registered" : df.groupby(['hour'])['registered'].mean().values},

    index = df.groupby(['hour'])['casual'].mean().index)

df_hours.plot.bar(rot=0)

plt.title("Evolution of casual /registered bikers numbers over hours of the day")



# ---------------------------------------------------------

plt.show()
plt.figure(figsize=(12, 6))

sns.pointplot(x=df["dow"], y=df["count"], hue=df["season"])

plt.xlabel("Day of the week")

plt.ylabel("Users Count") 

plt.title("Rentals Across week days")

plt.show()
# ---------------------------------------------------------

plt.figure(figsize=(6,3))

plt.stackplot(range(1,8),

              df.groupby(['dow'])['casual_percentage'].mean(), 

              df.groupby(['dow'])['registered_percentage'].mean(), 

              labels=['Casual','Registered'])

plt.legend(loc='upper left')

plt.margins(0,0)

plt.title("Evolution of casual /registered bikers' share over weekdays")



# ---------------------------------------------------------

plt.figure(figsize=(6,6))

df_hours = pd.DataFrame(

    {"casual" : df.groupby(['dow'])['casual'].mean().values,

    "registered" : df.groupby(['dow'])['registered'].mean().values},

    index = df.groupby(['dow'])['casual'].mean().index)

df_hours.plot.bar(rot=0)

plt.title("Evolution of casual /registered bikers numbers over weekdays")



# ---------------------------------------------------------

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10, 8)

sns.boxplot(data=df, y="count", x="month_str", orient="v")

ax.set(xlabel="Months" , ylabel="Count", title="Count Across Month");
sns.swarmplot(x='hour', y='temp', data=df, hue='season')

plt.show()
# ---------------------------------------------------------

plt.figure(figsize=(6,3))

plt.stackplot(range(1,13),

              df.groupby(['month'])['casual_percentage'].mean(), 

              df.groupby(['month'])['registered_percentage'].mean(), 

              labels=['Casual','Registered'])

plt.legend(loc='upper left')

plt.margins(0,0)

plt.title("Evolution of casual /registered bikers' share over months of the year")



# ---------------------------------------------------------

plt.figure(figsize=(6,6))

df_hours = pd.DataFrame(

    {"casual" : df.groupby(['month'])['casual'].mean().values,

    "registered" : df.groupby(['month'])['registered'].mean().values},

    index = df.groupby(['month'])['casual'].mean().index)

df_hours.plot.bar(rot=0)

plt.title("Evolution of casual /registered bikers numbers over months of the year")



# ---------------------------------------------------------

plt.show()
plt.figure(figsize=(10, 5))



bars = ['casual not on working days', 'casual on working days',\

        'registered not on working days', 'registered on working days',\

        'casual not on holidays', 'casual on holidays',\

        'registered not on holidays', 'registered on holidays']



qty = [df.groupby(['workingday'])['casual'].mean()[0], df.groupby(['workingday'])['casual'].mean()[1],\

      df.groupby(['workingday'])['registered'].mean()[0], df.groupby(['workingday'])['registered'].mean()[1],\

      df.groupby(['holiday'])['casual'].mean()[0], df.groupby(['holiday'])['casual'].mean()[1],\

      df.groupby(['holiday'])['registered'].mean()[0], df.groupby(['holiday'])['registered'].mean()[1]]



y_pos = np.arange(len(bars))

plt.barh(y_pos, qty, align='center')



plt.yticks(y_pos, labels=bars)

#plt.invert_yaxis()  # labels read top-to-bottom

plt.xlabel('Mean nb of bikers')

plt.title("Number of bikers on holidays / working days")

plt.show()
# ---------------------------------------------------------

plt.figure(figsize=(6,3))

plt.stackplot(range(1,5),

              df.groupby(['season'])['casual_percentage'].mean(), 

              df.groupby(['season'])['registered_percentage'].mean(), 

              labels=['Casual','Registered'])

plt.legend(loc='upper left')

plt.margins(0,0)

plt.title("Evolution of casual /registered bikers' share over seasons")



# ---------------------------------------------------------

plt.figure(figsize=(6,6))

df_hours = pd.DataFrame(

    {"casual" : df.groupby(['season'])['casual'].mean().values,

    "registered" : df.groupby(['season'])['registered'].mean().values},

    index = df.groupby(['season'])['casual'].mean().index)

df_hours.plot.bar(rot=0)

plt.title("Evolution of casual /registered bikers numbers over seasons")



# ---------------------------------------------------------

plt.show()
sns.set(style="white")



# Compute the correlation matrix

corr = df[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize=(5, 5))

sns.scatterplot(df.registered, df['count'])

plt.show()
# target 

y = (df["count"])



# drop irrelevant cols and target

cols_dropped = ["count", "datetime", "atemp", "month_str", "season", "dow_str", "weather_str",\

                "casual", "registered", "casual_percentage", "registered_percentage", "y_log", "Delta"] 

X = df.drop(columns=cols_dropped)

            

X.shape, y.shape
y.head()
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def get_rmse(reg, model_name):

    """Print the score for the model passed in argument and retrun scores for the train/test sets"""

    

    y_train_pred, y_pred = reg.predict(X_train), reg.predict(X_test)

    rmse_train, rmse_test = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_pred))

    print(model_name, f'\t - RMSE on Training  = {rmse_train:.2f} / RMSE on Test = {rmse_test:.2f}')

    

    return rmse_train, rmse_test
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

_, _ = get_rmse(rf, 'rondom forrest')



features = pd.DataFrame()

features["features"] = X_train.columns

features["coefficient"] = rf.feature_importances_



features.sort_values(by=["coefficient"], ascending=False, inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,5)

sns.barplot(data=features, x="coefficient", y="features");
gb = GradientBoostingRegressor(n_estimators=100).fit(X_train, y_train)

_, _ = get_rmse(gb, 'gb')



features = pd.DataFrame()

features["features"] = X_train.columns

features["coefficient"] = gb.feature_importances_



features.sort_values(by=["coefficient"], ascending=False, inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,5)

sns.barplot(data=features, x="coefficient", y="features");
xgb_reg = xgb.XGBRegressor(n_estimators=100).fit(X_train, y_train)

_, _ = get_rmse(xgb_reg, 'xgb_reg')



features = pd.DataFrame()

features["features"] = X_train.columns

features["coefficient"] = xgb_reg.feature_importances_



features.sort_values(by=["coefficient"], ascending=False, inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,5)

sns.barplot(data=features, x="coefficient", y="features");
lgbm_reg = lgbm.LGBMRegressor(n_estimators=100).fit(X_train, y_train)

_, _ = get_rmse(lgbm_reg, 'lgbm_reg')



features = pd.DataFrame()

features["features"] = X_train.columns

features["coefficient"] = lgbm_reg.feature_importances_



features.sort_values(by=["coefficient"], ascending=False, inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,5)

sns.barplot(data=features, x="coefficient", y="features");
def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
# list of all the basic models used at first

model_list = [

    LinearRegression(), Lasso(), Ridge(), ElasticNet(),

    RandomForestRegressor(), GradientBoostingRegressor(), ExtraTreesRegressor(),

    xgb.XGBRegressor(), lgbm.LGBMRegressor()

             ]



# creation of list of names and scores for the train / test

model_names = [str(m)[:str(m).index('(')] for m in model_list]

rmse_train, rmse_test = [], []



# fit and predict all models

for model, name in zip(model_list, model_names):

    model.fit(X_train, y_train)

    sc_train, sc_test = get_rmse(model, name)

    rmse_train.append(sc_train)

    rmse_test.append(sc_test)
from sklearn.preprocessing import PolynomialFeatures



poly_lin_reg = Pipeline([

    ("poly_feat", PolynomialFeatures(degree=3)),

    ("scaler", StandardScaler()),

    ("linear_reg", LinearRegression())

])



poly_lin_reg.fit(X_train, y_train)



sc_train, sc_test = get_rmse(poly_lin_reg, "Poly Linear Reg")



model_names.append('Poly Linear Reg')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
rd_cv = Ridge()

rd_params_ = {'max_iter':[1000, 2000, 3000],

                 'alpha':[0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]}



rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

rd_cv = GridSearchCV(rd_cv,

                  rd_params_,

                  scoring = rmsle_scorer,

                  cv=5)



rd_cv.fit(X_train, y_train)
sc_train, sc_test = get_rmse(rd_cv, "Ridge CV")



model_names.append('Ridge CV')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
la_cv = Lasso()



alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])

la_params = {'max_iter':[1000, 2000, 3000],'alpha':alpha}



la_cv = GridSearchCV(la_cv, la_params, scoring = rmsle_scorer, cv=5)

la_cv.fit(X_train, y_train).best_params_
sc_train, sc_test = get_rmse(la_cv, "Lasso CV")



model_names.append('Lasso CV')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
knn_reg = KNeighborsRegressor()

knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6]}



knn_reg = GridSearchCV(knn_reg, knn_params, scoring = rmsle_scorer, cv=5)

knn_reg.fit(X_train, y_train).best_params_
sc_train, sc_test = get_rmse(knn_reg, "kNN Reg")



model_names.append('kNN Reg')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
svm_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svr", LinearSVR())

])



svm_reg.fit(X_train, y_train)



sc_train, sc_test = get_rmse(svm_reg, "SVM Reg")



model_names.append('SVM Reg')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



X_train_, X_test_, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
import tensorflow as tf
def model_five_layers(input_dim):



    model = tf.keras.models.Sequential()



    # Add the first Dense layers of 100 units with the input dimension

    model.add(tf.keras.layers.Dense(100, input_dim=input_dim, activation='sigmoid'))



    # Add four more layers of decreasing units

    model.add(tf.keras.layers.Dense(100, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(100, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(100, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(100, activation='sigmoid'))



    # Add finally the output layer with one unit: the predicted result

    model.add(tf.keras.layers.Dense(1, activation='relu'))

    

    return model
model = model_five_layers(input_dim=X_train.shape[1])



# Compile the model with mean squared error (for regression)

model.compile(optimizer='SGD', loss='mean_squared_error')



# Now fit the model on XXX epoches with a batch size of XXX

# You can add the test/validation set into the fit: it will give insights on this dataset too

model.fit(X_train_, y_train, validation_data=(X_test_, y_test), epochs=200, batch_size=8)
y_train_pred, y_pred = model.predict(X_train_), model.predict(X_test_)

rmse_train_, rmse_test_ = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_pred))

print("MLP reg", f'\t - RMSE on Training  = {rmse_train_:.2f} / RMSE on Test = {rmse_test_:.2f}')



#sc_train, sc_test = get_rmse(model, "MLP reg")



model_names.append('MLP reg')

rmse_train.append(rmse_train_)

rmse_test.append(rmse_test_)
df_score = pd.DataFrame({'model_names' : model_names,

                         'rmse_train' : rmse_train,

                         'rmse_test' : rmse_test})

df_score = pd.melt(df_score, id_vars=['model_names'], value_vars=['rmse_train', 'rmse_test'])

df_score.head(10)
plt.figure(figsize=(12, 10))

sns.barplot(y="model_names", x="value", hue="variable", data=df_score)
y_sample = pd.read_csv("../input//bike-sharing-demand/sampleSubmission.csv")

y_sample.head()
df_test = pd.read_csv("../input/bike-sharing-demand/test.csv")

df_test = change_datetime(df_test)



# keep this col for the submission

datetimecol = df_test["datetime"]



test_cols_dropped = ['datetime',

 'atemp',

 'month_str',

 'season',

 'dow_str',

 'weather_str']



df_test = df_test.drop(columns=test_cols_dropped)

df_test.head()
lgbm_reg = lgbm.LGBMRegressor()

lgbm_reg.fit(X, y)

y_pred_final = lgbm_reg.predict(df_test)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in y_pred_final]

    })

submission.to_csv('bike_prediction_output.csv', index=False)



submission.head()