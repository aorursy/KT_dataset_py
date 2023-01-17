import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
SUB_PATH = "/kaggle/input/covid19-global-forecasting-week-5/submission.csv"
TRAIN_PATH = "/kaggle/input/covid19-global-forecasting-week-5/train.csv"
TEST_PATH = "/kaggle/input/covid19-global-forecasting-week-5/test.csv"
train = pd.read_csv(TRAIN_PATH)
train.head()
train.describe()
train.info()
train[(train["Target"].str.contains("Fatalities")) & (train["TargetValue"] < 0)]
START_DAY = pd.to_datetime(train["Date"]).min()

def preprocess(df, drop_cols):
    df["Region"] = df["Country_Region"].str.cat(others=[df["Province_State"], df["County"]], sep=" ", na_rep="")
    df["Region"] = df["Region"].str.strip()
    df["Day"] = (pd.to_datetime(df["Date"]) - START_DAY).dt.days
    df["Target"] = df["Target"].str.contains("ConfirmedCases").astype(int)
    return df.drop(drop_cols, axis=1)

train = preprocess(train, ["Date", "County", "Province_State", "Country_Region"])

assert train[train["Day"] < 0].count().all() == 0
    
train.head()
import random
random.seed(1234)

all_regions = [r for r in train["Region"].unique() if "US" not in r]
regions_to_plot = random.sample(all_regions, 12)

confirmed_cases = train[train["Target"] & train["Region"].isin(regions_to_plot)]
fatalities = train[train["Target"] & train["Region"].isin(regions_to_plot)]

sns.set(rc={'figure.figsize':(24,10)})

f, (ax1, ax2) = plt.subplots(1, 2)
sns.lineplot(x="Day", y="TargetValue", hue="Region", data=confirmed_cases, ax=ax1).set(ylabel="ConfirmedCases", title="ConfirmedCases in different countries");
sns.lineplot(x="Day", y="TargetValue", hue="Region", data=fatalities, ax=ax2).set(ylabel="Fatalities", title="Fatalities in different countries");
def plot_outliers(df):
    all_regions = [r for r in df["Region"].unique() if "US" not in r]
    confirmed_cases = df[df["Target"] & (df["Region"].isin(all_regions))]
    fatalities = df[~df["Target"] & (df["Region"].isin(all_regions))]

    most_confirmed_cases = confirmed_cases.groupby("Region").sum()["TargetValue"].nlargest(10).index
    least_confirmed_cases = confirmed_cases.groupby("Region").sum()["TargetValue"].nsmallest(10).index
    most_fatalities = fatalities.groupby("Region").sum()["TargetValue"].nlargest(10).index
    least_fatalities = fatalities.groupby("Region").sum()["TargetValue"].nsmallest(10).index

    sns.set(rc={'figure.figsize':(24,24)})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sns.lineplot(x="Day", y="TargetValue", hue="Region", data=confirmed_cases[confirmed_cases["Region"].isin(most_confirmed_cases)], ax=ax1).set(ylabel="ConfirmedCases", title="Countries with most confirmed cases");
    sns.lineplot(x="Day", y="TargetValue", hue="Region", data=confirmed_cases[confirmed_cases["Region"].isin(least_confirmed_cases)], ax=ax3).set(ylabel="ConfirmedCases", title="Countries with least confirmed cases");
    
    sns.lineplot(x="Day", y="TargetValue", hue="Region", data=fatalities[fatalities["Region"].isin(most_fatalities)], ax=ax2).set(ylabel="Fatalities", title="Countries with most fatalities");
    sns.lineplot(x="Day", y="TargetValue", hue="Region", data=fatalities[fatalities["Region"].isin(least_fatalities)], ax=ax4).set(ylabel="Fatalities", title="Countries with least fatalities");
    
plot_outliers(train)
def remove_zeros(df):
    df["TargetValue"][df["TargetValue"] < 0] = 0
    return df
    
train = remove_zeros(train)
train.describe()
plot_outliers(train)
train["Day"].value_counts().unique()
sns.set(rc={'figure.figsize':(20,8)})
f, (ax1, ax2) = plt.subplots(1, 2)


cases = train[train["Target"] == 1].drop(["Target"], axis=1)
cases["Region"] = cases["Region"].astype("category").cat.codes
fatalities = train[train["Target"] == 0].drop(["Target"], axis=1)
fatalities["Region"] = fatalities["Region"].astype("category").cat.codes

corr = cases.corr(method="spearman")
sns.heatmap(corr, vmin=-1, vmax=1, cmap="YlGnBu", ax=ax1).set(title="Spearman correlation for cases");

corr = fatalities.corr(method="spearman")
plot = sns.heatmap(corr, vmin=-1, vmax=1, cmap="YlGnBu", ax=ax2).set(title="Spearman correlation for fatalities");
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SUB_PATH)
test.head()
test = preprocess(test, ["County", "Province_State", "Country_Region", "Date"])
test.head()
print(f"Time span for train set: {train['Day'].min()}-{train['Day'].max()}")
print(f"Time span for test set: {test['Day'].min()}-{test['Day'].max()}")
sub.head()
from sklearn.model_selection import train_test_split

X = train.drop(['TargetValue', 'Id', 'Region'], axis=1)
Y = train["TargetValue"]

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, train_size=0.8, random_state=123)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression().fit(train_X, train_Y)
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def evaluate(model, x, y):
    pred = model.predict(x)
    
    combined_mse = mse(y, pred)
    combined_mae = mae(y, pred)
    
    pred_cases = pred[x["Target"] == 1]
    pred_fatalities = pred[x["Target"] == 0]
    true_cases = y[x["Target"] == 1]
    true_fatalities = y[x["Target"] == 0]
    
    cases_mse = mse(true_cases, pred_cases)
    fatalities_mse = mse(true_fatalities, pred_fatalities)
    cases_mae = mae(true_cases, pred_cases)
    fatalities_mae = mae(true_fatalities, pred_fatalities)
    
    print(f"Combined\t\tMSE = {combined_mse},\tMAE = {combined_mae},\tmean true value: {y.mean()}")
    print(f"Cases\t\t\tMSE = {cases_mse},\tMAE = {cases_mae},\tmean true value: {true_cases.mean()}")
    print(f"Fatalities\t\tMSE = {fatalities_mse},\tMAE = {fatalities_mae},\tmean true value: {true_fatalities.mean()}")

evaluate(linear_model, val_X, val_Y)
from sklearn.preprocessing import PolynomialFeatures
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

def mapper(population, weight, day):
    mapper = DataFrameMapper([
        (['Population'], PolynomialFeatures(population, include_bias=False)),
        (['Weight'], PolynomialFeatures(weight, include_bias=False)),
        (['Day'], PolynomialFeatures(day, include_bias=False)),
        ('Target', None)])
    return mapper

for day in [1, 2, 3]:
    for weight in [0, 1, 2, 3]:
        for population in [1, 2, 3]:
            print(f"Evaluation for polynomial coefficients: day={day}, weight={weight}, population={population}")
            pipeline = Pipeline([('mapper', mapper(population, weight, day)), ('linear_regression', LinearRegression())])
            pipeline.fit(train_X, train_Y);
            evaluate(pipeline, val_X, val_Y)
            print()
from sklearn.ensemble import RandomForestRegressor

rfr_model = RandomForestRegressor(n_estimators=10, verbose=1, n_jobs=-1)
rfr_model.fit(train_X, train_Y);
rfr_model.verbose=0
evaluate(rfr_model, val_X, val_Y)
from sklearn.preprocessing import StandardScaler

def map_scale():
    mapper = DataFrameMapper([
        (['Population'], StandardScaler()),
        (['Weight'], StandardScaler()),
        (['Day'], StandardScaler()),
        ('Target', None)])
    return mapper

rbf_pipe_scale = Pipeline([('scale', map_scale()), 
                           ('regressor', RandomForestRegressor(n_estimators=10, verbose=1, n_jobs=-1))]);
rbf_pipe_scale.fit(train_X, train_Y);
rbf_pipe_scale.steps[1][1].verbose=0
evaluate(rbf_pipe_scale, val_X, val_Y)
from sklearn.model_selection import GridSearchCV

""" This is commented out as it takes a lot of time """

# param_grid = {
#     "n_estimators": [5, 10, 15, 20]
# }

# grid_search = GridSearchCV(RandomForestRegressor(), param_grid, verbose=10, n_jobs=-1);
# grid_search.fit(X, Y)
# rfr_model = grid_search.best_estimator_

# print(f"Best parameter (CV score={grid_search.best_score_})")
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
# evaluate(rfr_model, train_X, train_Y)
def plot_train_test(model, train, test, country):
    pd.set_option('mode.chained_assignment', None)
    train = train[train["Region"].str.match(country)];
    test = test[test["Region"].str.match(country)];
    
    test["TargetValue"] = model.predict(test.drop(["ForecastId", "Region"], axis=1));
    
    train_cases = train[train["Target"] == 1]
    train_fatalities = train[train["Target"] == 0]
    test_cases = test[test["Target"] == 1]
    test_fatalities = test[test["Target"] == 0]
    
    sns.set(rc={'figure.figsize':(24,12)})
    f, (ax1, ax2) = plt.subplots(1, 2)
    sns.lineplot(x="Day", y="TargetValue", data=train_cases, ax=ax1, label="True values");
    sns.lineplot(x="Day", y="TargetValue", data=test_cases, ax=ax1, label="Predicted values").set(ylabel="ConfirmedCases", title=f"Real and predicted cases in {country}");
    
    sns.lineplot(x="Day", y="TargetValue", data=train_fatalities, ax=ax2, label="True values");
    sns.lineplot(x="Day", y="TargetValue", data=test_fatalities, ax=ax2, label="Predicted values").set(ylabel="Fatalities", title=f"Real and predicted fatalities in {country}");
    pd.set_option('mode.chained_assignment', 'warn')
plot_train_test(rfr_model, train, test, "Russia")
plot_train_test(rfr_model, train, test, "Brazil")
plot_train_test(rfr_model, train, test, "Poland")
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

s_pop = MinMaxScaler(feature_range=(0, 1))
train["Population"] = s_pop.fit_transform(train["Population"].values[:,np.newaxis])

cases_data = train[train["Target"] == 1]
fatalities_data = train[train["Target"] == 0]

s_cas = MinMaxScaler(feature_range=(0, 1))
s_fat = MinMaxScaler(feature_range=(0, 1))

cases_data["TargetValue"] = s_cas.fit_transform(cases_data["TargetValue"].values[:,np.newaxis])
fatalities_data["TargetValue"] = s_fat.fit_transform(fatalities_data["TargetValue"].values[:,np.newaxis])

def split_x_y(df, time_span):
    df = df.drop(["Id", "Target", "Region"], axis=1)
    end_day = df["Day"].max()+1
    df = df.drop(["Day"], axis=1)
    df = df.values.reshape((-1, end_day, 3))

    x = []
    y = []
    
    for i in tqdm(range(df.shape[0])):
        for j in range(time_span, end_day):
            idx = range(j-time_span, j)
            x.append(df[i,idx,:])
            y.append(df[i,j,2])
    
    return np.array(x), np.array(y)

c_x, c_y = split_x_y(cases_data, 50)
f_x, f_y = split_x_y(fatalities_data, 50)
import keras
import keras.backend as K
from keras import Model
from keras.layers import *

def swish(x):
    return x * K.sigmoid(x)

def get_model(input_shape):
    inp = Input(input_shape)
    x = Bidirectional(GRU(16, return_sequences=True))(inp)
    x = Activation(swish)(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(16, return_sequences=True))(x)
    x = Activation(swish)(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(16, return_sequences=False))(x)
    x = Activation(swish)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation=swish)(x)

    model = Model(inp, [out])
    model.compile(loss=['mean_squared_error'],
              optimizer='RMSprop',
              metrics=['mean_squared_error'])
    print(model.summary())
    return model
    
cases_model = get_model((c_x.shape[1],c_x.shape[2]))
cases_model.fit(c_x, c_y, epochs=100, batch_size=1024, shuffle=True, validation_split=0.2, callbacks=[
    keras.callbacks.ModelCheckpoint("model_cases.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'), 
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)]);
def evaluate_separate(model, X, Y, scaler):
    pred = model.predict(X, batch_size=1024, verbose=1)
    pred = scaler.inverse_transform(pred)
    Y = scaler.inverse_transform(Y[:,np.newaxis])
    print(f"MSE: {((Y-pred)**2).mean()}\tMAE: {(np.abs(Y-pred)).mean()}\tmean Y value: {Y.mean()}")

cases_model.load_weights("model_cases.h5")
evaluate_separate(cases_model, c_x, c_y, s_cas)
fatalities_model = get_model((f_x.shape[1],f_x.shape[2]))

fatalities_model.fit(f_x, f_y, epochs=100, batch_size=1024, shuffle=True, validation_split=0.2, callbacks=[
    keras.callbacks.ModelCheckpoint("model_fatalities.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'), 
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)]);
fatalities_model.load_weights("model_fatalities.h5")
evaluate_separate(fatalities_model, f_x, f_y, s_fat)
cases_preds = cases_model.predict(c_x, batch_size=1024, verbose=1)
fatalities_preds = fatalities_model.predict(f_x, batch_size=1024, verbose=1)
def plot_nn(y, preds, predict_last_n, time_step, regions, scaler, ax, region_idx):    
    time_span = int(y.shape[0] / regions)
    y = scaler.inverse_transform(y[:,np.newaxis])
    y = y.reshape(regions, time_span)
    y = y[region_idx, :]
    y_days = list(range(time_step, time_step+time_span))
    
    preds = scaler.inverse_transform(preds)
    preds = preds.reshape(regions, time_span)
    preds = preds[region_idx, -predict_last_n:]
    preds_days = list(range(time_step+time_span-predict_last_n, time_step+time_span))

    sns.lineplot(x=y_days, y=y, ax=ax, label="True values");
    return sns.lineplot(x=preds_days, y=preds, ax=ax, label="Predicted values")
    
last_n = train["Day"].max() - test["Day"].min()

sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_nn(c_y, cases_preds, last_n, 50, len(train["Region"].unique()), s_cas, ax1, 154).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_nn(f_y, fatalities_preds, last_n, 50, len(train["Region"].unique()), s_fat, ax2, 154).set(ylabel="Fatalities", title=f"Fatalities");
sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_nn(c_y, cases_preds, last_n, 50, len(train["Region"].unique()), s_cas, ax1, 141).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_nn(f_y, fatalities_preds, last_n, 50, len(train["Region"].unique()), s_fat, ax2, 141).set(ylabel="Fatalities", title=f"Fatalities");
sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_nn(c_y, cases_preds, last_n, 50, len(train["Region"].unique()), s_cas, ax1, 120).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_nn(f_y, fatalities_preds, last_n, 50, len(train["Region"].unique()), s_fat, ax2, 120).set(ylabel="Fatalities", title=f"Fatalities");
def predict_unknown(model, x, time_span, regions, n_times):
    x = x.reshape((regions, -1, time_span, 3))
    x = x[:, -1:, :, :]
    x = x.reshape(-1, time_span, 3)
    
    preds = []
    for i in range(n_times):
        y = model.predict(x, batch_size=1024, verbose=1)
        preds.append(y[:,0])
        temp = np.zeros(x.shape)
        temp[:,:-1,:] = x[:,1:,:]
        temp[:,-1,:] = x[:,-1,:]
        temp[:,-1,2] = y[:,0]
        x = temp
    return np.array(preds)

days_to_predict = test["Day"].max() - train["Day"].max() + 1
output_cases = predict_unknown(cases_model, c_x, 50, len(train["Region"].unique()), days_to_predict)
output_fatalities = predict_unknown(fatalities_model, f_x, 50, len(train["Region"].unique()), days_to_predict)
output_cases = s_cas.inverse_transform(np.swapaxes(output_cases,0,1))
output_fatalities = s_fat.inverse_transform(np.swapaxes(output_fatalities,0,1))
train_set_cases_preds = s_cas.inverse_transform(np.reshape(cases_preds, (output_cases.shape[0], -1)))
train_set_fatalities_preds = s_fat.inverse_transform(np.reshape(fatalities_preds, (output_fatalities.shape[0], -1)))

final_cases = np.concatenate((train_set_cases_preds, output_cases), axis=1)
final_fatalities = np.concatenate((train_set_fatalities_preds, output_fatalities), axis=1)
cases_y = s_cas.inverse_transform(c_y[:,np.newaxis])
fatalities_y = s_fat.inverse_transform(f_y[:,np.newaxis])
cases_y = np.reshape(cases_y, (final_cases.shape[0],-1))
fatalities_y = np.reshape(fatalities_y, (final_fatalities.shape[0],-1))
def plot_final(true, pred, true_days, pred_days, ax, batch_idx): 
    true = true[batch_idx,:]
    pred = pred[batch_idx,:]
    sns.lineplot(x=true_days, y=true, ax=ax, label="True values");
    return sns.lineplot(x=pred_days, y=pred, ax=ax, label="Predicted values")
    
true_days = list(range(49, train["Day"].max()))
pred_days = list(range(test["Day"].max() - final_cases.shape[1], test["Day"].max()))
sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_final(cases_y, final_cases, true_days, pred_days, ax1, 154).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_final(fatalities_y, final_fatalities, true_days, pred_days, ax2, 154).set(ylabel="Fatalities", title=f"Fatalities");
sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_final(cases_y, final_cases, true_days, pred_days, ax1, 141).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_final(fatalities_y, final_fatalities, true_days, pred_days, ax2, 141).set(ylabel="Fatalities", title=f"Fatalities");
sns.set(rc={'figure.figsize':(24,12)})
f, (ax1, ax2) = plt.subplots(1, 2)
cases_plot = plot_final(cases_y, final_cases, true_days, pred_days, ax1, 120).set(ylabel="ConfirmedCases", title=f"Real and predicted cases");
fatalities_plot = plot_final(fatalities_y, final_fatalities, true_days, pred_days, ax2, 120).set(ylabel="Fatalities", title=f"Fatalities");
