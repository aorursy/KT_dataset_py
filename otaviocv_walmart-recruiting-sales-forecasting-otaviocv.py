import os

import itertools

from tqdm.notebook import tqdm

import pandas as pd

import numpy as np



from sklearn.metrics import mean_absolute_error



from sklearn.linear_model import ElasticNet

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid

import xgboost as xgb

import catboost as cat



import shap



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data_path = '/kaggle/input/walmart-recruiting-store-sales-forecasting/'
!ls {data_path}
for dirname, _, filenames in os.walk(data_path):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pwd
stores = pd.read_csv(data_path + "stores.csv")

features = pd.read_csv(data_path + "features.csv.zip")

train_data = pd.read_csv(data_path + "train.csv.zip")

test_data = pd.read_csv(data_path + "test.csv.zip")

sample_submission = pd.read_csv(data_path + "sampleSubmission.csv.zip")
features.head()
train_data.head()
test_data.head()
holidays = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08", "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-13",

                           "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29", "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"])

holidays_dict = {

    "2010-02-12": "Super Bowl",

    "2011-02-11": "Super Bowl",

    "2012-02-10": "Super Bowl",

    "2013-02-08": "Super Bowl",

    "2010-09-10": "Labor Day",

    "2011-09-09": "Labor Day",

    "2012-09-07": "Labor Day",

    "2013-09-13": "Labor Day",

    "2010-11-26": "Thanksgiving",

    "2011-11-25": "Thanksgiving",

    "2012-11-23": "Thanksgiving",

    "2013-11-29": "Thanksgiving",

    "2010-12-31": "Christmas",

    "2011-12-30": "Christmas",

    "2012-12-28": "Christmas",

    "2013-12-27": "Christmas"

}
y_true = np.array([1, 2, 3, 4, 5, 6])

y_pred = np.array([2 ,3, 0, 5, 6, 1]) # 1 + 1 + 3 + 1 + 1 + 5 = 12

                                      # 1 + 1 + 15 + 1 + 1 + 25 = 44

weights = np.array([1, 1, 5, 1, 1, 5]) # 1 + 1 + 5 + 1 + 1 + 5 = 14



assert mean_absolute_error(y_true, y_pred) == 2

assert mean_absolute_error(y_true, y_pred, sample_weight=weights) == 44/14
f, ax = plt.subplots(figsize=(5,1))



def plot_holidays(holidays, ax=plt, color="red", linestyle="--", linewidth="0.5", **kwargs):

    for hday in holidays:

        ax.axvline(x=hday, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

        

plot_holidays(holidays, ax=ax)
stores.head()
stores.dtypes
stores.shape
stores.Type.value_counts(normalize=True, dropna=False)
f, ax = plt.subplots(figsize=(8, 3), dpi=140)

sns.distplot(stores.Size);
stores.Size.describe()
f, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

sns.heatmap(features.sort_values("Store").isna(), ax=ax[0], cmap="plasma", cbar=None, yticklabels=False);

ax[0].set_title("NANs sorted by store")

sns.heatmap(features.sort_values("Date").isna(), ax=ax[1], cmap="plasma", yticklabels=False);

ax[1].set_title("NANs sorted by date");
features.isna().mean()
features.head()
train_data.Date = pd.to_datetime(train_data.Date)

test_data.Date = pd.to_datetime(test_data.Date)

features.Date = pd.to_datetime(features.Date)

features_dates = features.groupby("Date", as_index=False).agg({"Store": "nunique"})

#features_dates.Date = pd.to_datetime(features_dates.Date)

features_dates.plot(x="Date", y="Store");
print("Min: ", features_dates.Date.min(), "Max: ", features_dates.Date.max())
f, ax = plt.subplots(figsize=(12, 4), dpi=150)



for year in range(features_dates.Date.min().year, features_dates.Date.max().year+1):

    plot_data = features[features.Date.dt.year == year].groupby("Date", as_index=False).agg({"Temperature": "mean"})

    ax.plot(plot_data.Date.dt.dayofyear, plot_data.Temperature, label=year)



ax.legend()

ax.set_title("Temperature over Time")

ax.set_ylabel("Temperature [°F]")

ax.set_xlabel("Day of year");
f, ax = plt.subplots(figsize=(12, 4), dpi=150)



for year in range(features_dates.Date.min().year, features_dates.Date.max().year +1):

    plot_data = features[features.Date.dt.year == year].groupby("Date", as_index=False).agg({"Unemployment": "mean"})

    ax.plot(plot_data.Date.dt.dayofyear, plot_data.Unemployment, label=year)



ax.legend()

ax.set_title("Unemployment over Time")

ax.set_ylabel("Unemployment")

ax.set_xlabel("Day of year");
features.groupby(["Date", "Store"], as_index=False).agg({"Unemployment": "nunique"}).Unemployment.value_counts()
f, ax = plt.subplots(figsize=(12, 4), dpi=150)



for year in range(features_dates.Date.min().year, features_dates.Date.max().year+1):

    plot_data = features[features.Date.dt.year == year].groupby("Date", as_index=False).agg({"Fuel_Price": "mean"})

    ax.plot(plot_data.Date.dt.dayofyear, plot_data.Fuel_Price, label=year)



ax.legend()

ax.set_title("Fuel Price over Time")

ax.set_ylabel("Fuel Price [USD]") # I'm assuming US Dollars

ax.set_xlabel("Day of year");
f, ax = plt.subplots(figsize=(12, 4), dpi=150)



for year in range(features_dates.Date.min().year, features_dates.Date.max().year+1):

    plot_data = features[features.Date.dt.year == year].groupby("Date", as_index=False).agg({"CPI": "mean"})

    ax.plot(plot_data.Date.dt.dayofyear, plot_data.CPI, label=year)

    ax.axhline(y=plot_data.CPI.max(), color="grey", linestyle="--", linewidth=0.2)



ax.legend()

ax.set_title("Consumer Price Index over Time")

ax.set_ylabel("Consumer Price Index")

ax.set_xlabel("Day of year");
f, ax = plt.subplots(figsize=(12, 4), dpi=150)



for i in range(1, 6):

    plot_data = features.groupby("Date", as_index=False).agg({"MarkDown" + str(i): "mean"})

    ax.plot(plot_data.Date, plot_data["MarkDown" + str(i)], label="MarkDown" + str(i))

    #ax.axhline(y=plot_data.CPI.max(), color="grey", linestyle="--", linewidth=0.2)



ax.legend()

ax.set_title("Markdowns over Time")

ax.set_ylabel("Markdowns")

ax.set_xlabel("Day of year");

plot_holidays(holidays[holidays > pd.to_datetime("2011-11-01")], ax)
markdown_corr = np.zeros((5,5))

markdown_cols = ["MarkDown" + str(i) for i in range(1, 6)]

nan_mask = features.loc[:, markdown_cols].isnull().sum(axis=1) > 0

filtered_markdowns = features.loc[~nan_mask, markdown_cols]
sns.heatmap(filtered_markdowns.corr());
f, ax = plt.subplots(figsize=(12, 4), dpi=130)





scatter_data = train_data.groupby(["Date", "Dept"], as_index=False).Weekly_Sales.agg("median")

palette = sns.color_palette("hls", scatter_data.Dept.nunique())

color_pallete = {d: palette[i] for i, d in enumerate(scatter_data.Dept.unique())}

xjitter = np.random.randint(0, 5, size=len(scatter_data)).astype("timedelta64[D]") # points are falling in the same spot, let's given them a little jitter no more than 5 days

color = scatter_data.Dept.apply(lambda x: color_pallete.get(x))

ax.scatter(scatter_data.Date + xjitter, scatter_data.Weekly_Sales, s=0.3, color=color);

ax.set_ylim(0,80000);

ax.set_facecolor('black')
bla = sns.palplot(palette)
train_data.groupby("Store").agg({"Dept": "nunique"}).sort_values("Dept", ascending=False).describe()
f ,ax = plt.subplots(figsize=(12, 2), dpi=180)

train_data.groupby("Store").agg({"Dept": "nunique"}).sort_values("Dept", ascending=False).plot.bar(ax=ax);
dept_fill_matrix = train_data.pivot_table(index="Store", columns="Dept", values="Date", aggfunc="count")

dept_fill_matrix_mask = dept_fill_matrix > 0
f, ax = plt.subplots(figsize=(12, 6), dpi=160)

sns.heatmap(dept_fill_matrix_mask);
year_sales = train_data.groupby(["Store", train_data.Date.dt.year]).agg({"Weekly_Sales": "sum"}).reset_index()

year_sales = year_sales.merge(stores, on="Store", how="left")
f, axs = plt.subplots(3, 1, figsize=(10, 15), dpi=120)

color_pallete = sns.color_palette("plasma", year_sales.Store.nunique())



for i, year in enumerate(range(features_dates.Date.min().year, features_dates.Date.max().year)):

    ax = axs[i]

    data = year_sales[year_sales.Date == year]

    colors = data.Store.apply(lambda s: color_pallete[s-1])

    ax.scatter(data.Size, data.Weekly_Sales, color=colors)

    ax.set_title(year)
dept_sales_matrix = train_data.pivot_table(index="Store", columns="Dept", values="Weekly_Sales", aggfunc="mean")
columns_sort = dept_sales_matrix.sum(axis=0).sort_values(ascending=False).index.values

rows_sort = dept_sales_matrix.sum(axis=1).sort_values(ascending=False).index.values
f, ax = plt.subplots(figsize=(12, 6), dpi=160)

sns.heatmap(dept_sales_matrix.loc[rows_sort, columns_sort], ax=ax, cmap="plasma");
dept_sales_over_time = train_data.groupby(["Date", "Dept"]).agg({"Weekly_Sales": "sum"}).reset_index()

dept_sales_avg = dept_sales_over_time.groupby("Dept", as_index=False).agg(dept_mean=pd.NamedAgg(column='Weekly_Sales', aggfunc='mean'))

dept_sales_over_time = dept_sales_over_time.merge(dept_sales_avg, on="Dept")

dept_sales_over_time["variance"] = (dept_sales_over_time["Weekly_Sales"] - dept_sales_over_time["dept_mean"])

dept_sales_over_time["proportional_variance"] = dept_sales_over_time["variance"]/dept_sales_over_time["dept_mean"]
dept_sales_over_time
f, ax = plt.subplots(3, 1, figsize=(12, 9), dpi=160)

dept_sales_over_time.groupby("Dept").plot(x="Date", y="Weekly_Sales", ax=ax[0]);

ax[0].set_title("Absolute Values")

ax[0].legend([]);

plot_holidays(holidays, ax[0])



dept_sales_over_time.groupby("Dept").plot(x="Date", y="variance", ax=ax[1]);

ax[1].set_title("Absolute Variance")

ax[1].legend([]);

plot_holidays(holidays, ax[1])



dept_sales_over_time.groupby("Dept").plot(x="Date", y="proportional_variance", ax=ax[2]);

ax[2].set_title("Proportional Variance")

ax[2].legend([]);

ax[2].set_ylim([-20, 20])

plot_holidays(holidays, ax[2])
plt.figure(figsize=(14,5), dpi=120)

sns.boxplot(train_data["Store"], train_data["Weekly_Sales"],showfliers=False);
plt.figure(figsize=(14,5), dpi=120)

sns.boxplot(train_data["Dept"], train_data["Weekly_Sales"],showfliers=False);
all_features = features.merge(stores, on="Store")

all_features["HolidayType"] = all_features.Date.apply(lambda x: holidays_dict.get(x.date().isoformat(), "Not Holiday"))

all_features["Year"] = all_features.Date.dt.year

all_features["Month"] = all_features.Date.dt.month

all_features["Day"] = all_features.Date.dt.day

all_features["DayOfWeek"] = all_features.Date.dt.dayofweek

all_features["WeekOfYear"] = all_features.Date.dt.isocalendar().week

all_features["DayOfYear"] = all_features.Date.dt.dayofyear
full_train_data = train_data.drop("IsHoliday", axis=1).merge(all_features, on=["Store", "Date"])

full_test_data = test_data.drop("IsHoliday", axis=1).merge(all_features, on=["Store", "Date"])
test_ids = full_test_data.Store.astype(str) + '_' + full_test_data.Dept.astype(str) + '_' + full_test_data.Date.astype(str)
full_train_data.head()
full_test_data.head()
print("Min: ", full_train_data.Date.min(), "Max: ", full_train_data.Date.max())
target = "Weekly_Sales"

continuous_features = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",

                       "CPI", "Unemployment", "Size", "Year", "Day", "Month", "WeekOfYear", "DayOfYear"]

categorical_features = ["Store", "Dept", "IsHoliday", "Type", "HolidayType", "DayOfWeek"]

features = continuous_features + categorical_features

drop_features = ["Date"]
split_date = (full_train_data.Date.min() + (full_train_data.Date.max() - full_train_data.Date.min()) * 0.8)

split_date
optimization_data = full_train_data[full_train_data.Date < split_date]

optimization_target = optimization_data[target]

optimization_features = optimization_data[features]

validation_data = full_train_data[full_train_data.Date >= split_date]

validation_target = validation_data[target]

validation_features = validation_data[features]
print(f"Optimization proportion: {len(optimization_data)/len(full_train_data)}")

print(f"Validation proportion: {len(validation_data)/len(full_train_data)}")
print("Optimization shapes: ", optimization_data.shape, optimization_features.shape, optimization_target.shape)

print("Validation shapes: ", validation_data.shape, validation_features.shape, validation_target.shape)
f, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=120)

sns.distplot(optimization_target, label="Optimization")

sns.distplot(validation_target, label="Validation")

ax.legend()
optimization_target.describe()
validation_target.describe()
pred_mean = optimization_target.mean()

pred_mean
y_mean_pred_train = np.zeros(len(optimization_data))

y_mean_pred_train[:] = pred_mean



y_mean_pred = np.zeros(len(validation_data))

y_mean_pred[:] = pred_mean
optimization_weights = optimization_data["IsHoliday"].apply(lambda x: 5 if x else 1).values

weights = validation_data["IsHoliday"].apply(lambda x: 5 if x else 1).values

weights
optimization_performance_dummy_1 = mean_absolute_error(optimization_target, y_mean_pred_train, sample_weight=optimization_weights)

optimization_performance_dummy_1
validation_performance_dummy_1 = mean_absolute_error(validation_target, y_mean_pred, sample_weight=weights)

validation_performance_dummy_1
stores_and_deps_mean = optimization_data.groupby(["Store", "Dept"], as_index=False).agg({"Weekly_Sales": "mean"})

stores_and_deps_mean
stores_and_dept_preds_train = optimization_data.loc[:, ["Store", "Dept"]].merge(stores_and_deps_mean, on=["Store", "Dept"], how="left").Weekly_Sales

stores_and_dept_preds_train[stores_and_dept_preds_train.isna()] = pred_mean



stores_and_dept_preds = validation_data.loc[:, ["Store", "Dept"]].merge(stores_and_deps_mean, on=["Store", "Dept"], how="left").Weekly_Sales

stores_and_dept_preds[stores_and_dept_preds.isna()] = pred_mean
optimization_performance_dummy_2 = mean_absolute_error(optimization_target, stores_and_dept_preds_train, sample_weight=optimization_weights)

optimization_performance_dummy_2
validation_performance_dummy_2 = mean_absolute_error(validation_target, stores_and_dept_preds, sample_weight=weights)

validation_performance_dummy_2
mean_absolute_error(validation_target, stores_and_dept_preds)
stores_and_dept_preds_test = test_data.loc[:, ["Store", "Dept", "Date"]].merge(stores_and_deps_mean, on=["Store", "Dept"], how="left")

stores_and_dept_preds_test.loc[stores_and_dept_preds_test.Weekly_Sales.isnull(), "Weekly_Sales"] = pred_mean

test_ids = stores_and_dept_preds_test.Store.astype(str) + '_' + stores_and_dept_preds_test.Dept.astype(str) + '_' + stores_and_dept_preds_test.Date.astype(str)

sample_submission['Id'] = test_ids.values

sample_submission['Weekly_Sales'] = stores_and_dept_preds_test.Weekly_Sales.values

sample_submission.to_csv('submission_dummy_2.csv',index=False)
print("Contnuous features:", continuous_features)

print("Categorical features:", categorical_features)
optimization_data.loc[:, categorical_features].dtypes
linear_preprocessor = ColumnTransformer([

    

    ('scaled_continous',

     Pipeline([

         ('imputer', SimpleImputer()), # This is not strictly necessary for well behaved features but it will serve as a general protection under bad data.

         ('scaler', StandardScaler())

     ]),

    ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Year', 'Day', 'Month', 'WeekOfYear', 'DayOfYear']

    ),

    

    ('markdowns',

     Pipeline([

         ('imputer', SimpleImputer(strategy="constant", fill_value=0)), # Since markdowns has a lot of missing values and change a lot over time I will simply fill it with zeros

         ('scaler', StandardScaler())

     ]),

     ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    ),

    

    ("categorical",

     Pipeline([

         ("one_hot", OneHotEncoder(handle_unknown='ignore'))

     ]),

     (['Store', 'Dept', 'Type', 'HolidayType', 'DayOfWeek'])

    ),

    

    ("others",

     "passthrough",

     ['IsHoliday'] # IsHoliday is not actually categorical, it isn't necessary to put it in a OneHotEncoder

    )

    

])
linear_preprocessor.fit_transform(optimization_features)
linear_estimator = ElasticNet()
linear_model = Pipeline([

    ('preprocessor', linear_preprocessor),

    ('estimator', linear_estimator)

])
linear_hyperparameters = {

    "estimator__l1_ratio": [0.2, 0.5, 0.8, 1],

    "estimator__alpha": [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]

}
kf = KFold(5)

splits = kf.split(optimization_features)
linear_optimizer = GridSearchCV(

    linear_model,

    linear_hyperparameters,

    scoring="neg_mean_absolute_error",

    cv=5,

    n_jobs=4,

    return_train_score=True,

    verbose=10

)
linear_optimizer.fit(optimization_features, optimization_target)
linear_optimizer.best_params_
pd.DataFrame(linear_optimizer.cv_results_).sort_values(by="rank_test_score").loc[:, ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score"]].head(8)
y_linear_pred_train = linear_optimizer.predict(optimization_features)

y_linear_pred = linear_optimizer.predict(validation_features)
optimization_performance_linear = mean_absolute_error(optimization_target, y_linear_pred_train, sample_weight=optimization_weights)

optimization_performance_linear
validation_performance_linear = mean_absolute_error(validation_target, y_linear_pred, sample_weight=weights)

validation_performance_linear
mean_absolute_error(optimization_target, y_linear_pred_train)
mean_absolute_error(validation_target, y_linear_pred)
test_ids = full_test_data.Store.astype(str) + '_' + full_test_data.Dept.astype(str) + '_' + full_test_data.Date.astype(str)

sample_submission["Id"] = test_ids

y_linear_test = linear_optimizer.predict(full_test_data.loc[:, features])

sample_submission["Weekly_Sales"] = y_linear_test

sample_submission.to_csv('submission_linear.csv',index=False)
best_model = linear_optimizer.best_estimator_
linear_preprocessed_feature_names = []

for transformation, transformer, columns in best_model.named_steps["preprocessor"].transformers_:

    if transformer == "passthrough":

        linear_preprocessed_feature_names += columns

        continue

    last_transformer = transformer.steps[-1]

    if last_transformer[0] == "scaler":

        linear_preprocessed_feature_names += columns

    if last_transformer[0] == "one_hot":

        categories = last_transformer[1].categories_

        for column, category in zip(columns, categories):

            linear_preprocessed_feature_names += [column + "_" + str(i) for i in category]
len(linear_preprocessed_feature_names)
optimization_features_linear_preprocessed = best_model.named_steps["preprocessor"].transform(optimization_features).toarray()

validation_features_linear_preprocessed = best_model.named_steps["preprocessor"].transform(validation_features).toarray()

linear_explainer = shap.LinearExplainer(best_model.named_steps["estimator"],

                                        optimization_features_linear_preprocessed,

                                        feature_perturbation="interventional")

linear_shap_values = linear_explainer.shap_values(validation_features_linear_preprocessed)
shap.summary_plot(linear_shap_values, validation_features_linear_preprocessed, linear_preprocessed_feature_names)
columns_sort
linear_model_weights_df = pd.DataFrame({"column": linear_preprocessed_feature_names, "weights": best_model.named_steps["estimator"].coef_}).sort_values("weights", ascending=False).reset_index(drop=True)

linear_model_weights_df.head()
f, ax = plt.subplots(figsize=(15, 3), dpi=200)

linear_model_weights_df[~(linear_model_weights_df["weights"].abs() < 100)].plot.bar(x="column", y="weights", ax=ax)
xgb_preprocessor = ColumnTransformer([

    

    ('labeler',

     OrdinalEncoder(),

    ['WeekOfYear', 'Type', 'HolidayType']

    ),

    

    ("others",

     "passthrough",

     ['Temperature','Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',

      'CPI', 'Unemployment', 'Size', 'Year', 'Day', 'Month', 'DayOfYear', 'Store',

      'Dept', 'IsHoliday', 'DayOfWeek']

    )

    

])



xgb_preprocessed_feature_names = ['WeekOfYear', 'Type', 'HolidayType'] + ['Temperature','Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',

      'CPI', 'Unemployment', 'Size', 'Year', 'Day', 'Month', 'DayOfYear', 'Store',

      'Dept', 'IsHoliday', 'DayOfWeek']
xgb_preprocessed_feature_names
optimization_features_xgb_preprocessed = xgb_preprocessor.fit_transform(optimization_features)

validation_features_xgb_preprocessed = xgb_preprocessor.transform(validation_features)
dmatrix_optimization = xgb.DMatrix(optimization_features_xgb_preprocessed, optimization_target)

dmatrix_validation = xgb.DMatrix(validation_features_xgb_preprocessed, validation_target)
dmatrix_optimization.num_col()
base_xgb_params = {

    "objective": "reg:squarederror",

    "booster": "gbtree",

    "subsample": 0.5,

    "colsample_bytree": 0.8,

    "colsample_bylevel": 0.95,

    "num_parallel_tree": 3, # Forests for the Win! This parameter controls the number of simultaneous tree trained at each boosting round.

    "eval_metric": "mae"

}



xgb_hyperparameters = {

    "max_depth": [3, 5, 10],

    "reg_alpha": [1e1, 1e3, 1e5],

    "reg_lambda": [1e2, 1e3, 1e4, 1e5], # let's choose some heavy regularization parameters

}



parameter_grid = [dict(**base_xgb_params, **i) for i in list(ParameterGrid(xgb_hyperparameters))]
results = []

for params in tqdm(parameter_grid[:]):

    print(params)

    xgb_cv_results = xgb.cv(params, dmatrix_optimization, num_boost_round=20, nfold=5, early_stopping_rounds=1, verbose_eval=False)

    rounds = len(xgb_cv_results)

    performance_values = xgb_cv_results.iloc[-1, :].to_dict()

    print(performance_values)

    result_dict = dict(**params, **performance_values, rounds=rounds, params=params)

    results.append(result_dict)
results_df = pd.DataFrame(results).sort_values("test-mae-mean")

best_model = results_df.iloc[0, :]

best_params = best_model["params"]



results_df
best_params
xgb_estimator = xgb.train(best_params, dmatrix_optimization, num_boost_round=best_model["rounds"], verbose_eval=True)
y_xgb_preds_train = xgb_estimator.predict(dmatrix_optimization)

y_xgb_preds = xgb_estimator.predict(dmatrix_validation)
optimization_performance_xgboost = mean_absolute_error(optimization_target, y_xgb_preds_train, sample_weight=optimization_weights)

optimization_performance_xgboost
validation_performance_xgboost = mean_absolute_error(validation_target, y_xgb_preds, sample_weight=weights)

validation_performance_xgboost
mean_absolute_error(optimization_target, y_xgb_preds_train)
mean_absolute_error(validation_target, y_xgb_preds)
test_ids = full_test_data.Store.astype(str) + '_' + full_test_data.Dept.astype(str) + '_' + full_test_data.Date.astype(str)

sample_submission['Id'] = test_ids.values

y_xgb_test = xgb_estimator.predict(xgb.DMatrix(xgb_preprocessor.transform(full_test_data.loc[:, features])))

sample_submission['Weekly_Sales'] = y_xgb_test

sample_submission.to_csv('submission_xgboost.csv',index=False)
model_bytearray = xgb_estimator.save_raw()[4:]

def myfun(self=None):

    return model_bytearray



xgb_estimator.save_raw = myfun
explainer = shap.TreeExplainer(xgb_estimator, feature_perturbation='tree_path_dependent')

shap_values = explainer.shap_values(validation_features_xgb_preprocessed, check_additivity=False) # The additivty test is failing for less than 1e3 of difference.
shap.summary_plot(shap_values, validation_features_xgb_preprocessed, xgb_preprocessed_feature_names)
shap.dependence_plot("Dept", shap_values, validation_features_xgb_preprocessed, xgb_preprocessed_feature_names)
features
reduced_features = ['Size', 'Store', 'Dept',]

cat_cat_feaures = ['Store', 'Dept']
cat_estimator = cat.CatBoostRegressor(cat_features=cat_cat_feaures)
cat_estimator.fit(optimization_features.loc[:, reduced_features], optimization_target)
cat_estimator.tree_count_
y_cat_preds_train = cat_estimator.predict(optimization_features.loc[:, reduced_features])

y_cat_preds = cat_estimator.predict(validation_features.loc[:, reduced_features])
optimization_performance_cat = mean_absolute_error(optimization_target, y_cat_preds_train, sample_weight=optimization_weights)

optimization_performance_cat
validation_performance_cat = mean_absolute_error(validation_target, y_cat_preds, sample_weight=weights)

validation_performance_cat
mean_absolute_error(validation_target, y_cat_preds)
mean_absolute_error(optimization_target, y_cat_preds_train)
test_ids = full_test_data.Store.astype(str) + '_' + full_test_data.Dept.astype(str) + '_' + full_test_data.Date.astype(str)

sample_submission['Id'] = test_ids.values

y_cat_test = cat_estimator.predict(full_test_data.loc[:, reduced_features])

sample_submission['Weekly_Sales'] = y_cat_test

sample_submission.to_csv('submission_cat.csv',index=False)
performances = pd.DataFrame(

    [[optimization_performance_dummy_1, optimization_performance_dummy_2, optimization_performance_linear, optimization_performance_xgboost, optimization_performance_cat],

     [validation_performance_dummy_1, validation_performance_dummy_2, validation_performance_linear, validation_performance_xgboost, validation_performance_cat]],

    columns = ["Dummy Model 1", "Dummy Model 2", "Linear Model", "XGBoost", "CatBoost"],

    index = ["Optimization", "Validation"]

)

performances
store_candidate = stores.sample(1).iloc[0, 0]

store_candidate



optimization_preds_df = pd.DataFrame({

    "Store": optimization_data.Store.values,

    "Dept": optimization_data.Dept.values,

    "Date": optimization_data.Date.values,

    "Weekly_Sales": optimization_data.Weekly_Sales.values,

    "dummy_1": y_mean_pred_train,

    "dummy_2": stores_and_dept_preds_train,

    "linear": y_linear_pred_train,

    "xgboost": y_xgb_preds_train,

    "catboost": y_cat_preds_train,

})



validation_preds_df = pd.DataFrame({

    "Store": validation_data.Store.values,

    "Dept": validation_data.Dept.values,

    "Date": validation_data.Date.values,

    "Weekly_Sales": validation_data.Weekly_Sales.values,

    "dummy_1": y_mean_pred,

    "dummy_2": stores_and_dept_preds,

    "linear": y_linear_pred,

    "xgboost": y_xgb_preds,

    "catboost": y_cat_preds,

})



test_preds_df = pd.DataFrame({

    "Store": full_test_data.Store.values,

    "Dept": full_test_data.Dept.values,

    "Date": full_test_data.Date.values,

    "dummy_2": stores_and_dept_preds_test.Weekly_Sales.values,

    "linear": y_linear_test,

    "xgboost": y_xgb_test,

    "catboost": y_cat_test,

})



summary_optimization = optimization_preds_df[optimization_preds_df.Store == store_candidate].groupby(["Date"], as_index=False).sum()

summary_validation = validation_preds_df[validation_preds_df.Store == store_candidate].groupby(["Date"], as_index=False).sum()

summary_test = test_preds_df[test_preds_df.Store == store_candidate].groupby(["Date"], as_index=False).sum()



plots = ["Weekly_Sales", "dummy_1", "dummy_2", "linear", "xgboost", "catboost"]

colors = {

    "Weekly_Sales": "red",

    "dummy_1": "green",

    "dummy_2": "orange",

    "linear": "brown",

    "xgboost": "blue",

    "catboost": "cyan"

}



linestyle = {

    "Weekly_Sales": "-."

}



f, ax = plt.subplots(figsize=(12, 4), dpi=130)

for p in plots:

    kwargs = {

        "label": p,

        "color": colors.get(p),

        "linewidth": 0.8,

        "linestyle": linestyle.get(p, "-")

    }

    ax.plot(summary_optimization.Date, summary_optimization[p], **kwargs)

    ax.plot(summary_validation.Date, summary_validation[p], **kwargs)

    if p != "Weekly_Sales" and p != "dummy_1":

        ax.plot(summary_test.Date, summary_test[p], **kwargs)



ax.axvspan(optimization_data.Date.min(), optimization_data.Date.max(), color='grey',alpha=0.14)

ax.axvspan(validation_data.Date.min(), validation_data.Date.max(), color='green',alpha=0.14)

ax.axvspan(test_preds_df.Date.min(), test_preds_df.Date.max(), color='red',alpha=0.14)

ax.set_title("Store " + str(store_candidate))

ax.legend(loc="upper right")

plot_holidays(holidays, ax)