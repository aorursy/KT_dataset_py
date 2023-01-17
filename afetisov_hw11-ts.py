import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union, make_pipeline

%matplotlib inline
df_train = pd.read_csv("../input/train.csv", parse_dates=["Date"], date_parser=pd.to_datetime, low_memory=False)
df_test = pd.read_csv("../input/test.csv", parse_dates=["Date"], date_parser=pd.to_datetime, low_memory=False)
df_store = pd.read_csv("../input/store.csv")
train = df_train.merge(df_store)
test = df_test.merge(df_store)
df_train.head()
df_train.info()
df_test.info()
df_store.head()
df_store.info()
print("train")
print("max: ", df_train.Date.min())
print("min:", df_train.Date.max())
print("delta: ", df_train.Date.max() - df_train.Date.min())
print("test")
print("max: ", df_test.Date.min())
print("min:", df_test.Date.max())
print("delta: ", df_test.Date.max() - df_test.Date.min())
df_train.groupby("DayOfWeek").agg({"Sales": "mean"}).plot(kind="bar")
df_train[(df_train.DayOfWeek == 7) & (df_train.Open == 1)].Store.unique().shape[0]
df_train.groupby("StateHoliday").agg({"Sales": "mean"}).plot(kind="bar")
df_train.groupby("SchoolHoliday").agg({"Sales": "mean"}).plot(kind="bar")
df_train.groupby("Promo").agg({"Sales": "mean"}).plot(kind="bar")
train = df_train.merge(df_store)
test = df_test.merge(df_store)
train.PromoInterval.unique()
def get_Promo2Active(df):
    months_map = {v:i+1 for i, v in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                               "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])}
    def is_Promo2_active(row):
        if row.Promo2 == 0:
            return 0
        
        current_week, current_month, current_year = row.Date.week, row.Date.month, row.Date.year
        start_week, start_year = row.Promo2SinceWeek, row.Promo2SinceYear
        active_months = set([months_map[m] for m in row.PromoInterval.split(",")])
        has_started = (current_year == start_year and current_week >= start_week) or current_year > start_year
        return int(has_started and current_month in active_months)
                        
    return df.apply(is_Promo2_active, axis=1)
def get_CompetitionActive(df):
    def is_competition_active(row):
        if np.isnan(row.CompetitionDistance):
            return 0
        
        if np.isnan(row.CompetitionOpenSinceMonth) and np.isnan(row.CompetitionOpenSinceYear):
            return 1
        
        current_month, current_year = row.Date.month, row.Date.year
        opened_month, opened_year = row.CompetitionOpenSinceMonth, row.CompetitionOpenSinceYear
        
        return int((current_year == opened_year and current_month >= opened_month) or current_year > opened_year)
        
        
    return df.apply(is_competition_active, axis=1)
train["Promo2Active"] = get_Promo2Active(train)
train["CompetitionActive"] = get_CompetitionActive(train)
train[train.Promo2 == 1].groupby("Promo2Active").agg({"Sales": "mean"}).plot(kind="bar")
train.groupby("CompetitionActive").agg({"Sales": "mean"}).plot(kind="bar")
# we already have these, but for the sake of consistence, let's do it again
train = df_train.merge(df_store)
test = df_test.merge(df_store)

train["Promo2Active"] = get_Promo2Active(train)
train["CompetitionActive"] = get_CompetitionActive(train)

test["Promo2Active"] = get_Promo2Active(test)
test["CompetitionActive"] = get_CompetitionActive(test)
train["DayOfYear"] = train.Date.apply(lambda x: x.timetuple().tm_yday)
test["DayOfYear"] = test.Date.apply(lambda x: x.timetuple().tm_yday)
min_date = train.Date.min() # we know that all test data happened later
def date_to_day_number(df):
    return (df.Date - min_date).apply(lambda x: x.days)
train["Day"] = date_to_day_number(train)
test["Day"] = date_to_day_number(test)
train.sort_values("Day", inplace=True)
test.sort_values("Day", inplace=True)
def rmspe(y_true, y_pred):
    w = np.zeros(y_true.shape, dtype=float)
    ind = y_true != 0
    w[ind] = 1./ (y_true[ind]**2)
    return np.sqrt(np.mean(w * (y_true - y_pred)**2))
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
train_baseline = train.copy()
train_baseline['Last_Week_Sales'] = train_baseline.groupby("Store")["Sales"].shift()
train_baseline['Last_Week_Diff'] = train_baseline.groupby("Store")["Last_Week_Sales"].diff()
train_baseline.dropna(inplace=True, subset=["Last_Week_Sales", "Last_Week_Diff"])
train_baseline.head()
mean_error = []
for day in range(2, train_baseline.Day.max() + 1):
    val = train_baseline[train_baseline.Day == day]
    p = val.Last_Week_Sales.values
    error = rmspe(val.Sales.values, p)
    mean_error.append(error)
    
print('Mean Error = %.5f' % np.mean(mean_error))
class LabelEncoderPipelineFriendly(LabelEncoder):
    
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelEncoderPipelineFriendly, self).fit(X)
        
    def transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).transform(X).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X).reshape(-1, 1)
def prepare_pipeline(df):
    
    def get_DayOfWeek(df):
        return df["DayOfWeek"]
    
    def get_Open(df):
        return df[["Open"]]
    
    def get_Promo(df):
        return df[["Promo"]]
    
    def get_StateHoliday(df):
        return df["StateHoliday"]
    
    def get_SchoolHoliday(df):
        return df[["SchoolHoliday"]]
    
    def get_StoreType(df):
        return df["StoreType"]
    
    def get_Assortment(df):
        return df["Assortment"]
    
    def get_Promo2Active(df):
        return df[["Promo2Active"]]
    
    def get_CompetitionActive(df):
        return df[["CompetitionActive"]]
    
    def get_CompetitionDistance(df):
        return df[["CompetitionDistance"]]
    
    def get_DayOfYear(df):
        return df["DayOfYear"]
    
    p = make_union(*[
        make_pipeline(FunctionTransformer(get_DayOfWeek, validate=False), 
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Open, validate=False),
                      Imputer(strategy="most_frequent")),
        make_pipeline(FunctionTransformer(get_Promo, validate=False)),
        make_pipeline(FunctionTransformer(get_StateHoliday, validate=False), 
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_SchoolHoliday, validate=False)),
        make_pipeline(FunctionTransformer(get_StoreType, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Assortment, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Promo2Active, validate=False)),
        make_pipeline(FunctionTransformer(get_CompetitionActive, validate=False)),
        make_pipeline(FunctionTransformer(get_CompetitionDistance, validate=False),
                      Imputer(),
                      StandardScaler()),        
        make_pipeline(FunctionTransformer(get_DayOfYear, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder())
    ])
    
    return p
pipeline = prepare_pipeline(train)
x_train, y_train = pipeline.fit_transform(train), train.Sales
x_test = pipeline.transform(test)
params = {"boosting_type" : ["gbdt"],
          "learning_rate": [0.1],
          "n_estimators": [200],
          "objective": ["regression"],
          "reg_alpha": [1.0],# [0.0, 0.5, 1.0], # no time for an actual CV on kaggle
          "reg_lambda": [1.0],# [0.0, 0.5, 1.0],
          "random_state": [0],
          "n_jobs": [-1]
         }
gs = GridSearchCV(LGBMRegressor(), params, scoring=rmspe_scorer, cv=2, n_jobs=1)
gs.fit(x_train, y_train)
prediction = gs.predict(x_test)
pd.DataFrame({"Id": test.Id, "Sales": prediction}).to_csv("submission.csv", sep=",", index=False)
def prepare_pipeline_ts(df, min_shift, max_shift):
    
    def get_shifted_date(df, for_sales=False):
        return (df.Date.min() + pd.DateOffset(days_to_shift))
    
    def get_DayOfWeek(df):
        return df["DayOfWeek"]
    
    def get_Open(df):
        return df[["Open"]]
    
    def get_Promo(df):
        return df[["Promo"]]
    
    def get_StateHoliday(df):
        return df["StateHoliday"]
    
    def get_SchoolHoliday(df):
        return df[["SchoolHoliday"]]
    
    def get_StoreType(df):
        return df["StoreType"]
    
    def get_Assortment(df):
        return df["Assortment"]
    
    def get_Promo2Active(df):
        return df[["Promo2Active"]]
    
    def get_CompetitionActive(df):
        return df[["CompetitionActive"]]
    
    def get_CompetitionDistance(df):
        return df[["CompetitionDistance"]]
    
    def get_DayOfYear(df):
        return df["DayOfYear"]
    
    def get_previous_sales(df):
        sales = df[["Store", "Sales"]].copy()
        for day in range(min_shift, max_shift + 1):
            sales["Last-{}_Day_Sales".format(day)] = sales.groupby("Store")["Sales"].shift(day)
            sales["Last-{}_Day_Diff".format(day)] = sales.groupby("Store")["Last-{}_Day_Sales".format(day)].diff()
        
        return sales.drop(["Store", "Sales"], axis=1)
    
    p = make_union(*[
        make_pipeline(FunctionTransformer(get_DayOfWeek, validate=False), 
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Open, validate=False),
                      Imputer(strategy="most_frequent")),
        make_pipeline(FunctionTransformer(get_Promo, validate=False)),
        make_pipeline(FunctionTransformer(get_StateHoliday, validate=False), 
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_SchoolHoliday, validate=False)),
        make_pipeline(FunctionTransformer(get_StoreType, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Assortment, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_Promo2Active, validate=False)),
        make_pipeline(FunctionTransformer(get_CompetitionActive, validate=False)),
        make_pipeline(FunctionTransformer(get_CompetitionDistance, validate=False),
                      Imputer(),
                      StandardScaler()),        
        make_pipeline(FunctionTransformer(get_DayOfYear, validate=False),
                      LabelEncoderPipelineFriendly(), 
                      OneHotEncoder()),
        make_pipeline(FunctionTransformer(get_previous_sales, validate=False), 
                      Imputer(),
                      StandardScaler())
    ])
    
    return p
test_size = len(test)
min_shift = (test.Date.max() - test.Date.min()).days 
max_shift = 180
to_drop = len(train[train.Date < train.Date.min() + pd.DateOffset(max_shift)])
full = pd.concat([train, test], ignore_index=True) # we need to use full dataset to fill previous sales for test
pipeline_ts = prepare_pipeline_ts(train, min_shift, max_shift)
full_transformed = pipeline_ts.fit_transform(full)
x_train_ts, y_train_ts = full_transformed[to_drop:-test_size], train.Sales[to_drop:]
x_test_ts = full_transformed[-test_size:]
gs_ts = GridSearchCV(LGBMRegressor(), params, scoring=rmspe_scorer, cv=2, n_jobs=1)
gs_ts.fit(x_train_ts, y_train_ts)
prediction_ts = gs_ts.predict(x_test_ts)
pd.DataFrame({"Id": test.Id, "Sales": prediction_ts}).to_csv("submission_ts.csv", sep=",", index=False)