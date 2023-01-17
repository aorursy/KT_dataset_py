import pandas as pd 

data = pd.read_csv("../input/betscsv/bets.csv")

# The rows have duplicates because of home/away, so remove every other row
data = data[data.site == "home"]

# Adding points for both teams together
data["real_total"] = data["points"] + data["o:points"]

# Calculating the difference between the line and outcome 
data["error"] = data["real_total"] - data["total"]

# Making the error the absolute value
data["error"] = data.error.abs()

# Calculating mean absolute error between the lines and outcome 
n = len(data)
total_error = data.error.sum()
mae_lines = total_error / n

print ("MAE over last 10 years",mae_lines)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn import preprocessing 

# Read the data
X_full = pd.read_csv("../input/cleandata2csv/cleandata2.csv")
X_full["GAME_DATE_EST"] = pd.to_datetime(X_full["GAME_DATE_EST"],infer_datetime_format=True)
X_full = X_full.sort_values(by=["GAME_DATE_EST"], ascending=False)
X_full = X_full.reset_index()



# Label / One-Hot encoding team IDs and season
le = preprocessing.OneHotEncoder()
le2 = preprocessing.LabelEncoder()
ohe_home = le.fit_transform(X_full[["HOME_TEAM_ID"]]).toarray()
ohe_away = le.transform(X_full[["VISITOR_TEAM_ID"]]).toarray()
X_full["SEASON"] = X_full["SEASON"] - 2010

# Drop old columns
X_full = X_full.drop(columns =[
    "HOME_TEAM_ID","VISITOR_TEAM_ID", "PTS_home", "PTS_away","GAME_DATE_EST","index"], axis=1)
# Testing features:
X_full["diff"] = X_full["avgpointtotal_home"] - X_full["avgpointtotal_away"]
#X_full["diff"] = X_full["diff"]**2
#X_full["diff"] = X_full["diff"].abs()
#X_full["multi"] = X_full.apply(lambda row: row.avgpointtotal_home / row.avgpointtotal_away, axis=1)

#X_full["off_power_home"] = X_full["point_average_last10"] - X_full["away_point_againts_average_last10"]
#X_full["off_power_away"] = X_full["away_point_average_last10"] - X_full["point_againts_average_last10"]
#X_full["gap"] = X_full["off_power_home"].abs() + X_full["off_power_away"].abs()
X_full.head(5)
X_full.describe()
X_full.point_total.describe()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 12))
sns.heatmap(X_full.corr(),
            cmap="Blues",annot=True, fmt='.2f', vmin=0);
ohe_home_df = pd.DataFrame(ohe_home)
ohe_away_df = pd.DataFrame(ohe_away, columns=list("abcdefghijklmnopqrstuvwxyzABCD"))
X_full = pd.concat([X_full,ohe_home_df], axis=1)
X_full = pd.concat([X_full,ohe_away_df], axis=1)
print(X_full.shape)
X_full.head(5)
import matplotlib.pyplot as plt

with plt.style.context("ggplot"):
    plt.scatter(X_full.meanpointtotal, X_full.point_total, marker="o", alpha=0.1, color='#9467bd')
    plt.xlabel("Mean last 50 games")
    plt.ylabel("Total points (y)")
    plt.title("Total points (y) vs mean last 50 games ")
fig1 = plt.figure();
with plt.style.context("ggplot"):
    plt.scatter(X_full["diff"], X_full.point_total, marker="o", alpha=0.1)
    plt.xlabel("Difference")
    plt.ylabel("Total points (y)")
    plt.title("Total points (y) vs mean last 50 games ")
fig2 = plt.figure();
with plt.style.context("ggplot"):
    plt.scatter(X_full["diff"].abs(), X_full.point_total, marker="o", alpha=0.1)
    plt.xlabel("Difference")
    plt.ylabel("Total points (y)")
    plt.title("Total points (y) vs mean last 50 games ")
fig3 = plt.figure();
#mean_df = X_full.copy()
#mean_df["target"] = y
season_avg = X_full.groupby(by=["SEASON"]).mean()

with plt.style.context("ggplot"):
    plt.scatter(season_avg.index, season_avg.point_total, color = "#d62728" )
    plt.xlabel("Season")
    plt.ylabel("Total points (y)")
    plt.title("Mean Total Points Per Season")
fig4 = plt.figure();
y = X_full["point_total"]
X_full = X_full.drop(columns=['point_total'])

test = len(X_full) - int(len(X_full)*0.9)

X_train = X_full.iloc[test:]
X_valid = X_full.iloc[:test]
y_train = y.iloc[test:]
y_valid = y.iloc[:test]


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

EN = ElasticNet().fit(X_train,y_train)

pred_EN = EN.predict(X_valid)
mae_EN = mean_absolute_error(pred_EN, y_valid)
print ("mae",mae_EN)
from sklearn.linear_model import Lasso 

ls = Lasso().fit(X_train, y_train)

pred_ls = ls.predict(X_valid)
mae_ls = mean_absolute_error(pred_ls, y_valid)
print("mae",mae_ls)
from sklearn.linear_model import Ridge

rg = Ridge().fit(X_train, y_train)

pred_rg = rg.predict(X_valid)
mae_rg = mean_absolute_error(pred_rg, y_valid)
print("mae",mae_rg)
from sklearn.linear_model import TheilSenRegressor

ts = TheilSenRegressor().fit(X_train, y_train)
pred_ts = ts.predict(X_valid)
mae_ts = mean_absolute_error(pred_ts,y_valid)
print("mae",mae_ts)
#lin reg 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score

lin = LinearRegression(normalize=True, ).fit(
    X_train,y_train
)
pred_lin = lin.predict(X_valid)
mae_lin = mean_absolute_error(pred_lin, y_valid)
print("mae", mae_lin)
from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(random_state=1, max_iter=1000).fit(
    X_train, y_train
)

pred_mlp = regr.predict(X_valid)
mae_regr = mean_absolute_error(pred_mlp, y_valid)
print("mae",mae_regr)
import xgboost as xgb

xg = xgb.XGBRegressor(
    booster="gblinear",
    objective="reg:squarederror",
    base_score="1",
    eval_metric="cox-nloglik"
    ).fit(X_train,y_train)
pred_xg = xg.predict(X_valid)
xg_regr = mean_absolute_error(pred_xg, y_valid)
                         
print("mae",xg_regr)
print("done")
error = []
mean_last_season = season_avg.iloc[6,0]
for i in y_valid:
    error.append(abs(mean_last_season - i))
mae_mean = sum(error) / len(error)

print(mae_mean)
print(mean_last_season)
from hpsklearn import HyperoptEstimator, any_regressor
from hyperopt import tpe

estim = HyperoptEstimator(regressor=any_regressor("svr"))

estim.fit(X_train.values, y_train.values)

print(estim.best_model())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_testing = scaler.fit_transform(X_train)
X_testing_valid = scaler.transform(X_valid)
xg1 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
             colsample_bylevel=0.8886553695663921, colsample_bynode=1,
             colsample_bytree=0.6068338278675369, gamma=0.0017889282555567038,
             gpu_id=-1, importance_type='gain', interaction_constraints='',
             learning_rate=0.0013963222902296055, max_delta_step=0, max_depth=7,
             min_child_weight=19, monotone_constraints='()',
             n_estimators=5400, n_jobs=0, num_parallel_tree=1,
             objective='reg:linear', random_state=1,
             reg_alpha=0.22448614695919908, reg_lambda=2.9482948529431052,
             scale_pos_weight=1, seed=1, subsample=0.8571414026048771,
             tree_method='exact', validate_parameters=1, verbosity=None).fit(X_testing,y_train)
        
pred_xg1 = xg1.predict(X_testing_valid)
xg1_regr = mean_absolute_error(pred_xg1, y_valid)
print("mae",xg1_regr)
results = {"Model": ["ElasticNet","LinearRegression",
                     "MLPRegression","XGBoost",
                     "Lasso","Ridge","TheilSen",
                     "Betting lines","Guessing the mean",
                     "HyperoptEstimator"],
          "MAE Score": [mae_EN,mae_lin,
                       mae_regr,xg_regr,mae_ls,
                       mae_rg,mae_ts,mae_lines,
                       mae_mean,xg1_regr]}
result_df = pd.DataFrame(data=results)
result_df = result_df.sort_values(by=["MAE Score"])
print(result_df.to_string(index=False))
bttm = (1 - (xg1_regr / mae_mean))*100
wttl = (1 - (mae_lines / xg1_regr))*100
print("The model is {:.0f}% better than guessing the mean".format(bttm))
print("The model is {:.0f}% worse than the betting lines".format(wttl))
import numpy as np
with plt.style.context("ggplot"):
    plt.scatter(range(len(pred_xg1)), pred_xg1, marker="o", alpha=0.3,color = "#0066ff")
    plt.xlabel("Games")
    plt.ylabel("Estimated total points (y)")
    plt.title("Best Model Predictions")
fig6 = plt.figure();
with plt.style.context("ggplot"):
    plt.scatter(range(len(y_valid)), y_valid, marker="o", alpha=0.3, color="#0066ff")
    plt.xlabel("Games")
    plt.ylabel("Total points (y)")
    plt.title("Real Outcomes")
fig7 = plt.figure();

ls = []
for i in range(len(pred_ts)):
    ls.append(pred_xg1[i] - y_valid.iloc[i])
print ("Average Error of model:",sum(ls))
with plt.style.context("ggplot"):
    plt.scatter(range(len(ls)), ls, marker="o", alpha=0.3, color="#ff0000")
    plt.xlabel("Games")
    plt.ylabel("Prediction-Outcome Difference")
    plt.title("Error")
fig7 = plt.figure();