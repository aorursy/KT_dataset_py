import h2o
from h2o.automl import H2OAutoML
h2o.init()
import pandas as pd
import numpy as np

train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_df = h2o.H2OFrame(train_df)
test_df = h2o.H2OFrame(test_df)
y = "SalePrice"
splits = train_df.split_frame(ratios = [0.75], seed = 1)
train = splits[0]
test = splits[1]
aml = H2OAutoML(max_runtime_secs = 0, seed = 1, project_name = "housePredLeaderboardDF")
aml.train(y = y, training_frame = train, leaderboard_frame = test)
aml.leaderboard.head()
pred = aml.predict(test_df)
colsCombine_df = test_df['Id'].cbind(pred)
colsCombine_df['SalePrice'] = colsCombine_df['predict']
final_pred = colsCombine_df[:, ["Id", "SalePrice"]]


h2o.export_file(final_pred, 'house_sales_price_pred_full.csv')
perf = aml.leader.model_performance(train_df)
perf
