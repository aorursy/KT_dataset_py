!pip install fastai2 --quiet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from fastai2.basics import *
from fastai2.tabular.all import *

import catboost
from sklearn.ensemble import GradientBoostingRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv", parse_dates=["Date"])
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv", parse_dates=["Date"])
def fill_nulls(df):
    df["County"].fillna("<Unk>", inplace=True)
    df["Province_State"].fillna("<Unk>", inplace=True)
    
fill_nulls(df_train)
fill_nulls(df_test)
df_train_cases = df_train[df_train.Target == "ConfirmedCases"].drop(columns=["Target"])
df_train_fatalities = df_train[df_train.Target == "Fatalities"].drop(columns=["Target"])

df_test_cases = df_test[df_test.Target == "ConfirmedCases"].drop(columns=["Target"])
df_test_fatalities = df_test[df_test.Target == "Fatalities"].drop(columns=["Target"])
add_datepart(df_train_cases, 'Date', drop=False)
add_datepart(df_train_fatalities, 'Date', drop=False)

add_datepart(df_test_cases, 'Date', drop=False)
add_datepart(df_test_fatalities, 'Date', drop=False);
# Avoiding the leak will be done in the Predictor class

#df_train_cases = df_train_cases[df_train_cases.Date < "2020-04-27"]
#df_train_fatalities = df_train_fatalities[df_train_fatalities.Date < "2020-04-27"]
# FROM https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target[:, 0].unsqueeze(1) - preds[:, i].unsqueeze(1)

            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
            
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
def pinball(preds, target):
    assert preds.size(0) == target.size(0)
    target_vals = target[:, 0]
    target_weights = target[:, 1]
    
    losses = []

    for i, q in enumerate(quantiles):
        errors = (target_vals - preds[:, i]) * target_weights
        losses.append(
            torch.max(
               (q-1) * errors, 
               q * errors
            ).unsqueeze(1)
        )

    return torch.mean(
        torch.mean(torch.cat(losses, dim=1), dim=1)
    )
MAX_TRAIN_DATE = "2020-04-27"
quantiles = [0.95, 0.5, 0.05]
cont_vars = ["Population", 'Elapsed']
cat_vars = ["County", "Province_State", "Country_Region",
            "Month", "Week", "Day", "Dayofweek", "Dayofyear", 
            'Is_month_end', 'Is_month_start', 'Is_quarter_end',
           'Is_quarter_start', 'Is_year_end', 'Is_year_start']
class Predictor():
    def __init__(self, train_df, test_df):
        self.dep_var = ["TargetValue", "Weight"]
        self.train_df = self.train_df_processed(train_df)
        
        self.MAX_TRAIN_IDX = self.train_df[self.train_df['Date'] < MAX_TRAIN_DATE].shape[0]
        
        self.df_wrapper = self.prepare_df_wrapper(self.train_df)
        self.dls = self.df_wrapper.dataloaders(bs=500, path='/kaggle/working/')
        self.dls.c = len(quantiles) # Number of outputs of the network
        
        self.learner = tabular_learner(self.dls, 
                                      layers=[1000, 500],
                                      opt_func=ranger, 
                                      loss_func=QuantileLoss(quantiles),
                                      metrics=[pinball])
        
        self.test_dls = self.prepare_test_dl(test_df)
       
    
    def train_df_processed(self, train_df):
        df = train_df[cont_vars + cat_vars + self.dep_var + ['Date']].copy().sort_values('Date')
        df = df[df.TargetValue >= 0]
        df["TargetValue"] = np.log1p(df.TargetValue)
        return df
    
    
    def prepare_df_wrapper(self, train_df_processed):
        procs=[FillMissing, Categorify, Normalize]

        splits = list(range(self.MAX_TRAIN_IDX)), list(range(self.MAX_TRAIN_IDX, len(train_df_processed)))

        return TabularPandas(train_df_processed, 
                             procs,
                             cat_vars.copy(), 
                             cont_vars.copy(), 
                             self.dep_var,
                             y_block=TransformBlock(), 
                             splits=splits, )
    

    def prepare_test_dl(self, test_df_raw):
        to_tst = self.df_wrapper.new(test_df_raw)
        to_tst.process()
        return self.dls.valid.new(to_tst)
    
    
    def predict(self) -> np.ndarray:
        tst_preds, _ = self.learner.get_preds(dl=self.test_dls)
        tst_preds = tst_preds.data.numpy()
        return np.expm1(tst_preds)
    
    
    def lc(self):
        emb_szs = get_emb_sz(self.df_wrapper)
        print(emb_szs)
        self.dls.show_batch()
        self.test_dls.show_batch()
model_cases = Predictor(df_train_cases, df_test_cases)
model_fatalities = Predictor(df_train_fatalities, df_test_fatalities)
#model_cases.learner.lr_find()
model_cases.learner.fit_one_cycle(10, 0.05)
model_cases.learner.recorder.plot_loss()
#model_fatalities.learner.lr_find()
model_fatalities.learner.fit_one_cycle(5, lr_max=0.001)
model_fatalities.learner.recorder.plot_loss()
pred_cases = model_cases.predict()
pred_fatalities = model_fatalities.predict()
def format_preds(preds=pred_cases, df_test=df_test_cases):
    forecast_ids = []
    for index in df_test["ForecastId"]:
        for quantile in quantiles:
            forecast_ids.append(f"{index}_{quantile}")

    return pd.Series(data=preds.reshape(-1), index=forecast_ids)

cases_preds      = format_preds(pred_cases, df_test_cases)
fatalities_preds = format_preds(pred_fatalities, df_test_fatalities)
submission = pd.concat([cases_preds, fatalities_preds])
submission = pd.DataFrame(submission, columns=["TargetValue"])
submission = submission.reset_index().rename(columns={'index': 'ForecastId_Quantile'})
submission.to_csv("submission.csv", index=False)
!head submission.csv

