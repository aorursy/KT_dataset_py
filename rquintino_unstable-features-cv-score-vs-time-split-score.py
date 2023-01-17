import numpy as np

import pandas as pd

import os



pd.options.display.max_rows = 100
df_cv_score=pd.read_csv("../input/classification-auc-per-single-feature/cv_feature_results.csv").groupby("feature")["cv_score"].mean().reset_index().sort_values("cv_score",ascending=False)

df_timesplit_score=pd.read_csv("../input/classification-auc-per-single-feature-time-split/time_split_feature_results.csv")

df_scores=pd.merge(df_cv_score,df_timesplit_score).drop(columns="index")

df_scores["time_split_vs_cv"]=df_scores.time_split_score-df_scores.cv_score

df_scores.sort_values("time_split_vs_cv",ascending=True)