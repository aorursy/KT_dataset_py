import pandas as pd
from pandas_profiling import ProfileReport
df=pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv", encoding='UTF-8')
df.head()
profile = ProfileReport(df, title="Pandas Profiling Report", progress_bar=False)
profile