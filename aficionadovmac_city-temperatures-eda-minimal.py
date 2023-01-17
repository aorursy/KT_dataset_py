import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")
profile = ProfileReport(df, title="Pandas Profiling Report", minimal=True, progress_bar=False)
profile.to_notebook_iframe()