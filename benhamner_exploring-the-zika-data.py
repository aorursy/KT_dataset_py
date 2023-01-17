import pandas as pd

import seaborn as sns

sns.set_style("whitegrid")



zika = pd.read_csv("../input/cdc_zika.csv")



zika.groupby("location").size().reset_index().rename(columns={0: "count"})