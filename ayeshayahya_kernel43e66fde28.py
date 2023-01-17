import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
vg_filepath="../input/videogamesales/vgsales.csv"
vg_data=pd.read_csv(vg_filepath,index_col="Rank")
sns.scatterplot(x=vg_data['Year'], y=vg_data['NA_Sales'])# hue=vg_data['Year'])

sns.lmplot(x="Year", y="NA_Sales", data=vg_data)