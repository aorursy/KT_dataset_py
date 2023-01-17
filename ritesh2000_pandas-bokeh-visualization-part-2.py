import numpy as np
import pandas as pd
import pandas_bokeh
import seaborn as sns
import matplotlib.pyplot as plt
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource
df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
df.head()
data_table = DataTable( columns=[TableColumn(field=Ci, title=Ci) for Ci in df.columns],source=ColumnDataSource(df),height=300) # Setting Up Data Table

scatter = df.plot_bokeh.scatter(x="petal_length", y="sepal_width", category="species", title="Iris Dataset Visualization", show_figure=True) # Scatter Plot

pandas_bokeh.plot_grid([[data_table, scatter]], plot_width=400, plot_height=350)# Scatter Plot + Data Table Visuals
df.plot_bokeh.bar(xlabel="petal_length",ylabel="sepal_width",alpha=0.6,figsize=(2000,800),title="petal_length Vs Sepal_length")
df.plot_bokeh.bar(xlabel="petal_length",ylabel="sepal_width",alpha=0.6,figsize=(2000,800),title="petal_length Vs Sepal_length",stacked=True)
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv",index_col='SalePrice')# Loading Dataset.....
df.head()
df.describe()
numeric_features = df.select_dtypes(include=[np.number])

p_bar = numeric_features.plot_bokeh.bar(ylabel="Sale Price", figsize=(1000,800),title="Housing Prices", alpha=0.6)# Ploting the Bar Plot
df=pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")
df.head()
plt.figure(figsize=(16,8))
sns.scatterplot(x='longitude', y='latitude', data=df)
plt.grid()
plt.show()