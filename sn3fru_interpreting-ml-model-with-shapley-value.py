import pandas as pd

import xgboost as xgb

import shap

import pandas as pd

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams["figure.figsize"] = (12,8)
df = pd.read_csv('/kaggle/input/house-sales2/house_sales.csv')
Xfeats=['is_waterfront', 'latitude', 'longitude', 'num_bed', 'num_floors', 'size_house', 'condition']

target=['price']
sns.heatmap(df[Xfeats+target].corr(), annot=True)
sns.heatmap(df)
def plotCorrGraph(df):

  # Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.

  corr = df.corr().abs()

  

  # Transform it in a links data frame (3 columns only):

  links = corr.stack().reset_index()

  links.columns = ['var1', 'var2','value']

  

  # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)

  links_filtered=links.loc[(links['var1'] != links['var2'])]



  temp = []

  checkvalues= []

  for row in links_filtered.values:

      if row[2]>.2 and row[2] not in checkvalues:

          temp.append(row)

          checkvalues.append(row[2])

        

  links_filtered = pd.DataFrame(temp, columns=['var1', 'var2','value'])



  # Build your graph

  G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', )

  

  # Plot the network:

  nx.draw(G, with_labels=True, font_size=10, alpha=.75, node_color='#A0CBE2',edge_color=links_filtered.value.values, width=4, edge_cmap=plt.cm.Blues)

#   plt.figure(figsize=(12,8))

  plt.show()



plotCorrGraph(df)
import statsmodels.formula.api as smf

function = ''' price ~ is_waterfront + latitude + longitude + num_bed + num_floors + size_house + condition'''



model = smf.ols(function, df).fit()

print(model.summary())
from sklearn import preprocessing



temp = df.values

min_max_scaler = preprocessing.MinMaxScaler()

temp = min_max_scaler.fit_transform(df)

temp = pd.DataFrame(temp, columns = df.columns)

temp.describe()
function = ''' price ~ is_waterfront + latitude + longitude + num_bed + num_floors + size_house + condition'''



model = smf.ols(function, temp).fit()

print(model.summary())
X = df[Xfeats]

y = df[['price']]
X.describe()
model = xgb.XGBRegressor(max_depth=10)

model.fit(X, y)
plt.rcParams["figure.figsize"] = (8,6)

for f in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:

    xgb.plot_importance(model, importance_type=f)
shap.initjs()

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X)
X.head(1)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[1,:], X.iloc[1,:])
shap.force_plot(explainer.expected_value, shap_values[2,:], X.iloc[2,:])
shap.force_plot(explainer.expected_value, shap_values[3,:], X.iloc[3,:])

shap.force_plot(explainer.expected_value, shap_values[:100], X.iloc[:100])
plt.rcParams["figure.figsize"] = (16,9)



for col in list(X):

    shap.dependence_plot(col, shap_values, X)

    plt.show()
import mplleaflet as mpll



points = np.arange(df.shape[0])

np.random.shuffle(points)

f, ax = plt.subplots(1, figsize=(12, 8))

df.iloc[points[:500], :].plot(kind='scatter', x='longitude',

                             y='latitude',s=30, linewidth=0, ax=ax);

mpll.display(fig=f,)
model = xgb.XGBRegressor(max_depth=1)

model.fit(X, y)



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)



plt.rcParams["figure.figsize"] = (16,9)



for col in list(X):

    shap.dependence_plot(col, shap_values, X)

    plt.show()