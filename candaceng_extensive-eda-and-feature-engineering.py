import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
sns.heatmap(df.corr())
sns.pairplot(df, hue='quality')
scaled_df = pd.DataFrame(StandardScaler().fit_transform(df.loc[:, df.columns != 'quality']))
scaled_df.columns = df.columns[:-1]
scaled_df.head()
fig = go.Figure() 
fig.add_trace(go.Box(x=df['fixed acidity'],name="Fixed Acidity"))
fig.show()

fig = go.Figure() 
fig.add_trace(go.Box(x=df['volatile acidity'],name="Volatile Acidity"))
fig.add_trace(go.Box(x=df['citric acid'],name="Citric Acid"))
fig.show()
adjusted_df = df[df['volatile acidity'] < 1.5]
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['chlorides'],name="Chlorides"))
fig.add_trace(go.Box(x=scaled_df['sulphates'],name="Sulphates"))
fig.show()
adjusted_df = df[df['sulphates'] < 7]
adjusted_df = df[df['chlorides'] < 6]
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['free sulfur dioxide'],name="Free Sulfur Dioxide"))
fig.add_trace(go.Box(x=scaled_df['total sulfur dioxide'],name="Total Sulfur Dioxide"))
fig.show()
adjusted_df = df[df['total sulfur dioxide'] < 6]
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['residual sugar'],name="Residual Sugar"))
fig.show()
adjusted_df = df[df['residual sugar'] < 7]
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['density'],name="Density"))
fig.show()
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['pH'],name="pH"))
fig.show()
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['alcohol'],name="Alcohol"))
fig.show()
fig = go.Figure() 
fig.add_trace(go.Box(x=scaled_df['fixed acidity'],name="Fixed Acidity"))
fig.add_trace(go.Box(x=scaled_df['volatile acidity'],name="Volatile Acidity"))
fig.add_trace(go.Box(x=scaled_df['citric acid'],name="Citric Acid"))
fig.add_trace(go.Box(x=scaled_df['residual sugar'],name="Residual Sugar"))
fig.add_trace(go.Box(x=scaled_df['chlorides'],name="Chlorides"))
fig.add_trace(go.Box(x=scaled_df['free sulfur dioxide'],name="Free Sulfur Dioxide"))
fig.add_trace(go.Box(x=scaled_df['total sulfur dioxide'],name="Total Sulfur Dioxide"))
fig.add_trace(go.Box(x=scaled_df['density'],name="Density"))
fig.add_trace(go.Box(x=scaled_df['pH'],name="pH"))
fig.add_trace(go.Box(x=scaled_df['sulphates'],name="Sulphates"))
fig.add_trace(go.Box(x=scaled_df['alcohol'],name="Alcohol"))
fig.update_layout(title="Summary of Wine Quality Features")
fig.show()
scaled_df = pd.DataFrame(StandardScaler().fit_transform(adjusted_df.loc[:, adjusted_df.columns != 'quality']))
scaled_df.columns = df.columns[:-1]
sns.pairplot(scaled_df)