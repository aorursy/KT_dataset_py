import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.tools as tls
import plotly.figure_factory as ff
sns.set(style="whitegrid")
data=pd.read_csv('../input/student.csv')
data
data.shape
data.info()
print ("\nVariables : \n" ,data.columns.tolist())
X1 = data['IQ']
X2 = data['Study hrs.']
Y = data['Marks'] 
data.describe()
hist_data = [data['IQ'], data['Study hrs.'],Y]

group_labels = ['IQ', 'Study Hrs','Marks']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=5,curve_type='normal',
                         show_hist = True, show_curve = True)
fig['layout'].update(title='Distribution Plot ')
fig.show()
data.drop(["Student"], axis = 1, inplace = True)
fig = ff.create_annotated_heatmap(data.corr().values.tolist(),
                                   
                                  y=data.columns.tolist(),
                                  x=data.columns.tolist(), 
                                  colorscale='Inferno',
                                  showscale=True
                                 )
fig.show()
import seaborn as sns
sns.set()
sns.pairplot(data,size = 2.5, kind = "reg",corner=True)
data.drop(["GPA"], axis = 1, inplace = True)
fig = ff.create_annotated_heatmap(data.corr().values.tolist(),
                                   
                                  y=data.columns.tolist(),
                                  x=data.columns.tolist(), 
                                  colorscale='Viridis',
                                  showscale=True
                                 )
fig.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor


t = data[['IQ', 'Study hrs.']]
t['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["Variables"] = t.columns
vif["VIF"] = [variance_inflation_factor(t.values, i) for i in range(t.shape[1])]

# View results using print
print(vif)
# y=b0+b1x1

x=sm.add_constant(X2)

results=sm.OLS(data['IQ'],x).fit()
#Contain Ordinary Least Square Regression 
results.summary()
# data["e"] = data["Study hrs."]/30 , ##error bar =  ,error_y='e'

fig = px.scatter(data, x="Study hrs.", y="IQ",title='IQ VS Study in Hrs',trendline="ols" , color= "IQ")

fig.show()
plt.figure(figsize=(13,7))
plt.title('Residual Plot for Marks and Study Hrs',size=20)
sns.residplot(X2, X1, lowess=True, color="r")
plt.ylabel('Residuals',size=15)
# y=b0+b1x1

x=sm.add_constant(X1)

results=sm.OLS(Y,x).fit()
#Contain Ordinary Least Square Regression 
results.summary()
fig = px.scatter(data, x="IQ", y="Marks", title= 'MARKS vs IQ',trendline="ols",color="Marks")

fig.show()
plt.figure(figsize=(13,7))
plt.title('Residual Plot for Marks and IQ',size=20)
sns.residplot(X1, Y, lowess=True, color="orange")
plt.ylabel('Residuals',size=15)
# y=b0+b1x1

x=sm.add_constant(X2)

results=sm.OLS(Y,x).fit()

#Contain Ordinary Least Square Regression 

results.summary()
fig = px.scatter(data, x="Study hrs.", y="Marks",title='MARKS vs Study Hrs  ', trendline="ols",color="Study hrs.")

fig.show()
plt.figure(figsize=(13,7))
plt.title('Residual Plot for Marks and Study Hrs',size=20)
sns.residplot(Y, X2, lowess=True, color="c")
plt.ylabel('Residuals',size=15)
# regression
# data['bestfit'] = sm.OLS(Y,sm.add_constant(X1)).fit().fittedvalues

# plotly figure setup
fig=go.Figure()
#********************#


fig.add_trace(go.Scatter(name='Marks vs IQ', x=X1, y=Y, mode='markers'))

data['bestfit'] = sm.OLS(Y,sm.add_constant(X1)).fit().fittedvalues

fig.add_trace(go.Scatter(name='line of best fit', x=X1, y=data['bestfit'], mode='lines'))

#********************#



fig.add_trace(go.Scatter(name='MARKS vs Study hrs', x=X2, y=Y, mode='markers'))

data['bestfit'] = sm.OLS(Y,sm.add_constant(X2)).fit().fittedvalues

fig.add_trace(go.Scatter(name='line of best fit', x=X2, y=data['bestfit'], mode='lines'))

#********************#



# plotly figure layout
fig.update_layout(xaxis_title = 'Independent Variables', yaxis_title = 'MARKS',title='MARKS VS IQ and Study hrs')

fig.show()
# y=b0+b1x1
X = data[['IQ','Study hrs.']]
x=sm.add_constant(X)

results=sm.OLS(Y,x).fit()

#Contain Ordinary Least Square Regression 
results.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor


t = data[['IQ', 'Study hrs.']]
t['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["Variables"] = t.columns
vif["VIF"] = [variance_inflation_factor(t.values, i) for i in range(t.shape[1])]

# View results using print
print(vif)