# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sb
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
data.head(10)
data.info()
data.tail(10)
data.describe()
data.columns
data.shape
suicide_in_countries = data.groupby('country')
country_suicide_series = suicide_in_countries['suicides_no'].sum()
country_suicide_series.head()
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
# 2016 yılına ait veriseti
sene_2016 = data[(data['year'] == 2016)]

# 2016 yılındaki toplam intihar sayısı (ülke çapında)
sene_2016 = sene_2016.groupby('country')[['suicides_no']].sum().reset_index()

# Değerleri artan düzende sıraladım
sene_2016 = sene_2016.sort_values(by='suicides_no', ascending=False)
fig = px.bar(sene_2016, x=sene_2016['suicides_no'], y=sene_2016['country'], color=sene_2016['country'], color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_layout(title={
                         'text':'2016 yılında intiharlar (Ülke bazında)',
                         'y':0.98,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'},
                  plot_bgcolor='white', 
                  height=1000,
                  showlegend=False
                 )

fig.show()
trace1 = go.Bar(
    y=country_suicide_series.values,
    x=country_suicide_series.index,
)

data = [trace1]
layout = go.Layout(
    title="Her ülkede yapılan intihar sayısı",
    xaxis={
        'title':"Ülkeler",
    },
    yaxis={
        'title':"İntihar Sayısı",
    }
)
figure=go.Figure(data=data,layout=layout)
py.offline.iplot(figure)
plt.figure(figsize=(10,30))
sb.set_style('dark')
sb.barplot(country_suicide_series.values,country_suicide_series.index)
plt.show()
sb.barplot(genderwise_suicide.index , genderwise_suicide.suicides_no)
plt.xlabel('Cinsiyet')
plt.ylabel('İntihar Sayısı(m)')
sb.set_style('white')
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
data.rename(columns={'gdp_per_capita ($)': 'gdp_per_capita'}, inplace = True)
data = data[['gdp_per_capita', 'suicides_no', 'sex']]
def t_c(df):
    if df['sex'] == 'female':
        return 0    
    else:
        return 1
data['sex'] = data.apply(t_c, axis=1)
data['gdp_per_capita'] /= data['gdp_per_capita'].max()*0.01
data['suicides_no'] /= data['suicides_no'].max()*0.01
data.head()
sb.scatterplot(data = data, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')
X = data.iloc[:,0:2]
y = data.iloc[:,2]
h = lambda x: 1 / (1 + np.exp(-g(x)))
X = np.concatenate((np.ones((X.shape[0], 1)) , X), axis = 1)
theta = np.zeros(X.shape[1])

lr = 0.15 # learning rate
epochs = 500
costs = []
paras = []

def cal_cost(h, x, y):
    return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()    

def log_reg(h, x, y, theta, lr, epochs):
    for i in range(epochs):
        z = np.dot(X, theta)
        h = 1/(1 + np.exp(-z))
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient
        cost = cal_cost(h, x, y)
        costs.append(cost)
        paras.append(theta)

    print(costs[-5:])
    print(paras[-5:])
    
log_reg(h, X, y, theta, lr, epochs)
def plot_line(theta, x):
    y = lambda x: -(theta[0] + theta[1] * x)/theta[2]
    x_values = [i for i in range(int(min(x))-1, int(max(x))+2)]
    y_values = [y(x) for x in x_values]
    color = list(np.random.random(size=3))
    plt.plot(x_values, y_values, c = color)
sb.scatterplot(data = data, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')
for i,t in enumerate(paras):
    if i%100 == 0: 
        plot_line(t, list(data.iloc[:, 0]))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

X = data.iloc[:,0:2]
model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y, predicted_classes)
parameters = model.coef_
intercept = model.intercept_

print(accuracy)
print(intercept, parameters)
sb.scatterplot(data = data, y = 'suicides_no', x = 'gdp_per_capita', hue = 'sex')
plot_line([intercept, parameters[0][0], parameters[0][1]], list(data.iloc[:, 0]))
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
(data['suicides_no'][data['sex']=='male']).sum()
genderwise_suicide = data.pivot_table(index='sex' , aggfunc='sum')
genderwise_suicide['suicides_no']
data['suicides_no'].sum()
import seaborn as sns
data = data.reset_index()
del data['country-year']
del data['HDI for year']
del data['suicides_no']
del data['population']
del data[' gdp_for_year ($) ']
header = ['country','year','sex','age','suicides/100k pop','gdp_per_capita ($)','generation','region']
data = data.reindex(columns = header)
Europe = ["Albania","Russian Federation","France","Ukraine","Germany","Poland","United Kingdom",
         "Italy","Spain","Hungary","Romania","Belgium","Belarus","Netherlands","Austria",
         "Czech Republic","Sweden","Bulgaria","Finland","Lithuania","Switzerland","Serbia",
         "Portugal","Croatia","Norway","Denmark","Slovakia","Latvia","Greece","Slovenia",
         "Turkey","Estonia","Georgia","Albania","Luxembourg","Armenia","Iceland","Montenegro",
         "Cyprus","Bosnia and Herzegovina","San Marino","Malta","Ireland"]
NorthAmerica = ["United States","Mexico","Canada","Cuba","El Salvador","Puerto Rico",
                "Guatemala","Costa Rica","Nicaragua","Belize","Jamaica"]
SouthAmerica = ["Brazil","Colombia", "Chile","Ecuador","Uruguay","Paraguay","Argentina",
                "Panama","Guyana","Suriname"]
MiddleEast = ["Kazakhstan","Uzbekistan","Kyrgyzstan","Israel","Turkmenistan","Azerbaijan",
              "Kuwait","United Arab Emirates","Qatar","Bahrain","Oman"]
Asia = ["Japan","Republic of Korea", "Thailand", "Sri Lanka","Philippines","New Zealand",
        "Australia","Singapore","Macau","Mongolia"]
for i in range(0,len(data)):
    if data.iloc[i,0] in Europe:
        data.iloc[i,7] = "Europe"
    elif data.iloc[i,0] in NorthAmerica:
        data.iloc[i,7] = "North America"
    elif data.iloc[i,0] in SouthAmerica:
        data.iloc[i,7] = "South America"
    elif data.iloc[i,0] in MiddleEast:
        data.iloc[i,7] = "Middle East"
    elif data.iloc[i,0] in Asia:
        data.iloc[i,7] = "Asia"
    else:
        data.iloc[i,7] = "Island Nation"
del data['country']
suicide_cat = data[['sex','age','generation','region']]
one_hot_data = pd.get_dummies(suicide_cat)
year = data['year']
gdp_per_cap = data['gdp_per_capita ($)']
suicide_per_100k = data['suicides/100k pop']
data2 = pd.concat([year, gdp_per_cap, one_hot_data], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data2, suicide_per_100k, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
params = {'max_leaf_nodes': list(range(93,95)), 'min_samples_split': list(range(6,8)), 'min_samples_leaf':list(range(2,4))}    
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42),
                              params, n_jobs=-1, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = grid_search_cv.predict(X_cv)
tree_reg_mse = mean_squared_error(y_cv, y_pred)
tree_reg_rmse = np.sqrt(tree_reg_mse)
print("Karar Ağacı Regresyon(Decision Tree Regression) modelinde ayarlanan CV için Kök Ortalama Kare Hatası(Root-Mean-Squared):",tree_reg_rmse)
####################################
from pandas.plotting import scatter_matrix
attributes = ['suicides/100k pop','year','gdp_per_capita ($)']
scatter_matrix(data[attributes], figsize=(12,8))
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred_lin_reg = lin_reg.predict(X_cv)
lin_reg_mse = mean_squared_error(y_cv, y_pred_lin_reg)
lin_reg_rmse = np.sqrt(lin_reg_mse)
print("Doğrusal regresyon modelinde ayarlanan CV için Kök Ortalama Karesi Hatası:",lin_reg_rmse)
y_pred = grid_search_cv.predict(X_test)
tree_reg_mse = mean_squared_error(y_test, y_pred)
tree_reg_rmse = np.sqrt(tree_reg_mse)
print("Karar Ağacı Regresyon modelinde ayarlanan Test için Kök Ortalama Kare Hatası: :",tree_reg_rmse)