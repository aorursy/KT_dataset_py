import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from IPython.display import HTML, display
from IPython.core import display as ICD
from plotly.offline import init_notebook_mode, iplot

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
import statsmodels.api as sm
import pylab
import scipy as sp

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model

init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
PATH = '../input/'
filename = 'winequality-white.csv'
white_data = pd.read_csv(PATH + filename)

data_head = white_data.head()
colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
df_table = ff.create_table(round(data_head.iloc[:,[0,1,2,3,4,5]], 3), colorscale=colorscale)
py.iplot(df_table, filename='wine_quality')
df_table = ff.create_table(round(data_head.iloc[:,[6,7,8,9,10,11]], 3), colorscale=colorscale)
py.iplot(df_table, filename='wine_quality')
value_counts = white_data.quality.value_counts()
target_counts = pd.DataFrame({'quality': list(value_counts.index), 'value_count': value_counts})
plt.figure(figsize=(10,4))
g = sns.barplot(x='quality', y='value_count', data=target_counts, capsize=0.3, palette='spring')
g.set_title("Frequency of target class", fontsize=15)
g.set_xlabel("Quality", fontsize=13)
g.set_ylabel("Frequency", fontsize=13)
g.set_yticks([0, 500, 1000, 1500, 2000, 2500])
for p in g.patches:
    g.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
plt.figure(figsize=(10,3))
sns.boxplot(data=white_data['quality'], orient='horizontal', palette='husl')
plt.title("Distribution of target variable")
white_data.describe().drop(columns=['quality'])

# data_head = white_data.describe().drop(columns=['quality'])
# data_head.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
#        'chlorides', 'free_SO2', 'total_SO2', 'density',
#        'pH', 'sulphates', 'alcohol']
# colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
# df_table = ff.create_table(round(data_head.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]], 3), colorscale=colorscale)
# py.iplot(df_table, filename='wine_quality')
# df_table = ff.create_table(round(data_head.iloc[:,[6,7,8,9,10]], 3), colorscale=colorscale)
# py.iplot(df_table, filename='wine_quality')
plt.figure(figsize=(10,10))
sns.boxplot(data=white_data.drop(columns=['quality']), orient='horizontal', palette='husl')
y = white_data['quality']
white_data = white_data.loc[:, ~white_data.columns.isin(['quality'])]

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(white_data)
white_data.loc[:,:] = scaled_values

white_data['quality'] = y
data_head = white_data.head()
colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
df_table = ff.create_table(round(data_head.iloc[:,[0,1,2,3,4,5]], 3), colorscale=colorscale, )
py.iplot(df_table, filename='wine_quality')
df_table = ff.create_table(round(data_head.iloc[:,[6,7,8,9,10,11]], 3), colorscale=colorscale, )
py.iplot(df_table, filename='wine_quality')
columns = list(white_data.columns)
new_column_names = []
for col in columns:
    new_column_names.append(col.replace(' ', '_'))
white_data.columns = new_column_names
plt.figure(figsize=(10,10))
sns.boxplot(data=white_data.drop(columns=['quality']), orient='horizontal', palette='husl')
corr_matrix = white_data.corr().abs()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map({'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
f, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
sns.distplot(features["fixed_acidity"], rug=False, color="skyblue", ax=axes[0, 0])
sns.distplot(features["volatile_acidity"], rug=False, color="olive", ax=axes[0, 1])
sns.distplot(features["citric_acid"], rug=False, color="gold", ax=axes[0, 2])
sns.distplot(features["residual_sugar"], rug=False, color="teal", ax=axes[1, 0])
sns.distplot(features["chlorides"], rug=False, ax=axes[1, 1])
sns.distplot(features["free_sulfur_dioxide"], rug=False, color="red", ax=axes[1, 2])
sns.distplot(features["total_sulfur_dioxide"], rug=False, color="skyblue", ax=axes[2, 0])
sns.distplot(features["density"], rug=False, color="olive", ax=axes[2, 1])
sns.distplot(features["pH"], rug=False, color="gold", ax=axes[2, 2])
sns.distplot(features["sulphates"], rug=False, color="teal", ax=axes[3, 0])
sns.distplot(features["alcohol"], rug=False, ax=axes[3, 1])
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map({'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
sns.pairplot(features, diag_kind='kde', palette='husl', hue='quality')
features = white_data.copy(deep=True)
features['quality'] = y.astype('str').map({'3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'})
sns.pairplot(features, vars=to_drop, diag_kind='kde', palette='husl', hue='quality')
model_reg = LinearRegression().fit(white_data.drop(columns=['quality']), y)
y_true = white_data.quality
y_pred = model_reg.predict(white_data.drop(columns=['quality']))
column_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
regression_coefficient = pd.DataFrame({'Feature': column_names, 'Coefficient': model_reg.coef_}, columns=['Feature', 'Coefficient'])
column_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

plt.figure(figsize=(15,5))
g = sns.barplot(x='Feature', y='Coefficient', data=regression_coefficient, capsize=0.3, palette='spring')
g.set_title("Contribution of features towards target variable", fontsize=15)
g.set_xlabel("Feature", fontsize=13)
g.set_ylabel("Degree of Coefficient", fontsize=13)
g.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
g.set_xticklabels(column_names)
for p in g.patches:
    g.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
               textcoords='offset points', fontsize=14, color='black')
model_ols = ols("""quality ~ fixed_acidity 
                        + volatile_acidity 
                        + citric_acid
                        + residual_sugar 
                        + chlorides 
                        + free_sulfur_dioxide
                        + total_sulfur_dioxide 
                        + density 
                        + pH 
                        + sulphates 
                        + alcohol""", data=white_data).fit()
model_summary = model_ols.summary()
HTML(
(model_ols.summary()
    .as_html()
    .replace('<th>Dep. Variable:</th>', '<th style="background-color:#c7e9c0;"> Dep. Variable: </th>')
    .replace('<th>Model:</th>', '<th style="background-color:#c7e9c0;"> Model: </th>')
    .replace('<th>Method:</th>', '<th style="background-color:#c7e9c0;"> Method: </th>')
    .replace('<th>No. Observations:</th>', '<th style="background-color:#c7e9c0;"> No. Observations: </th>')
    .replace('<th>  R-squared:         </th>', '<th style="background-color:#aec7e8;"> R-squared: </th>')
    .replace('<th>  Adj. R-squared:    </th>', '<th style="background-color:#aec7e8;"> Adj. R-squared: </th>')
    .replace('<th>coef</th>', '<th style="background-color:#ffbb78;">coef</th>')
    .replace('<th>std err</th>', '<th style="background-color:#c7e9c0;">std err</th>')
    .replace('<th>P>|t|</th>', '<th style="background-color:#bcbddc;">P>|t|</th>')
    .replace('<th>[0.025</th>    <th>0.975]</th>', '<th style="background-color:#ff9896;">[0.025</th>    <th style="background-color:#ff9896;">0.975]</th>'))
)
model_ols = ols("""quality ~ fixed_acidity 
                        + volatile_acidity 
                        + residual_sugar 
                        + free_sulfur_dioxide
                        + total_sulfur_dioxide 
                        + density 
                        + pH 
                        + sulphates 
                        + alcohol""", data=white_data).fit()
model_summary = model_ols.summary()
HTML(
(model_ols.summary()
    .as_html()
    .replace('<th>Dep. Variable:</th>', '<th style="background-color:#c7e9c0;"> Dep. Variable: </th>')
    .replace('<th>Model:</th>', '<th style="background-color:#c7e9c0;"> Model: </th>')
    .replace('<th>Method:</th>', '<th style="background-color:#c7e9c0;"> Method: </th>')
    .replace('<th>No. Observations:</th>', '<th style="background-color:#c7e9c0;"> No. Observations: </th>')
    .replace('<th>  R-squared:         </th>', '<th style="background-color:#aec7e8;"> R-squared: </th>')
    .replace('<th>  Adj. R-squared:    </th>', '<th style="background-color:#aec7e8;"> Adj. R-squared: </th>')
    .replace('<th>coef</th>', '<th style="background-color:#ffbb78;">coef</th>')
    .replace('<th>std err</th>', '<th style="background-color:#c7e9c0;">std err</th>')
    .replace('<th>P>|t|</th>', '<th style="background-color:#bcbddc;">P>|t|</th>')
    .replace('<th>[0.025</th>    <th>0.975]</th>', '<th style="background-color:#ff9896;">[0.025</th>    <th style="background-color:#ff9896;">0.975]</th>'))
)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def goodness(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return mape, mse, r_squared
model = LinearRegression().fit(white_data.drop(columns=['quality', 'citric_acid', 'chlorides']), y)
y_true = white_data.quality
y_pred = model.predict(white_data.drop(columns=['quality', 'citric_acid', 'chlorides']))
column_names = ['fixed_acidity', 'volatile_acidity', 'residual_sugar',
       'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
regression_coefficient = pd.DataFrame({'Feature': column_names, 'Coefficient': model.coef_}, columns=['Feature', 'Coefficient'])
column_names = ['fixed_acidity', 'volatile_acidity', 'residual_sugar',
       'free_SO2', 'total_SO2', 'density',
       'pH', 'sulphates', 'alcohol']

plt.figure(figsize=(15,5))
g = sns.barplot(x='Feature', y='Coefficient', data=regression_coefficient, capsize=0.3, palette='spring')
g.set_title("Contribution of features towards target variable", fontsize=15)
g.set_xlabel("Feature", fontsize=13)
g.set_ylabel("Degree of Coefficient", fontsize=13)
g.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
g.set_xticklabels(column_names)
for p in g.patches:
    g.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
               textcoords='offset points', fontsize=14, color='black')
error = y_true - y_pred
error_info = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'error': error}, columns=['y_true', 'y_pred', 'error'])
fig = plt.figure(figsize=(10,12))
fig = sm.graphics.plot_partregress_grid(model_ols, fig=fig)
plt.figure(figsize=(8,5))
g = sns.regplot(x="y_pred", y="error", data=error_info, color='blue')
g.set_title('Check Homoskedasticity', fontsize=15)
g.set_xlabel("predicted values", fontsize=13)
g.set_ylabel("Residual", fontsize=13)
fig, ax = plt.subplots(figsize=(8,5))
ax = error_info.error.plot()
ax.set_title('Uncorrelated errors', fontsize=15)
ax.set_xlabel("Data", fontsize=13)
ax.set_ylabel("Residual", fontsize=13)
fig, ax = plt.subplots(figsize=(6,4))
_ = sp.stats.probplot(error_info.error, plot=ax, fit=True)
ax.set_title('Probability plot', fontsize=15)
ax.set_xlabel("Theoritical Qunatiles", fontsize=13)
ax.set_ylabel("Ordered Values", fontsize=13)
ax = sm.qqplot(error_info.error, line='45')
plt.show()
pca = PCA()
transform_X = pca.fit_transform(white_data.drop(columns=['quality']), white_data.quality)

columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7',
            'feature_8', 'feature_9', 'feature_10', 'feature_11']
transform_df = pd.DataFrame.from_records(transform_X)
transform_df.columns = columns
transform_df['quality'] = white_data.quality
model_ols_new = ols("""quality ~ feature_1 
                        + feature_2 
                        + feature_3
                        + feature_4 
                        + feature_5 
                        + feature_6 
                        + feature_7 
                        + feature_8 
                        + feature_9 
                        + feature_10 
                        + feature_11""", data=transform_df).fit()
model_summary = model_ols_new.summary()
HTML(
(model_ols_new.summary()
    .as_html()
    .replace('<th>Dep. Variable:</th>', '<th style="background-color:#c7e9c0;"> Dep. Variable: </th>')
    .replace('<th>Model:</th>', '<th style="background-color:#c7e9c0;"> Model: </th>')
    .replace('<th>Method:</th>', '<th style="background-color:#c7e9c0;"> Method: </th>')
    .replace('<th>No. Observations:</th>', '<th style="background-color:#c7e9c0;"> No. Observations: </th>')
    .replace('<th>  R-squared:         </th>', '<th style="background-color:#aec7e8;"> R-squared: </th>')
    .replace('<th>  Adj. R-squared:    </th>', '<th style="background-color:#aec7e8;"> Adj. R-squared: </th>')
    .replace('<th>coef</th>', '<th style="background-color:#ffbb78;">coef</th>')
    .replace('<th>std err</th>', '<th style="background-color:#c7e9c0;">std err</th>')
    .replace('<th>P>|t|</th>', '<th style="background-color:#bcbddc;">P>|t|</th>')
    .replace('<th>[0.025</th>    <th>0.975]</th>', '<th style="background-color:#ff9896;">[0.025</th>    <th style="background-color:#ff9896;">0.975]</th>'))
)
r2_linear_regression = model_ols_new.rsquared

model_ridge=linear_model.Ridge()
model_ridge.fit(white_data.drop(columns=['quality']),white_data.quality)
y_predict_ridge = model_ridge.predict(white_data.drop(columns=['quality']))
r2_ridge = r2_score(y_true, y_predict_ridge)

model_lasso=linear_model.Lasso()
model_lasso.fit(white_data.drop(columns=['quality']),white_data.quality)
y_predict_lasso = model_lasso.predict(white_data.drop(columns=['quality']))
r2_score(y_true, y_predict_lasso)

n_neighbors=5
knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
knn.fit(white_data.drop(columns=['quality']),white_data.quality)
y_predict_knn=knn.predict(white_data.drop(columns=['quality']))
r2_knn = r2_score(y_true, y_predict_knn)

reg = linear_model.BayesianRidge()
reg.fit(white_data.drop(columns=['quality']),white_data.quality)
y_pred_reg=reg.predict(white_data.drop(columns=['quality']))
r2_bayesian = r2_score(y_true, y_pred_reg)

dec = tree.DecisionTreeRegressor(max_depth=6)
dec.fit(white_data.drop(columns=['quality']),white_data.quality)
y1_dec=dec.predict(white_data.drop(columns=['quality']))
r2_dt = r2_score(y_true, y1_dec)

svm_reg=svm.SVR()
svm_reg.fit(white_data.drop(columns=['quality']),white_data.quality)
y1_svm=svm_reg.predict(white_data.drop(columns=['quality']))
r2_svm = r2_score(y_true, y1_svm)
r2_list = [r2_linear_regression, r2_ridge, r2_knn, r2_dt, r2_bayesian, r2_svm]
r2_names = ['Linear Regression', 'Ridge Regression', 'KNN', 'Decision Tree', 'Bayesian Regression', 'SVM']

col = {'R-squared':r2_list, 'Method':r2_names}
df = pd.DataFrame(data=col, columns=['Method', 'R-squared'])

data_head = df
colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
df_table = ff.create_table(round(data_head.iloc[:,[0,1]], 3), colorscale=colorscale)
py.iplot(df_table, filename='wine_quality')
