import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
#Plotting

import plotly

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

import plotly.figure_factory as ff

from plotly import subplots

# Display plot in the same cell for plotly

init_notebook_mode(connected=True)
import sklearn

from sklearn import linear_model,metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error, make_scorer
print('Numpy : Version ',np.__version__)

print('Pandas : Version ',pd.__version__)

print('Plotly : Version ',plotly.__version__)

print('Plotly Express : Version ',plotly.express.__version__)

print('Scikit-Learn : Version ',sklearn.__version__)
# Colors from material design to make visualizations look awesome!

material_blue = '#81d4fa'

dark_blue = '#1e88e5'

material_green = '#a5d6a7'

dark_green = '#43a047'

material_indigo = '#3f51b5'

material_red = '#f44336'

bg_color = '#212121'
# Importing the train dataset

df_train = pd.read_csv('../input/train.csv')

df_train.head()
# Importing the test dataset

df_test = pd.read_csv('../input/test.csv')

df_test.head()
# Column names

df_train.columns
# Shape => Tuple of no. of rows and columns

df_train.shape
df_train.describe()
total = df_train.isnull().sum().sort_values(ascending=False)

missing_cols = list(total.index)

total_values = list(total[:])

df_missing = pd.DataFrame(dict({'columns':missing_cols,'total':total_values}))

df_missing = df_missing[df_missing['total']>0]

df_missing
fig = px.bar(df_missing, x='columns',y='total')

fig.update_traces(marker_color=dark_blue)

iplot(fig)
missing_data_cols = list(df_missing['columns'])

df_train[missing_data_cols].head()
df_train[df_train['PoolArea']>0].shape[0]
df_train[df_train['Fireplaces']>0].shape[0]
cols_to_be_del = ['PoolQC','PoolArea','MiscFeature','Alley','Fence','Fireplaces','FireplaceQu','LotFrontage']

df_train.drop(cols_to_be_del, inplace=True, axis=1)

df_train.shape
df_test.head()
compare_df = df_train[df_train['GarageCars']>0]['Id'] == df_train[df_train['GarageArea']>0]['Id']

compare_df.shape[0]
df_missing.tail(13)
cols_to_be_del_2 = ['GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageType','MasVnrType']

df_train.drop(cols_to_be_del_2, inplace=True, axis=1)

df_train.shape
cols_to_be_del_3 = ['BsmtQual','BsmtCond','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical']

df_train.drop(cols_to_be_del_3, axis=1, inplace=True)

df_train.shape
df_missing.tail(3)
df_train['MasVnrArea'].describe()
df_train.fillna(df_train.median(), inplace=True)

df_train.isnull().values.sum()
print("Skewness: {}".format(str(df_train['SalePrice'].skew())))

print("Kurtosis: {}".format(str(df_train['SalePrice'].kurt())))
fig = px.histogram(df_train, x='SalePrice', nbins=100)

iplot(fig)
fig = ff.create_distplot([df_train['SalePrice']],['SalePrice'],bin_size=20000, colors=[dark_blue],show_hist=False)

iplot(fig, filename='Basic Distplot')
def box_plot(dataframe, columns):

    data = []

    for column in columns:

        data.append(go.Box(y=df_train[column], name=column, boxmean='sd',fillcolor=material_blue,marker=dict(color=dark_blue)))

    return data
target_box_data = box_plot(df_train,['SalePrice'])

iplot(target_box_data)
def violin_plot(dataframe, columns):

    data = []

    for column in columns:

        data.append(go.Violin(y=dataframe[column], box_visible=True, line_color=bg_color,

                               meanline_visible=True, fillcolor=material_green, opacity=0.8,

                               x0=column))

    return data
violin_fig = violin_plot(df_train, ['SalePrice'])

iplot(violin_fig, filename = 'SalePriceViolin')
def qqplots(df, col_name, distribution):

    qq = stats.probplot(df, dist=distribution, sparams=(0.5))

    x = np.array([qq[0][0][0],qq[0][0][-1]])

    pts = go.Scatter(x=qq[0][0],

                     y=qq[0][1], 

                     mode = 'markers',

                     showlegend=False,

                     name=col_name,

                     marker = dict(

                            size = 5,

                            color = material_indigo,

                        )

                    )

    line = go.Scatter(x=x,

                      y=qq[1][1] + qq[1][0]*x,

                      showlegend=False,

                      mode='lines',

                      name=distribution,

                      marker = dict(

                            size = 5,

                            color = material_red,

                        )

                     )

    

    data = [pts, line]

    return data
#Plot data for 4 different distributions

norm_data = qqplots(df_train['SalePrice'], 'SalePrice','norm')

power_law_data = qqplots(df_train['SalePrice'], 'SalePrice','powerlaw')

poisson_data = qqplots(df_train['SalePrice'], 'SalePrice','poisson')

lognorm_data = qqplots(df_train['SalePrice'], 'SalePrice','lognorm')
fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=('Normal Distribution', 'Power Law Distribution',

                                                          'Poisson Distribution', 'Log Normal Distribution'))

fig.append_trace(norm_data[0], 1, 1)

fig.append_trace(power_law_data[0], 1, 2)

fig.append_trace(poisson_data[0], 2, 1)

fig.append_trace(lognorm_data[0], 2, 2)

fig['layout'].update(height=600, width=900, title='Comparision of QQ-plots')



iplot(fig, filename='make-subplots-multiple-with-titles')
layout = dict(xaxis = dict(zeroline = False,

                           linewidth = 1,

                           mirror = True),

              yaxis = dict(zeroline = False, 

                           linewidth = 1,

                           mirror = True),

             )



fig = dict(data=lognorm_data, layout=layout)

iplot(fig, show_link=False)
# Creating a pipeline

df_pipe = df_train.copy()
df_pipe['SalePrice'] = np.log(df_train['SalePrice'])

print("Skewness: {}".format(str(df_pipe['SalePrice'].skew())))

print("Kurtosis: {}".format(str(df_pipe['SalePrice'].kurt())))
fig = px.histogram(df_pipe,'SalePrice')

iplot(fig)
fig = ff.create_distplot([df_pipe['SalePrice']],['SalePrice Log Normal'],bin_size=0.08, colors=[dark_blue], show_hist=False)

iplot(fig, filename='Distribution plot for Sale Price (Log transform)')
log_transformed_qqplot_data = qqplots(df_pipe['SalePrice'], 'SalePrice Log transform','norm')

layout = dict(xaxis = dict(zeroline = False,

                       linewidth = 1,

                       mirror = True),

          yaxis = dict(zeroline = False, 

                       linewidth = 1,

                       mirror = True),

         )

qqplot_fig = dict(data=log_transformed_qqplot_data, layout=layout)

iplot(qqplot_fig, show_link=False)
target_transformed_box_data = [go.Box(y=df_pipe['SalePrice'], name='SalePrice Log transform', boxmean='sd',fillcolor=material_green,marker=dict(color=dark_green))]

iplot(target_transformed_box_data)
target_transformed_violin_data = violin_plot(df_pipe,['SalePrice'])

iplot(target_transformed_violin_data, filename = 'SalePriceLogViolin', validate = False)
fig = subplots.make_subplots(rows=1, cols=2)

fig.append_trace(target_box_data[0], 1, 1)

fig.append_trace(target_transformed_box_data[0], 1, 2)

fig['layout'].update(height=600, width=950, title='SalePrice Unchanged vs Log transformed')

iplot(fig, filename='SalePrice-unch-vs-log-box')
print("Skewness: {}".format(str(df_pipe['SalePrice'].skew())))

print("Kurtosis: {}".format(str(df_pipe['SalePrice'].kurt())))
#Feature Engineering



df_pipe['TotalSF']=df_pipe['TotalBsmtSF'] + df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF']

df_pipe['TotalSQR_Footage'] = (df_pipe['BsmtFinSF1'] + df_pipe['BsmtFinSF2'] +

                                df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF'])



df_pipe['Total_Bathrooms'] = (df_pipe['FullBath'] + (0.5 * df_pipe['HalfBath']) +

                              df_pipe['BsmtFullBath'] + (0.5 * df_pipe['BsmtHalfBath']))



df_pipe['AgeSinceRemodel'] = 2010 - df_train['YearRemodAdd']

df_pipe['AgeSinceBuilt'] = 2010 - df_train['YearBuilt']
corr_matrix = df_pipe.corr()

corr_matrix = corr_matrix.abs()

fig = go.Figure(data=go.Heatmap(z=corr_matrix))

iplot(fig)
k = 15 #number of variables for heatmap

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index



cols_to_be_del_pipe = ['FullBath','1stFlrSF']

cols = list(cols)

for i in cols_to_be_del_pipe:

    cols.remove(i)



correlations = list(df_pipe[cols].corr().round(2).values)

correlation_matrix = [list(correlations[i]) for i in range(len(correlations))]

fig = ff.create_annotated_heatmap(z=correlation_matrix,x=list(cols),y=list(cols))

iplot(fig, filename='annotated_heatmap')
#Categorical variables

ordinal_cols = ['BldgType','HeatingQC','Functional']

binary_cols = ['PavedDrive','CentralAir']

df_pipe[ordinal_cols] = df_pipe[ordinal_cols].astype('category')

df_pipe[ordinal_cols].head()
def count_plot(df,col_name):

    value_counts_series = df[col_name].value_counts()

    categories = list(value_counts_series.index)

    values = list(value_counts_series)

    fig = go.Figure(data=[go.Bar(

            x=categories, 

            y=values,

            textposition='auto',

        )])

    iplot(fig)
count_plot(df_pipe, 'HeatingQC')
fig = px.box(df_train, x='HeatingQC', y='SalePrice')

iplot(fig)
# Label Encoding for HeatingQC

categories = list(df_pipe['HeatingQC'].unique())

encoding_dict = {col:x for col,x in zip(categories,range(5,0,-1))}

replace_dict_heatingqc = {'HeatingQC':encoding_dict}

df_pipe.replace(replace_dict_heatingqc, inplace = True)

df_pipe[ordinal_cols].head()
skews = []

kurts = []

for col in cols:

    skews.append(df_pipe[col].skew())

    kurts.append(df_pipe[col].kurt())

dict_skew_data = {'Feature':cols, 'Skew':skews, 'Kurt':kurts}

df_skews = pd.DataFrame(dict_skew_data, columns=['Feature','Skew','Kurt'])

df_skews
df_pipe['TotalSF'] = np.log(df_pipe['TotalSF'])
iplot(qqplots(df_pipe['TotalSF'],'TotalSF Log transform','norm'))
fig = px.scatter_matrix(df_pipe, dimensions=['TotalSF','GrLivArea','TotalSQR_Footage','SalePrice'],color='OverallQual')

iplot(fig)
fig = px.scatter_matrix(df_pipe, dimensions=['Total_Bathrooms','GarageArea','GarageCars','SalePrice'],color='OverallQual')

iplot(fig)
fig = px.scatter_matrix(df_pipe, dimensions=['AgeSinceBuilt','AgeSinceRemodel','TotRmsAbvGrd','TotalBsmtSF','SalePrice'],color='OverallQual')

iplot(fig)
def draw_scatter_plot(col_name_x, col_name_y):

    trace = go.Scatter(

        x = df_pipe[col_name_x],

        y = df_pipe[col_name_y],

        mode = 'markers'

    )

    data = [trace]

    iplot(data, filename='basic-scatter')
y_tr = df_pipe['SalePrice']

x_tr = df_pipe[cols]

lasso = linear_model.Lasso()

parameters = {'alpha': [1]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring=make_scorer(metrics.mean_squared_error), cv=10)

lasso_regressor.fit(x_tr, y_tr)

y_pred_lasso = lasso_regressor.predict(x_tr)
mse = metrics.mean_squared_error(y_tr, y_pred_lasso, sample_weight=None)

rmse = np.sqrt(mse)

print(rmse)