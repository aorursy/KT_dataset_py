import pandas as pd

import numpy as np



from plotly.subplots import make_subplots

import plotly.graph_objects as go



from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_squared_error
data = pd.read_csv('../input/fish-market/Fish.csv')



data.head()
data.rename(columns={'Length1':'Vertical length', 

                     'Length2':'Diagonal length', 

                     'Length3':'Cross length'}, 

            inplace=True)



data.head()
data.describe()
features_to_plot = {'Weight':[1,1], 

                    'Vertical length':[1,2], 

                    'Diagonal length':[1,3], 

                    'Cross length':[2,1], 

                    'Height':[2,2], 

                    'Width':[2,3]}



box_violin_plot = make_subplots(rows=2, cols=3)



for i in features_to_plot.keys():

    box_violin_plot.add_trace(go.Box(y=data[i], name=i, marker_color='#342ead'), 

                              row=features_to_plot[i][0], col=features_to_plot[i][1])

    box_violin_plot.add_trace(go.Violin(y=data[i], name=i, marker_color='#ea6227'), 

                              row=features_to_plot[i][0], col=features_to_plot[i][1])

    

box_violin_plot.update_layout(width=850, height=700, showlegend=False)



box_violin_plot.show()
features_to_plot = {'Vertical length':[1,1], 

                    'Diagonal length':[1,2], 

                    'Cross length':[1,3], 

                    'Height':[2,1], 

                    'Width':[2,2]}



scatter_fig = make_subplots(rows=2, cols=3,   y_title='Weight', shared_yaxes=True)



for i in features_to_plot.keys():

    scatter_fig.add_trace(go.Scatter(x=data[i], y=data['Weight'], mode='markers', 

                                     name=i), 

                          row=features_to_plot[i][0], col=features_to_plot[i][1])



scatter_fig.update_layout(width=1380, height=700)



scatter_fig.show()
for k in data.columns.to_list()[1:]:

    percentile_25 = np.percentile(data[k], 25)

    percentile_75 = np.percentile(data[k], 75)

    iqr = percentile_75 - percentile_25

    lower = percentile_25 - (iqr*1.5)

    upper = percentile_75 + (iqr*1.5)

    for i in data[k]:

        if ((i > upper) or (i < lower)):

            data.drop((data.index[data[k] == i].to_list()), axis=0, inplace=True)
data = pd.get_dummies(data, columns = ['Species'])



data.drop(['Species_Whitefish'], axis = 1, inplace = True)



data.head()
scaler = StandardScaler()



scaler.fit(data.iloc[:, 1:6])



data_scaled = scaler.transform(data.iloc[:, 1:6])
X  = np.concatenate((data_scaled, data.iloc[:, 6:-1]), axis = 1)



y = data['Weight']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
linear_regression = LinearRegression()



linear_regression.fit(X_train, y_train)



y_pred_linear = linear_regression.predict(X_test)



rsq_linear = r2_score(y_test, y_pred_linear)



rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
polynomial_features = PolynomialFeatures(degree=2)



polynomial_features.fit(X_train)



X_train_poly = polynomial_features.transform(X_train)

X_test_poly = polynomial_features.transform(X_test)



polynomial_regression = LinearRegression()



polynomial_regression.fit(X_train_poly, y_train)



y_pred_poly = polynomial_regression.predict(X_test_poly)



rsq_poly = r2_score(y_test, y_pred_poly)



rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
svr_parameters = {'kernel' : ['linear', 'poly', 'rbf'],

                  'degree' : [1, 3, 5],

                  'C' : [1, 10, 100, 1000]

                 }



grid_search_svr = GridSearchCV(estimator = SVR(), 

                           param_grid = svr_parameters,

                           cv = 10,

                           n_jobs = -1)



grid_search_svr.fit(X_train, y_train)



svr = grid_search_svr.best_estimator_



y_pred_svr = svr.predict(X_test)



rsq_svr = r2_score(y_test, y_pred_svr)



rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
models = [['Linear Regression', rsq_linear, rmse_linear],

          ['Polynomial Regression', rsq_poly, rmse_poly],

          ['Support Vector Regression', rsq_svr, rmse_svr]]



df_comparasion = pd.DataFrame(models, columns = ['Model', 'R²', 'RMSE'])



df_comparasion