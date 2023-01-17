import pandas as pd

import numpy as np



import plotly.graph_objects as go



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')



data.head()
X = data.drop('Salary', axis=1)



y = data['Salary']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
simple_linear_reg = LinearRegression()



simple_linear_reg.fit(X_train, y_train)



y_pred = simple_linear_reg.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)



rsq = r2_score(y_test,y_pred)



ajusted_rsq = 1 - ((1 - rsq) * (len(y_test) - 1))/(len(y_test)- X_test.shape[1] - 1)



print('MAE: %.2f' %mae)



print('\nR²: %.2f' %rsq)



print('\nAjusted R²: %.2f' %ajusted_rsq)
coef_a = simple_linear_reg.coef_[0]



coef_b =  simple_linear_reg.intercept_



print('Estimated model: y = %.2fx + %.2f' %(coef_a, coef_b))
fig_model = go.Figure()



fig_model.add_trace(go.Scatter(x = X_test['YearsExperience'], 

                               y = y_test, 

                               mode = 'markers',

                               name = 'Actual Values'))



fig_model.add_trace(go.Scatter(x = X_test['YearsExperience'], 

                               y = y_pred, 

                               mode = 'lines',

                               name = 'Predictions'))



fig_model.update_layout(

    title_text = ('Linear Regression Model (R² = %.2f)' %rsq),

    xaxis_title_text='Years Of Experience',

    yaxis_title_text='Salary', 

    template = 'plotly_dark',

    width=750, 

    height=600

)