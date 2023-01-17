
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
emp_data= pd.read_csv('../input/emp_data.csv')
emp_data
emp_data.describe()
emp_data.plot(x='Salary_hike', y='Churn_out_rate', style='o') 
plt.title('Salaryhike vs Churn_out_rate')  
plt.xlabel('Salary_hike') 
plt.ylabel('Churn_out_rate')
plt.show()
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(emp_data['Salary_hike'], emp_data['Churn_out_rate'])
print('Pearsons correlation: %.3f' % corr)

from scipy.stats import spearmanr
# calculate spearman's correlation
corr, _ = spearmanr(emp_data['Salary_hike'], emp_data['Churn_out_rate'])
print('Spearmans correlation: %.3f' % corr)
import seaborn as sns
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(emp_data['Salary_hike'])
plt.show() 
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(emp_data['Churn_out_rate'])
plt.show()
# Input dataset
X = emp_data['Salary_hike'].values.reshape(-1,1)
print(X)
# Output or Predicted Value of data
y = emp_data['Churn_out_rate'].values.reshape(-1,1)
#print(log(y))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state =42)
predict_reg = LinearRegression()
predict_reg.fit(X_train, y_train)
print(" Intercept value of Model is " ,predict_reg.intercept_)
print("Coefficient value of Model is ", predict_reg.coef_)
y_pred = predict_reg.predict(X_test)
pmsh_pf = pd.DataFrame({'Actual':y_test.flatten(), 'Predict': y_pred.flatten()})
pmsh_pf
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R^2 Score :          ", metrics.r2_score(y_test, y_pred))
# Input dataset
X_log = np.log(emp_data['Salary_hike'].values.reshape(-1,1))

# Output or Predicted Value of data
y_log = emp_data['Churn_out_rate'].values.reshape(-1,1)
X_train_log, X_test_1og, Y_train_log, Y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state= 42)
y_pred_log= LinearRegression()
y_pred_log.fit(X_train_log,Y_train_log)
print(" Intercept value of Model is " ,y_pred_log.intercept_)
print("Co-efficient Value of Log Model is : ", y_pred_log.coef_)
l_model= y_pred_log.predict(X_test_1og)
l_model
pmsh_pf_1 = pd.DataFrame({'Actual':Y_test_log.flatten(), 'Predict': l_model.flatten()})
pmsh_pf_1
plt.scatter(X_test_1og, Y_test_log,  color='gray')
plt.plot(X_test_1og, l_model, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_log, l_model))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_log, l_model) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_log, l_model)))
print("R^2 Score :          ", metrics.r2_score(Y_test_log, l_model))
# Input dataset
X_e_log = emp_data['Salary_hike'].values.reshape(-1,1)

# Output or Predicted Value of data
y_e_log = np.log(emp_data['Churn_out_rate'].values.reshape(-1,1))
X_train_exp, X_test_exp, Y_train_exp, Y_test_exp = train_test_split(X_e_log, y_e_log, test_size=0.2, random_state= 42)
exp_model= LinearRegression()
exp_model.fit(X_train_exp, Y_train_exp)
print(" Exponent Model Intercept value is ", exp_model.intercept_)
print(" Exponent model Coefficient value is ", exp_model.coef_)
exp_model_pred= exp_model.predict(X_test_exp)
exp_model_pred
pmsh_exp = pd.DataFrame({'Actual':Y_test_exp.flatten(), 'Predict': exp_model_pred.flatten()})
pmsh_exp
plt.scatter(X_test_exp, Y_test_exp,  color='gray')
plt.plot(X_test_exp, exp_model_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_exp, exp_model_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_exp, exp_model_pred) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_exp, exp_model_pred)))
print("R^2 Score :          ", metrics.r2_score(Y_test_exp, exp_model_pred))
emp_data['Square_S_hike'] = emp_data.apply(lambda row: row.Salary_hike**2, axis =1 )
emp_data
X_q = emp_data.iloc[:,emp_data.columns != 'Churn_out_rate']
Y_q = emp_data.iloc[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X_q, Y_q, test_size=0.2, random_state= 42)
print(Y_test)

model = LinearRegression()
model.fit(X_train, Y_train)
print(" Intercept value of Model is " ,model.intercept_)

coeff_df = pd.DataFrame(model.coef_,X_q.columns ,columns=['Coefficient'])
print(coeff_df)
y_pred_q_q = model.predict(X_test)
df_qm = pd.DataFrame({'Actual':Y_test, 'Predicted': y_pred_q_q})
df_qm.head()
x_t =np.array(X_test)
plt.scatter(x_t[:,0], Y_test,  color='gray')
plt.plot(x_t[:,0], y_pred_q_q, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred_q_q))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred_q_q) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_q_q)))
print("R^2 Score :          ", metrics.r2_score(Y_test, y_pred_q_q))
emp_data['Cube_S_hike'] = emp_data.apply(lambda row: row.Salary_hike**3, axis =1 )
emp_data
X_c = emp_data.iloc[:,emp_data.columns != 'Churn_out_rate']
Y_c = emp_data.iloc[:,1]
X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X_c, Y_c, test_size=0.2, random_state= 42)
print(Y_test)
cube_model = LinearRegression()
cube_model.fit(X_train_c, Y_train_c)
print(" Intercept value of Cubic Model is " ,cube_model.intercept_)

coeff_df = pd.DataFrame(cube_model.coef_, X_c.columns, columns=['Coefficient'])
print(coeff_df)

y_pred_cube = cube_model.predict(X_test_c)
df_33 = pd.DataFrame({'Actual':Y_test_c, 'Predicted': y_pred_cube})
df_33.head()
x_t_c =np.array(X_test_c)
plt.scatter(x_t_c[:,0], Y_test_c,  color='gray')
plt.plot(x_t_c[:,0], y_pred_cube, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_c, y_pred_cube))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_c, y_pred_cube) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_c, y_pred_cube)))
print("R^2 Score :          ", metrics.r2_score(Y_test_c, y_pred_cube))
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

INPT =[('scale',StandardScaler()), ('polynomial',PolynomialFeatures()), ('model', LinearRegression()) ]
pipe =Pipeline(INPT)
pipe.fit(emp_data[['Salary_hike',  'Square_S_hike', 'Cube_S_hike']], emp_data['Churn_out_rate'])
pred= pipe.predict(emp_data[['Salary_hike',  'Square_S_hike', 'Cube_S_hike']])    
pmsh_exp_1 = pd.DataFrame({'Actual':emp_data['Churn_out_rate'], 'Predict': pred})
pmsh_exp_1
print('Mean Absolute Error:', metrics.mean_absolute_error(emp_data['Churn_out_rate'], pred))  
print('Mean Squared Error:', metrics.mean_squared_error(emp_data['Churn_out_rate'], pred) ) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(emp_data['Churn_out_rate'], pred)))
print("R^2 Score :          ", metrics.r2_score(emp_data['Churn_out_rate'], pred))

