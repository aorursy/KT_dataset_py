from sklearn.metrics import mean_squared_error 
  
# Given values 
Y_true = [1,1,2,2,4]  # Y_true = Y (original values) 
  
# calculated values 
Y_pred = [0.6,1.29,1.99,2.69,3.4]  # Y_pred = Y' 
  
# Calculation of Mean Squared Error (MSE) 
mean_squared_error(Y_true,Y_pred)
from sklearn.metrics import mean_absolute_error 
  
# Given values 
Y_true = [1,1,2,2,4]  # Y_true = Y (original values) 
  
# calculated values 
Y_pred = [0.6,1.29,1.99,2.69,3.4]  # Y_pred = Y' 
  
# Calculation of Mean Squared Error (MSE) 
mean_absolute_error(Y_true,Y_pred)