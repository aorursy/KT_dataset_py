import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats # QQ plot
from sklearn.preprocessing import OneHotEncoder # Dummy variable
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Set spliting
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import statsmodels.api as sm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

 
sns.set()

%matplotlib inline
dataset = pd.read_csv('../input/fish-market/Fish.csv')
dataset.head()
dataset.rename(columns= {'Species': 'Species', 'Length1':'DimVer', 'Length2':'DimDiag', 'Length3':'DimLong'}, inplace=True)
dataset.describe().T
print(pd.isnull(dataset).sum()) # Is there any missing value?
dataset.groupby('Species').size() # Luego lo veo gráficamente
species = dataset['Species'].value_counts()
species = pd.DataFrame(species) # Creating DF
sns.barplot(x = species.index, y = species['Species'])
plt.title('Species vs. Quantity', fontsize = 20)
plt.xlabel('Species', fontsize = 15)
plt.ylabel('Fish quantity', fontsize = 15)
plt.show()
sns.heatmap(dataset.corr(), annot = True, linewidths=.5, cmap='cubehelix')
plt.title('Correlations', fontsize = 20)
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

ax1.plot(dataset.DimVer, dataset.DimLong, c = 'green')
ax1.set_title('DimVer vs. DimLong', c = 'green')
ax2.scatter(dataset.DimDiag, dataset.DimLong, c='red')
ax2.set_title('DimDiag vs. DimLong', c ='red')

plt.ylabel('DimLong', fontsize = 20)

plt.show()
dataset_2 = dataset.copy() # Copying before dropping columns
dataset_2 = dataset_2.drop(['DimVer','DimDiag'], axis = 1) 

dataset_2 = dataset_2[['Species','DimLong','Height','Width','Weight']] # Rearrange columns.

dataset_2.head(3)
stats.probplot(dataset_2['DimLong'].values, dist="norm", plot=plt)
plt.show()
stats.probplot(dataset_2['Height'].values, dist="norm", plot=plt)
plt.show()
stats.probplot(dataset_2['Width'].values, dist="norm", plot=plt)
plt.show()
sns.boxplot(x=dataset_2['DimLong'], color = 'cyan')
plt.title('DimLong Boxplot', fontsize = 20)
plt.show()
DimLong= dataset_2['DimLong'] # Variable data
DimLong_Q1 = DimLong.quantile(0.25) # Q1 inf limit
DimLong_Q3 = DimLong.quantile(0.75) # Q3 sup limit
DimLong_IQR = DimLong_Q3 - DimLong_Q1 # IQR
DimLong_lowerend = DimLong_Q1 - (1.5 * DimLong_IQR) # q1 - 1.5 * q1
DimLong_upperend = DimLong_Q3 + (1.5 * DimLong_IQR) # q3 + 1.5 * q3

DimLong_outliers = DimLong[(DimLong < DimLong_lowerend) | (DimLong > DimLong_upperend)] # Outlier index
DimLong_outliers
sns.boxplot(x=dataset_2['Height'], color = 'cyan')
plt.title('Height boxplot', fontsize = 20)
plt.show()
sns.boxplot(x=dataset_2['Width'], color = 'cyan')
plt.title('Widht boxplot', fontsize = 20)
plt.show()
sns.boxplot(x=dataset_2['Weight'], color = 'cyan')
plt.title('Weight Boxplot', fontsize = 20)
plt.show()
Peso = dataset_2['Weight'] 
Peso_Q1 = Peso.quantile(0.25) 
Peso_Q3 = Peso.quantile(0.75) 
Peso_IQR = Peso_Q3 - Peso_Q1 
Peso_lowerend = Peso_Q1 - (1.5 * Peso_IQR) 
Peso_upperend = Peso_Q3 + (1.5 * Peso_IQR) 

Peso_outliers = Peso[(Peso < Peso_lowerend) | (Peso > Peso_upperend)] # Outlier Index
Peso_outliers
dataset_2 = dataset_2.drop([142,143,144], axis=0)
dataset_2.describe()
correlaciones = sns.pairplot(dataset_2, hue = "Species", palette = "husl", corner = True)
X = dataset_2.iloc[:,0:4].values 

y = dataset_2.iloc[:,-1].values
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
np.set_printoptions(suppress=True)
X = X[:,1:]
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Set Splitting

regression = LinearRegression() 
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test) # Predictions

Prediccion_y = pd.DataFrame({'Prediction_y': y_pred})
Prediccion_y

Y = pd.DataFrame({'Y test': y_test}) # Df 
Comparacion = Y.join(Prediccion_y) 
Comparacion.head()
Comparacion.plot(kind = 'bar', figsize=(15,15))
plt.grid(which = 'both', linestyle = '-', linewidth = '0.5', color = 'green')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
Adj_r2 = 1 - (1 - metrics.r2_score(y_test, y_pred)) * (156 - 1) / (156 - 9 - 1)
print('R2 adjusted', Adj_r2)
# Adding np one array (intercept)

X = np.append(np.ones((dataset_2.shape[0],1)).astype(int), values = X  , axis = 1) 

def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((X.shape[0], X.shape[1])).astype(int)   # Utilizo X con el vector de 1 agregado. 
    for i in range(0, numVars):        # Bucle para todas las variables
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)   # Me da el máximo valor de p de las variables
        adjR_before = regressor_OLS.rsquared_adj.astype(float)  # Me da el valor del R2 ajustado
        if maxVar > SL:            
            for j in range(0, numVars - i):  # Eliminación de variable si el p valor es mayor al SL pero
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):      # Tiene en cuenta que el R2 ajustado no sea menor que antes                   
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:,:]
X_Modeled = backwardElimination(X_opt, SL)
residuals = Comparacion['Y test'] - Comparacion['Prediction_y']
temp = {'Residuals' : residuals, 'Predicted_y' : Comparacion['Prediction_y']}
residual_df = pd.DataFrame(temp)
residual_df.head()
plt.scatter(residual_df['Residuals'], residual_df['Predicted_y'], c = 'red', s = 25, alpha = 0.8, label = 'Residuals')
plt.title('Residuals vs. Predicted values', fontsize = 20)
plt.xlabel('Residuals', fontsize = 15)
plt.legend()
plt.ylabel('Predicted values', fontsize = 15)
plt.show()
poly_reg = PolynomialFeatures(degree = 2) # Depende del set de datos que tenga.
X_poly = poly_reg.fit_transform(X_train) # Cambia la columna de variables dependientes a polinómicas

lin_reg_2 = LinearRegression() # Creo el modelo de regresión lineal polinómica.
lin_reg_2.fit(X_poly, y_train)
X_test_poly = poly_reg.fit_transform(X_test)
y_pred_pol = lin_reg_2.predict(X_test_poly)
Prediccion_y = pd.DataFrame({'Prediction_y': y_pred_pol})

Y = pd.DataFrame({'Y test': y_test}) # Df 
Comparacion = Y.join(Prediccion_y) 
Comparacion.head()
Comparacion.plot(kind = 'bar', figsize=(15,15))
plt.grid(which = 'both', linestyle = '-', linewidth = '0.5', color = 'green')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_pol))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_pol))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_pol)))
print('R2:', metrics.r2_score(y_test, y_pred_pol))
RF_reg = RandomForestRegressor(n_estimators = 1000, random_state = 123)
RF_reg.fit(X_poly, y_train)

y_pred_rf = RF_reg.predict(X_test_poly)
Prediccion_y = pd.DataFrame({'Prediction_y': y_pred_rf})

Y = pd.DataFrame({'Y test': y_test}) # Df con los valores reales y los predichos
Comparacion = Y.join(Prediccion_y) 
Comparacion.head()
Comparacion.plot(kind = 'bar', figsize=(15,15))
plt.grid(which = 'both', linestyle = '-', linewidth = '0.5', color = 'green')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print('R2:', metrics.r2_score(y_test, y_pred_rf))