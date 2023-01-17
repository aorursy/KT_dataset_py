import pandas as pd



#Gráfico

import seaborn as sns

import matplotlib.pyplot as plt



# ML

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import  DecisionTreeRegressor

from sklearn.metrics import r2_score #método para o cálculo do R2

from sklearn.metrics import mean_squared_error #erro absoluto
pnad = pd.read_csv('../input/testes/dados.csv')

pnad.head()
pnad.describe()
pnad.isnull().sum()
#removendo valores não significativos para análise 

pnad.drop(columns= ['UF', 'Sexo', 'Cor', 'Altura'], inplace= True)

pnad.head()
pnad.boxplot(['Renda'])
plt.bar( pnad['Idade'], pnad['Renda'])

plt.xlabel('Idade')

plt.ylabel('Renda')

plt.title('Idade x Renda')

plt.show()
plt.bar( pnad['Anos de Estudo'], pnad['Renda'])

plt.xlabel('Anos de Estudo')

plt.ylabel('Renda')

plt.title('Anos de Estudo x Renda')

plt.show()
#Correlação 

corr = pnad.corr()

corr.style.background_gradient()
x = pnad.iloc[:,:-1].values

#x_res = x.reshape((-1,1))

y = pnad.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
#treinamento

reg_lg = LinearRegression()

reg_lg.fit(x_train,y_train)
#previsão

pred_lg = reg_lg.predict(x_test)

print('Y = {}X {}'.format(reg_lg.coef_,reg_lg.intercept_))
#métricas para dados de TEST 

R2_reg = r2_score(y_test, pred_lg)  #realiza o cálculo do R2



print("Coeficiente de Determinação (R2):", R2_reg)



MSE_reg = mean_squared_error(y_test,pred_lg) # encontra o MSE através do sklearn

print('MSE: ', MSE_reg) 
#métricas para dados de TREINAMENTO 



pred_lg2 = reg_lg.predict(x_train)

R2_reg2 = r2_score(y_train, pred_lg2)  #realiza o cálculo do R2



print("Coeficiente de Determinação (R2):", R2_reg2)



MSE_reg2 = mean_squared_error(y_train,pred_lg2) # encontra o MSE através do sklearn

print('MSE: ', MSE_reg2) 
#treinando 

reg_dt = DecisionTreeRegressor() # Cria objeto DecisionTreeRegressor

reg_dt.fit(x_train,y_train)

#previsão

pred_dt= reg_dt.predict(x_test)

print(pred_dt)
#métricas para dados de TESTE

R2_dt= r2_score(y_test,pred_dt)

print("Coeficiente de Determinação (R2):", R2_dt)



MSE_dt = mean_squared_error(y_test,pred_dt) # encontra o MSE através do sklearn

print('MSE: ', MSE_dt)
#métricas para dados de TREINAMENTO 



pred_dt2 = reg_dt.predict(x_train)

R2_dt2 = r2_score(y_train, pred_dt2)  #realiza o cálculo do R2

print("Coeficiente de Determinação (R2):", R2_dt2)



MSE_dt2 = mean_squared_error(y_train,pred_dt2) # encontra o MSE através do sklearn

print('MSE: ', MSE_reg2) 