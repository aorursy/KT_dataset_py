import numpy as np
import matplotlib.pyplot as plt

# Valores conhecidos
idades = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
custo_plano_saude = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

%matplotlib inline
plt.scatter(idades, custo_plano_saude)
plt.title('Custo do Plano de Saúde X Idade')
plt.xlabel('Idade')
plt.ylabel('Custo')
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(idades, custo_plano_saude)
# Calcular o custo previsto para a idade de 40 anos
# y = b0 + b1 * x1 (equação da reta)

# constante
b0 = regression.intercept_
# coeficiente
b1 = regression.coef_

idade = 40

# fazendo o cálculo manualmente
custo_previsto = b0 + b1 * idade
print('Custo previsto para 40 anos: ', custo_previsto)

# fazendo o cálculo com sklearn
custo_previsto_sklearn = regression.predict([[40]])
print('Custo previsto para 40 anos: ', custo_previsto_sklearn)
# Calcular o custo previsto para todas as idades já conhecidas e então verificar a margem de erro

previsoes = regression.predict(idades)

# MAE calculado manualmente
mae_manual = abs(custo_plano_saude - previsoes).mean()
print('mae_manual: ', mae_manual)

# MAE calculado pelo sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_sklearn = mean_absolute_error(custo_plano_saude, previsoes)
print('mae_sklearn: ', mae_sklearn)
mse_sklearn = mean_squared_error(custo_plano_saude, previsoes)
print('mse_sklearn: ', mse_sklearn)
plt.plot(idades, custo_plano_saude, 'o')
plt.plot(idades, previsoes, color='Red')
plt.title('Previsão de custo do plano de saúde por idade')
plt.xlabel('Idade')
plt.ylabel('Custo')