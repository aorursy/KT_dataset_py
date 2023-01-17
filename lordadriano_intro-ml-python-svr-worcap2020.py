from sklearn import datasets

from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVR

import math

from sklearn import metrics
data = datasets.load_boston()

df = pd.DataFrame(data['data'], columns=data['feature_names'])

df['MEDV'] = data['target']

df.head()
df.describe()
plt.figure(figsize=(12, 12))

sns.heatmap(df.corr(), annot=df.corr(), cmap='seismic_r')

plt.show()
price_correlation = df.corr()['MEDV']

columns = list()



for i, corr in enumerate(price_correlation):

    if corr >= 0.5 or corr < -0.5:

        columns.append(df.columns[i])

        print(df.columns[i], '=', corr)

        

new_df = df[columns]

new_df.head()
sns.pairplot(new_df)

plt.show()
x = new_df.iloc[:, 0].values.reshape((-1, 1))

y = new_df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=0)



print('Tamanho do conjunto de dados de treino:%d'%x_train.size)

print('Tamanho do conjunto de dados de teste:%d'%x_test.size)
svr_linear = SVR(kernel='linear').fit(x_train, y_train)

svr_polyd2 = SVR(kernel='poly', degree=2).fit(x_train, y_train)

svr_polyd3 = SVR(kernel='poly', degree=3).fit(x_train, y_train)

svr_rbf = SVR(kernel='rbf').fit(x_train, y_train)
# Obtendo regressão para o conjunto de treinamento e teste

ypred_train_linear = svr_linear.predict(x_train)

ypred_test_linear = svr_linear.predict(x_test)



# Configurando estrutura geral do gráfico

fig, ax = plt.subplots(1, 2, figsize=(16, 6))



# Plotando informações do conjunto de treinamento

ax[0].set_title('SVR com kernel linear\nDados de treinamento', fontdict=dict(fontsize=14))

ax[0].scatter(x_train, y_train, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[0].scatter(x_train, ypred_train_linear, c='red', alpha=.75, label='SVR linear')

ax[0].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[0].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[0].legend()



# Plotando informações do conjunto de teste

ax[1].set_title('SVR com kernel linear\nDados de teste', fontdict=dict(fontsize=14))

ax[1].scatter(x_test, y_test, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[1].scatter(x_test, ypred_test_linear, c='red', alpha=.75, label='SVR linear')

ax[1].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[1].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[1].legend()

plt.show()
# Obtendo regressão para o conjunto de treinamento e teste para kernel polinomial d2

ypred_train_polyd2 = svr_polyd2.predict(x_train)

ypred_test_polyd2 = svr_polyd2.predict(x_test)



# Obtendo regressão para o conjunto de treinamento e teste para kernel polinomial d3

ypred_train_polyd3 = svr_polyd3.predict(x_train)

ypred_test_polyd3 = svr_polyd3.predict(x_test)



# Configurando estrutura geral do gráfico

fig, ax = plt.subplots(1, 2, figsize=(16, 6))



# Plotando informações do conjunto de treinamento

ax[0].set_title('SVR com kernel polinomial\nDados de treinamento', fontdict=dict(fontsize=14))

ax[0].scatter(x_train, y_train, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[0].scatter(x_train, ypred_train_polyd2, c='red', alpha=.75, label='SVR poly grau 2')

ax[0].scatter(x_train, ypred_train_polyd3, c='green', alpha=.75, label='SVR poly grau 3')

ax[0].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[0].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[0].legend()



# Plotando informações do conjunto de teste

ax[1].set_title('SVR com kernel polinomial\nDados de teste', fontdict=dict(fontsize=14))

ax[1].scatter(x_test, y_test, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[1].scatter(x_test, ypred_test_polyd2, c='red', alpha=.75, label='SVR poly grau 2')

ax[1].scatter(x_test, ypred_test_polyd3, c='green', alpha=.75, label='SVR poly grau 3')

ax[1].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[1].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[1].legend()

plt.show()
# Obtendo regressão para o conjunto de treinamento e teste

ypred_train_rbf = svr_rbf.predict(x_train)

ypred_test_rbf = svr_rbf.predict(x_test)



# Configurando estrutura geral do gráfico

fig, ax = plt.subplots(1, 2, figsize=(16, 6))



# Plotando informações do conjunto de treinamento

ax[0].set_title('SVR com kernel RVF\nDados de treinamento', fontdict=dict(fontsize=14))

ax[0].scatter(x_train, y_train, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[0].scatter(x_train, ypred_train_rbf, c='red', alpha=.75, label='SVR RBF')

ax[0].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[0].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[0].legend()



# Plotando informações do conjunto de teste

ax[1].set_title('SVR com kernel RBF\nDados de teste', fontdict=dict(fontsize=14))

ax[1].scatter(x_test, y_test, c='blue', ec='black', s=100, alpha=.75, label='Dados')

ax[1].scatter(x_test, ypred_test_rbf, c='red', alpha=.75, label='SVR RBF')

ax[1].set_xlabel('RM', fontdict=dict(fontsize=12))

ax[1].set_ylabel('MEDV', fontdict=dict(fontsize=12))

ax[1].legend()

plt.show()
# Configurando estrutura geral do gráfico

fig, ax = plt.subplots(1, 3, figsize=(17, 5))



# Monstrando comparação entre os valores reais e o previsto SVR de kernel linear

ax[0].set_title('Compararação SVR linear')

ax[0].scatter(y_test, ypred_test_linear, color='k', edgecolor='k', alpha=.75)

ax[0].set_xlabel('Preço real')

ax[0].set_ylabel('Preço previsto')

ax[0].plot([min(y_test), max(y_test)], [min(ypred_test_linear), max(ypred_test_linear)], label='Diagonal secundária', linestyle='--', alpha=.75)

ax[0].legend()



# Monstrando comparação entre os valores reais e o previsto SVR de kernel polinomial

ax[1].set_title('Compararação SVR polinomial')

ax[1].scatter(y_test, ypred_test_polyd3, color='k', edgecolor='k', alpha=.75)

ax[1].set_xlabel('Preço real')

ax[1].set_ylabel('Preço previsto')

ax[1].plot([min(y_test), max(y_test)], [min(ypred_test_polyd3), max(ypred_test_polyd3)], label='Diagonal secundária', linestyle='--', alpha=.75)

ax[1].legend()



# Monstrando comparação entre os valores reais e o previsto SVR de kernel RBF

ax[2].set_title('Compararação SVR polinomial')

ax[2].scatter(y_test, ypred_test_rbf, color='k', edgecolor='k', alpha=.75)

ax[2].set_xlabel('Preço real')

ax[2].set_ylabel('Preço previsto')

ax[2].plot([min(y_test), max(y_test)], [min(ypred_test_rbf), max(ypred_test_rbf)], label='Diagonal secundária', linestyle='--', alpha=.75)

ax[2].legend()



plt.show()
# Configurando estrutura geral do gráfico

fig, ax = plt.subplots(1, 3, figsize=(17, 5))



error_linear = y_test-ypred_test_linear

error_poly3d = y_test-ypred_test_polyd3

error_rbf = y_test-ypred_test_rbf



# Monstrando frequência dos erros entre os valores reais e o previsto SVR de kernel linear

ax[0].set_title('Compararação SVR linear')

ax[0].hist(error_linear, color='black')

ax[0].set_xlabel('Erro entre valor real e previsto')

ax[0].set_ylabel('Ocorrências')



# Monstrando frequência dos erros entre os valores reais e o previsto SVR de kernel linear

ax[1].set_title('Compararação SVR polinomial grau 3')

ax[1].hist(error_poly3d, color='black')

ax[1].set_xlabel('Erro entre valor real e previsto')

ax[1].set_ylabel('Ocorrências')



# Monstrando frequência dos erros entre os valores reais e o previsto SVR de kernel linear

ax[2].set_title('Compararação SVR RBF')

ax[2].hist(error_rbf, color='black')

ax[2].set_xlabel('Erro entre valor real e previsto')

ax[2].set_ylabel('Ocorrências')



plt.show()
kerners = ['Linear', 'Poli grau 2', 'Poli grau 3', 'RBF']



# Calculando erro médio absoluto para todos os modelos

mae_linear = metrics.mean_absolute_error(y_test, ypred_test_linear)

mae_poly2d = metrics.mean_absolute_error(y_test, ypred_test_polyd2)

mae_poly3d = metrics.mean_absolute_error(y_test, ypred_test_polyd3)

mae_rbf = metrics.mean_absolute_error(y_test, ypred_test_rbf)

mae_kernels = [mae_linear, mae_poly2d, mae_poly3d, mae_rbf]



# Calculando erro médio quadrático para todos os modelos

mse_linear = metrics.mean_squared_error(y_test, ypred_test_linear)

mse_poly2d = metrics.mean_squared_error(y_test, ypred_test_polyd2)

mse_poly3d = metrics.mean_squared_error(y_test, ypred_test_polyd3)

mse_rbf = metrics.mean_squared_error(y_test, ypred_test_rbf)

mse_kernels = [mse_linear, mse_poly2d, mse_poly3d, mse_rbf]



# Calculando raiz do erro médio quadrático para todos os modelos

rmse_linear = math.sqrt(metrics.mean_absolute_error(y_test, ypred_test_linear))

rmse_poly2d = math.sqrt(metrics.mean_absolute_error(y_test, ypred_test_polyd2))

rmse_poly3d = math.sqrt(metrics.mean_absolute_error(y_test, ypred_test_polyd3))

rmse_rbf = math.sqrt(metrics.mean_absolute_error(y_test, ypred_test_rbf))

rmse_kernels = [rmse_linear, rmse_poly2d, rmse_poly3d, rmse_rbf]



fig, ax = plt.subplots(1, 3, figsize=(20, 5))



ax[0].set_title('Erro médio absoluto')

ax[0].bar(kerners, mae_kernels, color='black')

ax[0].set_xlabel('Kernel')

ax[0].set_ylabel('MAE')



ax[1].set_title('Erro médio quadrático')

ax[1].bar(kerners, mse_kernels, color='black')

ax[1].set_xlabel('Kernel')

ax[1].set_ylabel('MSE')



ax[2].set_title('Raiz do erro médio quadrático')

ax[2].bar(kerners, rmse_kernels, color='black')

ax[2].set_xlabel('Kernel')

ax[2].set_ylabel('RMSE')

plt.show()