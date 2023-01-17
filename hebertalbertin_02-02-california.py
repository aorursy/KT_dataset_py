# 203142 André Felício de Sousa 
# 203143 Hebert Francisco Albertin 
# 203011 Lucas Francisco de Camargo 
# 203214 Marcelo Nogueira da Silva 
# 203144 Murilo Spinoza de Arruda 
# 191515 Rodrigo Lopes
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
houses_data = pd.read_csv("../input/housing-raw/housing_raw.csv")
houses_data.head(20)
houses_data.describe()
import seaborn as sns

for column in houses_data.columns:
    if houses_data[column].dtype == np.float64:
        plt.figure(figsize = (20, 3))
        ax = sns.boxplot(x = houses_data[column])
houses_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=houses_data["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
mapa = plt.legend()
# Visualizando dados categóricos
houses_data['ocean_proximity'].value_counts()
# Analisando a proximidade do Oceano com os valores
plt.figure(figsize=(10,6))
sns.boxplot(data=houses_data,x='ocean_proximity',y='median_house_value',palette='viridis')
plt.plot()
plt.figure(figsize = (20, 3))
ax = sns.boxplot(x = houses_data['housing_median_age'])
# Através dos boxplots, verificamos um valor máximo que a coluna pode assumir
selRows = houses_data[houses_data['housing_median_age'] > 100].index
houses_data = houses_data.drop(selRows, axis=0)
plt.figure(figsize = (20, 3))
ax = sns.boxplot(x = houses_data['median_house_value'])
# Verificar valores máximos
houses_data[houses_data['median_house_value']>450000]['median_house_value'].value_counts().head()
# Manter os dados abaixo do valor teto que encontramos
houses_data=houses_data.loc[houses_data['median_house_value']<500001,:]
plt.figure(figsize = (20, 3))
ax = sns.boxplot(x = houses_data['population'])
# Vimos os outliers a partir de 14000 de acordo com o boxplot
houses_data[houses_data['population']>14000]
# Manter os dados removendo os que encontramos
houses_data=houses_data[houses_data['population']<14000]
atributos_missing = []

for f in houses_data.columns:
    if f != 'ocean_proximity':
        missings = len(houses_data[np.isnan(houses_data[f])][f])
        if missings > 0:
            atributos_missing.append(f)
            missings_perc = missings/houses_data.shape[0]

            print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))
        
print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
# Atribuir aos NaN, a proporção da média de quartos em relação aos cômodos
# Optamos por essa estratégia, pois se atribuirmos apenas a média de toda a coluna, podemos assumir o erro de que haverão mais quartos do que cômodos
proporcao = houses_data['total_bedrooms'].mean() / houses_data['total_rooms'].mean()

houses_data.loc[np.isnan(houses_data['total_bedrooms']), 'total_bedrooms'] = (houses_data[np.isnan(houses_data['total_bedrooms'])]['total_rooms'] * proporcao).astype(int)
# Remover dados duplicados
print('Antes:', houses_data.shape)
houses_data = houses_data.drop_duplicates()
print('Depois:', houses_data.shape)
houses_data[houses_data['total_bedrooms'] >= houses_data['total_rooms']]
# Transformando em dummies
houses_data['<1H OCEAN'] = (houses_data['ocean_proximity'] == '<1H OCEAN').astype(int)
houses_data['INLAND'] = (houses_data['ocean_proximity'] == 'INLAND').astype(int)
houses_data['NEAR OCEAN'] = (houses_data['ocean_proximity'] == 'NEAR OCEAN').astype(int)
houses_data['NEAR BAY'] = (houses_data['ocean_proximity'] == 'NEAR BAY').astype(int)
houses_data['ISLAND'] = (houses_data['ocean_proximity'] == 'ISLAND').astype(int)
houses_data
# Dropando a coluna "ocean_proximity" após a criação das novas colunas dummies
houses_data = houses_data.drop('ocean_proximity', axis=1)
import seaborn as sns

sns.set(rc={'figure.figsize':(14,12)})

corr = houses_data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontdict={'fontsize': 17},
    horizontalalignment='right'
);
ax.set_yticklabels(
    ax.get_yticklabels(),
    fontdict={'fontsize': 16},
);
# Correlação na nossa dependente
corr['median_house_value'].abs().sort_values(ascending=False)
import statsmodels.api as sm

# Rodamos OLS e fomos chegando aos resultados:
# Primeiro: removemos a coluna housing_median_age
# Segundo: removemos a coluna ISLAND
# Terceiro: removemos as colunas latitude e longitude
variaveis = ['INLAND', '<1H OCEAN', 'NEAR BAY', 'total_rooms', 'NEAR OCEAN', 'households', 'median_income', 'total_bedrooms', 'population']
dependente = 'median_house_value'

y = houses_data[dependente]

for i in range(len(variaveis), 0, -1):

    X = houses_data[variaveis[0:i]]

    model = sm.OLS(y, X).fit()
    print(model.summary())
    print('\n\n\n----------------\n\n\n')
import statistics
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Amostragem aleatória
R = 300

# Variável independente
X = houses_data[variaveis]

# Variável dependente
y = houses_data['median_house_value']

linearRegressor = LinearRegression(fit_intercept=False)

mae = []
mse = []
rmse = []
result = []
model = []

for i in range(1,R):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)
    
    #Regressão Linear
    model.append(linearRegressor.fit(X_train, y_train))
    y_pred = linearRegressor.predict(X_test)
    
    mae.append(metrics.mean_absolute_error(y_pred, y_test))
    mse.append(metrics.mean_squared_error(y_pred, y_test))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))
    
print("MAE medio: ", np.mean(mae), " MAE desv pad: ", np.sqrt(np.var(mae)))
print("MSE medio: ", np.mean(mse), " MSE desv pad: ", np.sqrt(np.var(mse)))
print("RMSE medio: ", np.mean(rmse), " RMSE desv pad: ", np.sqrt(np.var(rmse)))
print("Intercepto ou Coeficiente Linear: ", linearRegressor.intercept_)
print("Coeficiente Angular (slope):", linearRegressor.coef_)
houses_data.head(5)
from sklearn.preprocessing import StandardScaler

# Aplicando a padronização de escala
scaler = StandardScaler()
scaler.fit(houses_data)
data_s = scaler.transform(houses_data)

df = pd.DataFrame(data_s, columns = houses_data.columns)
df.head(5)
# Separando em teste e treino
X = df.drop(['median_house_value'],axis=1)
y = df.median_house_value

from sklearn.model_selection import train_test_split
feature_train, feature_test,label_train, label_test = train_test_split(X, y, test_size=1/5, random_state=42)
# Rodamos com base nos OLSs anteriores
variaveis = ['INLAND', '<1H OCEAN', 'NEAR BAY', 'total_rooms', 'NEAR OCEAN', 'households', 'median_income', 'total_bedrooms', 'population']
dependente = 'median_house_value'

y = df[dependente]

for i in range(len(variaveis), 0, -1):

    X = df[variaveis[0:i]]

    model = sm.OLS(y, X).fit()
    print(model.summary())
    print('\n\n\n----------------\n\n\n')
# Amostragem aleatória
R = 300

# Variável independente
X = df[variaveis]

# Variável dependente
y = df['median_house_value']

linearRegressor = LinearRegression(fit_intercept=False)

mae = []
mse = []
rmse = []
result = []
model = []

for i in range(1,R):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)
    
    #Regressão Linear
    model.append(linearRegressor.fit(X_train, y_train))
    y_pred = linearRegressor.predict(X_test)
    
    mae.append(metrics.mean_absolute_error(y_pred, y_test))
    mse.append(metrics.mean_squared_error(y_pred, y_test))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))
    
print("MAE medio: ", np.mean(mae), " MAE desv pad: ", np.sqrt(np.var(mae)))
print("MSE medio: ", np.mean(mse), " MSE desv pad: ", np.sqrt(np.var(mse)))
print("RMSE medio: ", np.mean(rmse), " RMSE desv pad: ", np.sqrt(np.var(rmse)))
print("Intercepto ou Coeficiente Linear: ", linearRegressor.intercept_)
print("Coeficiente Angular (slope):", linearRegressor.coef_)
