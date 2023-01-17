%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
houses_data = pd.read_csv("../input/housing_raw.csv")
houses_data.head(20)
print('Antes:', houses_data.shape)
houses_data = houses_data.drop_duplicates()
print('Depois:', houses_data.shape)
houses_data.describe()
import seaborn as sns

for column in houses_data.columns:
    if houses_data[column].dtype == np.float64:
        plt.figure(figsize = (20, 3))
        ax = sns.boxplot(x = houses_data[column])

import numpy as np

houses_data[np.isnan(houses_data['total_bedrooms'])]
houses_data['ocean_proximity'].value_counts()
houses_data['<1H OCEAN'] = (houses_data['ocean_proximity'] == '<1H OCEAN').astype(int)
houses_data['INLAND'] = (houses_data['ocean_proximity'] == 'INLAND').astype(int)
houses_data['NEAR OCEAN'] = (houses_data['ocean_proximity'] == 'NEAR OCEAN').astype(int)
houses_data['NEAR BAY'] = (houses_data['ocean_proximity'] == 'NEAR BAY').astype(int)
houses_data['ISLAND'] = (houses_data['ocean_proximity'] == 'ISLAND').astype(int)

# houses_data = houses_data.drop('ocean_proximity', axis=1)
houses_data
atributos_missing = []

for f in houses_data.columns:
    if f != 'ocean_proximity':
        missings = len(houses_data[np.isnan(houses_data[f])][f])
        if missings > 0:
            atributos_missing.append(f)
            missings_perc = missings/houses_data.shape[0]

            print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))
        
print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
houses_data.describe()
proporcao = houses_data['total_bedrooms'].mean() / houses_data['total_rooms'].mean()

houses_data.loc[np.isnan(houses_data['total_bedrooms']), 'total_bedrooms'] = (houses_data[np.isnan(houses_data['total_bedrooms'])]['total_rooms'] * proporcao).astype(int)
houses_data[290:]
for column in houses_data.columns:
    if houses_data[column].dtype == np.float64:
        plt.figure(figsize = (20, 3))
        ax = sns.boxplot(x = houses_data[column])
# dropar outlier housing median age
selRows = houses_data[houses_data['housing_median_age'] > 100].index
houses_data = houses_data.drop(selRows, axis=0)
houses_data[houses_data['housing_median_age'] > 100]
houses_data[houses_data['total_bedrooms'] >= houses_data['total_rooms']]
houses_data[houses_data['median_income'] > 5000]
# remover outliers 500001
houses_data[houses_data['median_house_value']>450000]['median_house_value'].value_counts().head()
#remover esses outliers
houses_data=houses_data.loc[houses_data['median_house_value']<500001,:]

houses_data=houses_data[houses_data['population']<14000]
houses_data.describe()
#boxplot of house value on ocean_proximity categories
plt.figure(figsize=(10,6))
sns.boxplot(data=houses_data,x='ocean_proximity',y='median_house_value',palette='viridis')
plt.plot()
# analise de dados faltantes
houses_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=houses_data["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
#drop string
houses_data = houses_data.drop('ocean_proximity', axis=1)
# from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt

# scaler = StandardScaler()
# scaler.fit(houses_data)
# data_s = scaler.transform(houses_data)

# df = pd.DataFrame(data_s, columns = houses_data.columns)
# df.describe()
#boxplot of house value on ocean_proximity categories
plt.figure(figsize=(10,6))
sns.boxplot(data=houses_data,x='ISLAND',y='median_house_value',palette='viridis')
plt.plot()
corr['median_house_value'].abs().sort_values(ascending=False)
# melhorar com o heatmap igual de estatistica
import seaborn as sns

sns.set(rc={'figure.figsize':(12,10)})

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
#     rotation=45,
    fontdict={'fontsize': 16},
#     horizontalalignment='right'
);
import statsmodels.api as sm
# Remover housing_median_age
# Remover ISLAND
variaveis = ['INLAND', '<1H OCEAN', 'NEAR BAY', 'total_rooms', 'NEAR OCEAN', 'households', 'median_income', 'total_bedrooms', 'population']
dependente = 'median_house_value'

y = houses_data[dependente]

# for i in range(1, len(variaveis)+1):
for i in range(len(variaveis), len(variaveis)+1):

    X = houses_data[variaveis[0:i]]
#     X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())
    print('\n\n\n----------------\n\n\n')
import statsmodels.api as sm
# Remover housing_median_age
# Remover ISLAND
variaveis = ['INLAND', '<1H OCEAN', 'NEAR BAY', 'total_rooms', 'NEAR OCEAN', 'households', 'median_income', 'total_bedrooms', 'population']
dependente = 'median_house_value'


from sklearn import preprocessing
convert = preprocessing.StandardScaler() 

feature = houses_data.drop(['median_house_value'], axis=1)
label = houses_data.median_house_value

X = convert.fit_transform(feature[variaveis].values)
y = convert.fit_transform(houses_data.median_house_value.values.reshape(-1,1)).flatten()

# y = houses_data[dependente]

# for i in range(1, len(variaveis)+1):
# X = houses_data[variaveis[0:i]]
#     X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
print('\n\n\n----------------\n\n\n')
import statistics
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

R = 300

# Variável independente
X = houses_data[variaveis]

# Variável dependente
y = houses_data['median_house_value']

linearRegressor = LinearRegression()

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

# Original
houses_data_original = pd.read_csv("../input/housing_raw.csv")

# proporcao = houses_data_original['total_bedrooms'].mean() / houses_data_original['total_rooms'].mean()

# houses_data_original.loc[np.isnan(houses_data_original['total_bedrooms']), 'total_bedrooms'] = (houses_data_original[np.isnan(houses_data_original['total_bedrooms'])]['total_rooms'] * proporcao).astype(int)

selRows = houses_data_original[np.isnan(houses_data_original['total_bedrooms'])].index
houses_data_original = houses_data_original.drop(selRows, axis=0)

variaveis_originais = ['total_rooms', 'households', 'median_income', 'housing_median_age', 'total_bedrooms', 'population']
dependente = 'median_house_value'


R = 300

# Variável independente
X_original = houses_data_original[variaveis_originais]

# Variável dependente
y_original = houses_data_original['median_house_value']

linearRegressor = LinearRegression()

mae = []
mse = []
rmse = []
result = []
model = []

for i in range(1,R):
    X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=1/5)
    
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
y = houses_data_original[dependente]


X = houses_data_original[variaveis_originais]
# X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
print('\n\n\n----------------\n\n\n')
from sklearn import model_selection
from sklearn import linear_model

# LassoCV
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)


model = linear_model.LassoCV(n_alphas=10)
model.fit(X_train, y_train)


erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train), squared=False)
print('LassoCV RMSE no treino:', erro_treino)

erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test), squared=False)
print('LassoCV RMSE no teste:', erro_teste)


print('\n\n')


# RidgeCV
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

alphas = np.logspace(0, 1, 50)
model = linear_model.RidgeCV(alphas = alphas)
model.fit(X_train, y_train)

erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train), squared=False)
print('RidgeCV RMSE no treino:', erro_treino)

erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test), squared=False)
print('RidgeCV RMSE no teste:', erro_teste)
