# imports
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
# configs
mpld3.enable_notebook()
warnings.filterwarnings("ignore")
# read csvs & build dataframe
df = pd.read_csv('../input/2-years-restaurant-sale-with-multiple-external-var/sales_one_hot_encoding.csv', ',')
df = df.rename(columns={'IS_SEMANA_PAGAMENTO': 'SEMANA_PAGAMENTO'})
df = df.query('VENDAS < 225')
df.info()
df.describe()
df.median()
print(df.isnull().values.any())
print(df.isna().values.any())
import mpld3
import matplotlib.style as style
import matplotlib.pyplot as plt
mpld3.enable_notebook()

df.plot(y=['VENDAS', 'TEMPERATURA'], x='DATA', figsize=(14, 5), linestyle='solid', linewidth=1, markersize=1)
plt.legend(loc=1)
plt.title('AMOSTRAGEM DE TODOS OS DIAS ABERTOS ENTRE VENDAS E TEMPERATURA')
plt.xlabel('DATA')
plt.ylabel('VENDAS')
plt.show()
df.plot(y='VENDAS', x='DATA', figsize=(14, 5), kind='box')
plt.title('AMOSTRAGEM DA MEDIANA DE TODOS OS DIAS ABERTOS')
plt.show()
df.plot(y='VENDAS', x='DATA', figsize=(14, 5), kind='kde')
plt.legend(loc=1)
plt.title('ESTIMATIVA DE DENSIDADE DE KERNEL DE VENDAS')
plt.xlabel('VENDAS')
plt.ylabel('DENSIDADE')
plt.show()
query = (df['DATA'] >= "2019-06-01") & (df['DATA'] <= "2019-06-30")
df2 = df.loc[query]
df2.sort_values(by=['DATA'], inplace=True, ascending=True)
df2.plot(y=['VENDAS', 'PRECIPITACAO'], x='DATA', figsize=(14, 5), linestyle='solid', linewidth=1, markersize=1)
plt.legend(loc=3)
plt.title('AMOSTRAGEM DO MÊS DE MAIO DE 2019')
plt.xlabel('DATA')
plt.ylabel('VENDAS')
plt.show()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# featured engineering
X = df.drop(columns=['DATA', 'VENDAS'])
y = df.drop(columns=['DATA', 'FDS', 'DS', 'DATA_FESTIVA', 'VESPERA_DATA_FESTIVA', 'POS_DATA_FESTIVA', 'DATA_NAO_FESTIVA', 'FERIADO', 'NAO_FERIADO', 'SEMANA_PAGAMENTO', 'SEMANA_DE_NAO_PAGAMENTO', 'BAIXA_TEMPORADA', 'ALTA_TEMPORADA', 'QTD_CONCORRENTES', 'PRECIPITACAO', 'TEMPERATURA', 'UMIDADE'])

# shuffled and splitted into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

#feature scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use('ggplot')

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=['FDS', 'DS', 'DATA_FESTIVA', 'VESPERA_DATA_FESTIVA', 'POS_DATA_FESTIVA', 'DATA_NAO_FESTIVA', 'FERIADO', 'NAO_FERIADO', 'SEMANA_PAGAMENTO', 'SEMANA_DE_NAO_PAGAMENTO', 'BAIXA_TEMPORADA', 'ALTA_TEMPORADA', 'QTD_CONCORRENTES', 'PRECIPITACAO', 'TEMPERATURA', 'UMIDADE'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5))

sns.kdeplot(df['FDS'], bw=1.5, ax=ax1)
sns.kdeplot(df['DS'], bw=1.5, ax=ax1)
sns.kdeplot(df['DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['VESPERA_DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['POS_DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['DATA_NAO_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['FERIADO'], bw=1.5, ax=ax1)
sns.kdeplot(df['NAO_FERIADO'], bw=1.5, ax=ax1)
sns.kdeplot(df['SEMANA_PAGAMENTO'], bw=1.5, ax=ax1)
sns.kdeplot(df['SEMANA_DE_NAO_PAGAMENTO'], bw=1.5, ax=ax1)
sns.kdeplot(df['BAIXA_TEMPORADA'], bw=1.5, ax=ax1)
sns.kdeplot(df['ALTA_TEMPORADA'], bw=1.5, ax=ax1)
sns.kdeplot(df['QTD_CONCORRENTES'], bw=1.5, ax=ax1)
sns.kdeplot(df['PRECIPITACAO'], bw=1.5, ax=ax1)
sns.kdeplot(df['TEMPERATURA'], bw=1.5, ax=ax1)
sns.kdeplot(df['UMIDADE'], bw=1.5, ax=ax1)

sns.kdeplot(scaled_df['FDS'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DS'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VESPERA_DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['POS_DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DATA_NAO_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['FERIADO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['NAO_FERIADO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['SEMANA_PAGAMENTO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['SEMANA_DE_NAO_PAGAMENTO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['BAIXA_TEMPORADA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['ALTA_TEMPORADA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['QTD_CONCORRENTES'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['PRECIPITACAO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['TEMPERATURA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['UMIDADE'], bw=1.5, ax=ax2)

plt.show()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics

gb = GradientBoostingRegressor(random_state=0, n_estimators=6000, max_depth=20)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
scores = cross_val_score(gb, X_test, y_test, cv=10)
df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
print(f'Explained Variance Regression: {metrics.explained_variance_score(y_test, y_pred)}')
print(f'Max Error: {metrics.max_error(y_test, y_pred)}')
print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')

importances = pd.Series(gb.feature_importances_).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(X.columns.values, importances, align='center')
ax.set_yticks(X.columns.values)
ax.set_yticklabels(X.columns.values)
ax.set_title('Feature Selection - GradientBoostingRegressor')
ax.margins(tight=True)  
plt.tick_params(axis='y', labelsize=12, direction='in', grid_linewidth=2)
plt.show()
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

tree = DecisionTreeRegressor(min_samples_split=12, max_depth=600, ccp_alpha=0.0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
scores = cross_val_score(tree, X_test, y_test, cv=10)
df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
print(f'Explained Variance Regression: {metrics.explained_variance_score(y_test, y_pred)}')
print(f'Max Error: {metrics.max_error(y_test, y_pred)}')
print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')

importances = pd.Series(tree.feature_importances_).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(X.columns.values, importances, align='center')
ax.set_yticks(X.columns.values)
ax.set_yticklabels(X.columns.values)
ax.set_title('Feature Selection - DecisionTreeRegressor')
ax.margins(tight=True)  
plt.tick_params(axis='y', labelsize=12, direction='in', grid_linewidth=2)
plt.show()
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

tree2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=800), n_estimators=1000, random_state=0, loss='square')
tree2.fit(X_train, y_train)
y_pred = tree2.predict(X_test)
scores = cross_val_score(tree2, X_test, y_test, cv=10)
df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
print(f'Explained Variance Regression: {metrics.explained_variance_score(y_test, y_pred)}')
print(f'Max Error: {metrics.max_error(y_test, y_pred)}')
print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')

importances = pd.Series(tree2.feature_importances_).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(X.columns.values, importances, align='center')
ax.set_yticks(X.columns.values)
ax.set_yticklabels(X.columns.values)
ax.set_title('Feature Selection - AdaBoostRegressor / DecisionTreeRegressor')
ax.margins(tight=True)  
plt.tick_params(axis='y', labelsize=12, direction='in', grid_linewidth=2)
plt.show()
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

import eli5
from eli5.sklearn import PermutationImportance

nn = MLPRegressor(random_state=2, activation='logistic', hidden_layer_sizes=37, max_iter=9000)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
scores = cross_val_score(nn, X_test, y_test, cv=10)
df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
print(f'Explained Variance Regression: {metrics.explained_variance_score(y_test, y_pred)}')
print(f'Max Error: {metrics.max_error(y_test, y_pred)}')
print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')

perm = PermutationImportance(nn, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=X.columns.values.tolist())