import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import seaborn as sns
df = pd.read_csv('../input/testes/dados.csv')
df.head()
df.shape
# there are 76840 rows and 7 columns on the dataframe (df)
df.isnull().values.any()
df['Cor'].unique()
df.dtypes
# How many categories are present in each column?

for col in df.columns:
  print(col, " :", len(df[col].unique()))
Dict_UF = {
    11 : 'Rondônia',
    12 : 'Acre',
    13 : 'Amazonas',
    14 : 'Roraima',
    15 : 'Pará',
    16 : 'Amapá',
    17 : 'Tocantins',
    21 : 'Maranhão',
    22 : 'Piauí',
    23 : 'Ceará',
    24 : 'Rio Grande do Norte',
    25 : 'Paraíba',
    26 : 'Pernambuco',
    27 : 'Alagoas',
    28 : 'Sergipe',
    29 : 'Bahia',
    31 : 'Minas Gerais',
    32 : 'Espírito Santo',
    33 : 'Rio de Janeiro',
    35 : 'São Paulo',
    41 : 'Paraná',
    42 : 'Santa Catarina',
    43 : 'Rio Grande do Sul',
    50 : 'Mato Grosso do Sul',
    51 : 'Mato Grosso',
    52 : 'Goiás',
    53 : 'Distrito Federal'
}
df["UF"] = df["UF"].map(Dict_UF)
# Verifying changes made
df.loc[2000:2010]
Dict_Cor = {
    0 : 'Indígena',
    2 : 'Branca',
    4 : 'Preta',
    6 : 'Amarela',
    8 : 'Parda',
    9 : 'Sem declaração'
    }
df["Cor"] = df["Cor"].map(Dict_Cor)
# Verifying changes made
df.loc[45000:45005]
df.groupby('Cor').count()
df['Anos de Estudo'].value_counts()
Dict_Anos = {
    1 : 0,
    2 : 1,
    3 : 2,
    4 : 3,
    5 : 4,
    6 : 5,
    7 : 6,
    8 : 7,
    9 : 8,
    10 : 9,
    11 : 10,
    12 : 11,
    13 : 12,
    14 : 13,
    15 : 14,
    16 : 15,
    17 : 0
    }
df["Anos de Estudo"] = df["Anos de Estudo"].map(Dict_Anos)
df.dtypes
# Verifying changes made
df.tail()
df['Altura'] = round(df['Altura'], 2)
df.loc[900:905]
df.columns
df["UF"].value_counts()
df["UF"].value_counts().plot(kind = 'bar', figsize=(12,5))
plt.title("Number of observations by UF")
df["Sexo"].value_counts()
df["Sexo"].value_counts().plot(kind = 'bar')
plt.title("Number of observations by Sexo")
# 0 means male;
# 1 means female.
df["Idade"].value_counts()
plt.title("Number of observations by Idade")
df["Idade"].plot(kind = 'hist')
# from 76840 observations, there are 423 that have Age less than 20
len(df["Idade"][df["Idade"]<20])
print("Maximum value for Idade", df["Idade"].max())
print("Minimum value for Idade", df["Idade"].min())
df["Cor"].value_counts()
plt.figure(figsize = (5,5))
plt.title("Number of observations by Cor")
df["Cor"].value_counts().plot(kind = 'bar')
df["Anos de Estudo"].value_counts()
plt.title("Number of observations by Anos de Estudo")
df["Anos de Estudo"].value_counts().plot(kind = 'bar')
# Anos de estudo by Cor
sns.boxplot(x = df['Cor'], y = df['Anos de Estudo'], data = df)
plt.title("Anos de Estudo x Cor")
# Anos de estudo by Sexo
sns.boxplot(x = df['Sexo'], y = df['Anos de Estudo'], data = df)
plt.title("Anos de Estudo x Sexo")
df.groupby('UF').mean()[['Anos de Estudo']].plot(kind='bar')
plt.title("Anos de Estudo (Average) x UF")
df["Renda"].value_counts()
# Some insights
print("Number of observations that have Renda < 20 k :", len(df["Renda"][df["Renda"] < 20000]))
print("Number of observations that have Renda > 20 k :", len(df["Renda"][df["Renda"] > 20000]))
print("Number of observations that have Renda > 40 k :", len(df["Renda"][df["Renda"] > 40000]))
print("\nAverage Salary (Renda) :", round(df['Renda'].mean(), 2))
print("Maximum value for Renda :", df["Renda"].max())
print("Minimum value for Renda :", df["Renda"].min())
# hist plot with zoom
plt.style.use('seaborn-talk')
fig, ax = plt.subplots(1, 4, figsize = (14, 5))
ax[0].hist(df["Renda"][df["Renda"] < 40000], bins = 100)
ax[0].set_title('Frequency x Renda (<40k)')
ax[1].hist(df["Renda"][df["Renda"] < 15000], bins = 100)
ax[1].set_title('Frequency x Renda (<15k)')
ax[2].hist(df["Renda"][df["Renda"] < 10000], bins = 100)
ax[2].set_title('Frequency x Renda (<10k)')
ax[3].hist(df["Renda"][df["Renda"] < 5000], bins = 100)
ax[3].set_title('Frequency x Renda (<5k)')
df["Renda"][df["Renda"] > 40000].plot(kind = 'hist', bins = 100)
plt.title('Frequency x Renda (>40k)')
# Renda (<5000) by cor
sns.boxplot(x = df['Cor'], y = df['Renda'][df['Renda'] < 5000], data = df[df['Renda'] < 5000])
plt.title('Renda (<5k) x Cor')
# Renda (>5000) by cor
sns.boxplot(x = df['Cor'], y = df['Renda'][df['Renda'] > 25000], data = df[df['Renda'] > 25000])
plt.title('Renda (>25k)  x Cor')
sns.boxplot(x = df['Sexo'], y = df['Renda'][df['Renda'] > 25000], data = df[df['Renda'] > 25000])
plt.title('Renda (>25k) x Sexo')
sns.boxplot(x = df['Sexo'], y = df['Renda'][df['Renda'] < 10000], data = df[df['Renda'] < 10000])
plt.title('Renda (<10k) x Sexo')
sns.boxplot(x = df['Sexo'], y = df['Renda'][df['Renda'] < 4000], data = df[df['Renda'] < 4000])
plt.title('Renda (<4k) x Sexo')
sns.scatterplot(df['Idade'], df['Renda'], data = df, hue = df['Cor'])
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80], labels = [0, 10, 20, 30, 40, 50, 60, 70, 80])
plt.title("Renda x Idade x Cor")
less_than_five_years = df[df["Anos de Estudo"] <= 5]
five_nine_years = df[(df["Anos de Estudo"] > 5) &  (df["Anos de Estudo"] < 10)]
nine_fourteen_years = df[(df["Anos de Estudo"] >= 10) & (df["Anos de Estudo"] < 15)]
more_than_fifteen_years = df[df["Anos de Estudo"] >= 15]
print("Average Salary (Renda) for 0-5 years of study :", round(less_than_five_years['Renda'].mean(), 2))
print("Average Salary (Renda) for 6-9 years of study :", round(five_nine_years['Renda'].mean(), 2))
print("Average Salary (Renda) for 10-14 years of study :", round(nine_fourteen_years['Renda'].mean(), 2))
print("Average Salary (Renda) for 15+ years of study :", round(more_than_fifteen_years['Renda'].mean(), 2))
# plot averages
year_avgs = np.array([
    round(less_than_five_years['Renda'].mean(), 2),
    round(five_nine_years['Renda'].mean(), 2),
    round(nine_fourteen_years['Renda'].mean(), 2),
    round(more_than_fifteen_years['Renda'].mean(), 2)
    ])

categories = np.array(['<5', '5-9', '10-14', '15+'])
plt.figure(figsize=(5,3))
sns.barplot(x=categories, y=year_avgs)
plt.title("Renda x Anos de Estudo")
df.groupby('UF').mean()[['Renda']].plot(kind='bar')
# Altura by Sexo
plt.title("Altura x Sexo")
sns.boxplot(x = df['Sexo'], y = df['Altura'])
print("Average height for men :", round(df[df['Sexo'] == 0].Altura.mean(), 3))
print("Average height for women :", round(df[df['Sexo'] == 1].Altura.mean(), 3))
sns.pairplot(df, hue = 'Cor')
sns.pairplot(df, hue = 'Sexo')
# heatmap for correlations
corr = df.corr()
sns.heatmap(corr, annot = True, vmin = 0, vmax = 1, cmap = 'Purples')
# Creating new df, with fewer outliers
df_model = df[df['Renda'] <= 15000]
df.shape, df_model.shape
df_model = pd.get_dummies(df_model, drop_first = True)
df_model.shape
df_model.head()
df_model.columns
features = df_model.drop('Renda', axis = 1)
features.shape
features.head()
target = df_model['Renda']
target.shape
target.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaling dataset
scaler.fit(features)
features_scaled = scaler.transform(features)
features_scaled
from sklearn.model_selection import cross_val_score
cv = 10
scoring = 'neg_mean_squared_error'
random_state = 0
all_models = []
all_scores = []
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
np.sqrt(-scores.mean())
from sklearn.linear_model import Lasso
model = Lasso(random_state = random_state)
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('Lasso')
all_scores.append(res)
all_models, all_scores
from sklearn.linear_model import Ridge
model = Ridge(random_state = random_state)
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('Ridge')
all_scores.append(res)
all_models, all_scores
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('KNN')
all_scores.append(res)
all_models, all_scores
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = random_state)
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('Decision Tree')
all_scores.append(res)
all_models, all_scores
from xgboost import XGBRegressor
# this takes a minute to run
model = XGBRegressor(random_state = random_state)
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('XGB')
all_scores.append(res)
all_models, all_scores
from sklearn.neural_network import MLPRegressor
# this takes a few minutes to run
model = MLPRegressor(random_state = random_state)
scores = cross_val_score(model, features_scaled, target, cv = cv,
                         scoring = scoring, n_jobs = -1)
res = round(np.sqrt(-scores.mean()), 2)
all_models.append('MLP')
all_scores.append(res)
all_models, all_scores
names = list(all_models)
values = list(all_scores)
# plot results
bar1 = plt.bar(np.arange(len(values)), values)
plt.xticks(range(len(names)), names)
plt.title('Renda prediction: Model x Error')
plt.ylim(0,2500)
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom', fontsize = 12, fontweight = 'bold')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target,
                                                      test_size = 0.2, random_state = random_state)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
from sklearn.metrics import mean_squared_error
# Define model
model = XGBRegressor(objective='reg:squarederror', random_state = random_state)

# Fit (train) model
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          eval_metric='rmse',
          verbose=False)
# Evaluate model
# Predict on new data (X_test). The model wasn't trained on this data and hasn't seen it yet
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE on test data: %.2f" % rmse)
all_models.append('XGB trained')
all_scores.append(round(rmse, 2))
all_models, all_scores
names = list(all_models)
values = list(all_scores)
# plot results
bar1 = plt.bar(np.arange(len(values)), values)
plt.xticks(range(len(names)), names)
plt.title('Renda prediction: Model x Error')
plt.ylim(0,2500)
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom', fontsize = 12, fontweight = 'bold')
features_importances = model.feature_importances_
argsort = np.argsort(features_importances)
features_importances_sorted = features_importances[argsort]

feature_names = features.columns
features_sorted = feature_names[argsort]

# plot feature importances
plt.figure(figsize = (5,10))
plt.barh(features_sorted, features_importances_sorted)
plt.title("Feature Importances")
print_every = 50
fig = plt.figure(figsize=(20,5))
plt.bar(list(range(len(y_test[::print_every]))), y_test.values[::print_every],
        alpha = 1, color = 'red', width = 1, label = 'true values')
plt.bar(list(range(len(y_pred[::print_every]))), y_pred[::print_every],
        alpha = 0.5, color = 'blue', width = 1, label = 'predicted values')
plt.legend()
# Making predictions of Renda for the first 5 observations of the test set (X_test)
model.predict(X_test)[0:5]
# Make any prediction you want!
# Define your features array: Set the values below for each column

my_pred = np.array([[

# Sexo
1,
# Idade
25,
# Anos de Estudo
8,
# Altura
1.65,
# UF_Alagoas
0,
# UF_Amapá
0,
# UF_Amazonas
0,
# UF_Bahia
0,
# UF_Ceará
0,
# UF_Distrito Federal
0,
# UF_Espírito Santo
0,
# UF_Goiás
0,
# UF_Maranhão
0,
# UF_Mato Grosso
0,
# UF_Mato Grosso do Sul
0,
# UF_Minas Gerais
0,
# UF_Paraná
0,
# UF_Paraíba
0,
# UF_Pará
0,
# UF_Pernambuco
0,
# UF_Piauí
0,
# UF_Rio Grande do Norte
0,
# UF_Rio Grande do Sul
0,
# UF_Rio de Janeiro
1,
# UF_Rondônia
0,
# UF_Roraima
0,
# UF_Santa Catarina
0,
# UF_Sergipe
0,
# UF_São Paulo
0,
# UF_Tocantins
0,
# Cor_Branca 
0,
# Cor_Indígena
0,
# Cor_Parda 
1,
# Cor_Preta
0
]])
res = model.predict(my_pred)
print("Renda predicted for information in my_pred array:", round(res[0], 2), "reais.")