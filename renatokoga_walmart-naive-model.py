# Carrega os pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from sklearn.tree import DecisionTreeRegressor
from datetime import date, timedelta
# Importando os arquivos
TrainData = pd.read_csv("train.csv", parse_dates = ["Date"])
StoreData = pd.read_csv("stores.csv")
FeaturesData = pd.read_csv("features.csv", parse_dates = ["Date"])
TestData = pd.read_csv("test.csv", parse_dates = ["Date"])
# Extração das informações de data - Test Data
TestData["Month"] = pd.DatetimeIndex(TestData['Date']).month
TestData["Year"] = pd.DatetimeIndex(TestData['Date']).year
TestData["Day"] = pd.DatetimeIndex(TestData['Date']).day
TestData["WeekNumber"] = TestData["Date"].dt.week
TestData.tail()
# Extração das informações de data - Train Data
TrainData["Month"] = pd.DatetimeIndex(TrainData['Date']).month
TrainData["Year"] = pd.DatetimeIndex(TrainData['Date']).year
TrainData["Day"] = pd.DatetimeIndex(TrainData['Date']).day
TrainData["WeekNumber"] = TrainData["Date"].dt.week
TrainData.head()
# Extração das informações de data - Features Data
FeaturesData["Month"] = pd.DatetimeIndex(FeaturesData['Date']).month
FeaturesData["Year"] = pd.DatetimeIndex(FeaturesData['Date']).year
FeaturesData["Day"] = pd.DatetimeIndex(FeaturesData['Date']).day
FeaturesData["WeekNumber"] = FeaturesData["Date"].dt.week
FeaturesData.head()
# Busca dados do ano anterior (lag1)
TrainData1 = TrainData[((TrainData.Year == 2011) & (TrainData.WeekNumber >= 44) | (TrainData.Year == 2012) & (TrainData.WeekNumber <= 30))]
print(TrainData1.count())
TrainData1 = TrainData1.rename(columns={'Weekly_Sales': 'Weekly_Sales_lag1', 'IsHoliday': 'IsHoliday_lag1', 'Date': 'Date_lag1'})
TrainData1.head()
# Busca dados de 2 anos atrás (lag2)
TrainData2 = TrainData[((TrainData.Year == 2010) & (TrainData.WeekNumber >= 44) | (TrainData.Year == 2011) & (TrainData.WeekNumber <= 30))]
print(TrainData2.count())
TrainData2 = TrainData2.rename(columns={'Weekly_Sales': 'Weekly_Sales_lag2', 'IsHoliday': 'IsHoliday_lag2', 'Date': 'Date_lag2'})
TrainData2.head()
# Busca dados de 3 anos atrás (lag3)
TrainData3 = TrainData[((TrainData.Year == 2009) & (TrainData.WeekNumber >= 44) | (TrainData.Year == 2010) & (TrainData.WeekNumber <= 30))]
print(TrainData3.count())
TrainData3 = TrainData3.rename(columns={'Weekly_Sales': 'Weekly_Sales_lag3', 'IsHoliday': 'IsHoliday_lag3', 'Date': 'Date_lag3'})
TrainData3.head()
# merge para descobrir o valor de vendas no ano anterior
PredData = pd.merge(TestData, TrainData1[["WeekNumber", "Weekly_Sales_lag1", "Store", "Dept", "Date_lag1", ]], on = ["WeekNumber", "Store", "Dept"], how="left")
PredData = pd.merge(PredData, TrainData2[["WeekNumber", "Weekly_Sales_lag2", "Store", "Dept", "Date_lag2", ]], on = ["WeekNumber", "Store", "Dept"], how="left")
PredData = pd.merge(PredData, TrainData3[["WeekNumber", "Weekly_Sales_lag3", "Store", "Dept", "Date_lag3", ]], on = ["WeekNumber", "Store", "Dept"], how="left")
PredData["Lag1"] = PredData["Weekly_Sales_lag1"] - PredData["Weekly_Sales_lag2"]
PredData["Lag2"] = PredData["Weekly_Sales_lag2"] - PredData["Weekly_Sales_lag3"]
PredData["Lag3"] = PredData["Weekly_Sales_lag1"] - PredData["Weekly_Sales_lag3"]
PredData["Dif1"] = PredData["Lag1"] / PredData["Weekly_Sales_lag2"]
PredData.head()
PredData[PredData.Lag1 >= 10000].sort_values(["Dif1"], ascending=(False)).head(50)
PredData.count()
PredData[PredData.Dif1 > 3].count()
# Previsão 1: Repete o valor do ano anterior. Caso não tenha, busca o valor de 2 ou de 3 anos atrás.
PredData["Pred1"] = PredData["Weekly_Sales_lag1"].where(PredData["Weekly_Sales_lag1"].notnull(), PredData["Weekly_Sales_lag2"])
PredData["Pred1"] = PredData["Pred1"].where(PredData["Pred1"].notnull(), PredData["Weekly_Sales_lag3"])
PredData
# Previsão 2: Para os casos onde a dif entre lag1 e lag > 200%, vamos ponderar os resultados
PredData["Pred2"] = np.where(PredData.Dif1 <= 2, PredData.Pred1, PredData.Weekly_Sales_lag1 * .75 + PredData.Weekly_Sales_lag2 * .25)
PredData["Pred2"] = PredData["Pred2"].where(PredData["Pred2"].notnull(), PredData["Pred1"])
PredData.head()
PredData[PredData.Dif1 >= 2]
PredData.isnull().sum()
# Verificar se algum registro ficou sem valor predito
PredData[PredData.Pred2.isnull()]
# Casos sem histórico - colocar valor predito como zero
PredData = PredData.fillna(0)
# Monta o arquivo para subir no Kaggle
PredData["Date2"] = PredData['Date'].astype(str)
PredData["Date3"] = PredData["Date2"].str[0:10]
PredData["Id"] =  PredData['Store'].astype(str) + '_' +  PredData['Dept'].astype(str) + '_' +  PredData['Date3'].astype(str)

# Primeira inferência
Submit1 = PredData[["Id", "Pred1"]]
Submit1 = Submit1.rename(columns={'Pred1': 'Weekly_Sales'})
Submit1.head()
Submit1.to_csv('submit1.csv', index=False)

# Segunda inferência
Submit2 = PredData[["Id", "Pred1"]]
Submit2 = Submit2.rename(columns={'Pred1': 'Weekly_Sales'})
Submit2.head()
Submit2.to_csv('submit2.csv', index=False)
