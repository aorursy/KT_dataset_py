import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_excel("../input/co2data/data.xlsx")
df = df.rename(columns = {"country" : "Country",
                          "GDP per capita (current US$)" : "GDP",
                          "Trade (% of GDP)" : "Trade",
                          "Population, total" : "Total_Population",
                          "Rural population" : "Rural_Population"})
df.head()
df.isnull().sum()
owid = pd.read_csv("../input/co2data/owid-co2-data.csv")

df_filtered = owid[(owid["year"] == 2018)]

co2 = df_filtered[["country","co2"]]
co2 = co2.reset_index()

del co2["index"]

co2.head()
co2.isnull().sum()
main_df = pd.concat([df.set_index('Country'), co2.set_index('country')], axis = 1)
main_df = main_df.dropna(axis = 0, how = "any")
main_df = main_df.reset_index()

del main_df["index"]

main_df.head()
main_df.info()
sns.boxplot(x = main_df["GDP"]);
sns.boxplot(x = main_df["Rural_Population"]);
co2 = main_df["co2"]

del main_df["co2"]

co2.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(main_df, co2, test_size = 0.33)

from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor()
gbm.fit(x_train, y_train)

y_pred = gbm.predict(x_test)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, y_pred))
gbmtun = GradientBoostingRegressor(learning_rate = 1,
                                   loss = "lad",
                                   max_depth = 3,
                                   n_estimators = 75).fit(x_train, y_train)

y_pred2 = gbmtun.predict(x_test)

np.sqrt(mean_squared_error(y_test, y_pred2))
plt.plot(y_test, y_pred);
plt.plot(y_test, y_pred2);
from xgboost import XGBRegressor

xgb = XGBRegressor(colsample_bytree = 0.5,
                   learning_rate = 0.05,
                   max_depth = 1,
                   n_estimators = 50).fit(x_train, y_train)

y_pred3 = xgb.predict(x_test)

np.sqrt(mean_squared_error(y_test,y_pred3))
from sklearn.neighbors import KNeighborsRegressor

HKO  = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(x_train, y_train)
    y_pred4 = knn_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred4))
    HKO.append(rmse)
    print("k =", "k, için hatalar karesi ortalaması değeri = ", rmse)
