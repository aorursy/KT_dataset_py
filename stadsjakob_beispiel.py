# Laden des Packages

import pandas as pd



# Daten importieren

sales = pd.read_csv("/kaggle/input/dataakademie-zeitreihen/train.csv", parse_dates = ["Date"],  low_memory=False)

validation = pd.read_csv("/kaggle/input/dataakademie-zeitreihen/test.csv", parse_dates = ["Date"], low_memory=False)
# Sichten der Daten

print(sales.info())

print("\n")

print(validation.info())
# Sichten der Daten

print(sales.head())

print("\n")

print(validation.head())
# Anzeigen der fehlnden Daten

display(sales.isnull().sum(),validation.isnull().sum())
# Erstellen der Variablen

sales['Year'] = sales.Date.dt.year

sales['Month'] = sales.Date.dt.month

sales['WeekOfYear'] = sales.Date.dt.isocalendar().week

sales["Week"]= sales.Date.dt.isocalendar().week%4



validation['Year'] = validation.Date.dt.year

validation['Month'] = validation.Date.dt.month

validation['WeekOfYear'] = validation.Date.dt.isocalendar().week

validation["Week"]= validation.Date.dt.isocalendar().week%4
# Anzeigen der Informationen

print(sales.info())

print("\n")

print(validation.info())
# Importieren von Packages

import seaborn as sns

import matplotlib.pyplot as plt



#Find Correlation between the data columns

plt.figure(figsize=(15,15))

sns.heatmap((sales).corr())
sales.drop(columns=["StateHoliday"], inplace = True)

validation.drop(columns=["StateHoliday"], inplace = True)
sales.set_index("Date", inplace = True)

validation.set_index("Date", inplace = True)
train = sales.sort_index().loc["2013"]

test = sales.sort_index().loc["2014"]
X_train = train.drop(["Sales"], axis = 1).values

y_train = train["Sales"].values

X_test = test.drop(["Sales"], axis = 1).values

y_test = test["Sales"].values
validation_model = validation.drop("Id",axis = 1).values
# Zuerst importieren wir die Funktion

from sklearn.linear_model import Lasso 



# Dann definieren wir das Modell

lss = Lasso()



# Trainieren des Modells

model_lasso = lss.fit(X_train,y_train)
# Importieren der Funktion aus dem Package

from sklearn.metrics import mean_squared_error



# Berechnen der Vorhersage für die Trainingsdaten

y_pred_lasso_test = model_lasso.predict(X_test)



# Berechnen des MSE für Lasso

mean_squared_error(y_test,y_pred_lasso_test)

### 3.7 Modell exportieren



validation["target"] = model_lasso.predict(validation_model)

submission = validation[["Id","target"]]

submission.to_csv("submission.csv")