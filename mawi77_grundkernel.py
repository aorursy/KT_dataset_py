# Laden des Packages
import pandas as pd

# Daten importieren
sales = pd.read_csv("/kaggle/input/dataakademie-zeitreihen/train.csv", parse_dates = ["Date"],  low_memory=False).drop(columns=["Store","StateHoliday"], axis = 1)
validation = pd.read_csv("/kaggle/input/dataakademie-zeitreihen/test.csv", parse_dates = ["Date"], low_memory=False).drop(columns=["Store","StateHoliday"], axis = 1)
# Sichten der Daten
print(sales.info())
print("\n")
print(sales.head())
print(validation.info())
print("\n")
print(validation.head())
# Erstellen der Variablen in den Trainingsdaten
# sales['Year'] = sales.Date.dt.year
# sales['Month'] = sales.Date.dt.month
# sales['WeekOfYear'] = sales.Date.dt.isocalendar().week
# sales["Week"]= sales.Date.dt.isocalendar().week%4

# Erstellen der Variablen in den Validierungsdaten
# validation['Year'] = validation.Date.dt.year
# validation['Month'] = validation.Date.dt.month
# validation['WeekOfYear'] = validation.Date.dt.isocalendar().week
# validation["Week"]= validation.Date.dt.isocalendar().week%4
# Anzeigen der Informationen
print(sales.info())
print("\n")
print(validation.info())
# Importieren von Packages
import seaborn as sns
import matplotlib.pyplot as plt

#Find Correlation between the data columns
plt.figure(figsize=(15,15))
sns.heatmap(sales.corr())
# Entfernen von Variablen in den Trainingsdaten
# sales.drop(columns=["DayOfWeek","SchoolHoliday"],inplace = True, axis = 1)

# Entfernen von Variablen in den Validierungsdaten
# validation.drop(columns=["DayOfWeek","SchoolHoliday"],inplace = True, axis = 1)
# Hinzufügen eines Index 
sales_index = sales.set_index("Date")

# Aufteilen in Erklärende und Ziel Variable
X = sales_index.drop(["Sales"], axis = 1)
y = sales_index["Sales"]

# Aufteilen in Trainings- und Testdaten
X_train = X.sort_index().loc["2013"].values
X_test = X.sort_index().loc["2014"].values
y_train = y.sort_index().loc["2013"].values
y_test = y.sort_index().loc["2014"].values
# Zuerst importieren wir die Funktion
from sklearn.linear_model import Lasso 

# Dann definieren wir das Modell
lss = Lasso()

# Trainieren des Modells auf den Trainingsdaten
model = lss.fit(X_train,y_train)
# Zuerst importieren wir die Funktion
#from sklearn.svm import SVR

# Dann definieren wir das Modell
#svr = SVR(C=0)

# Trainieren des Modells auf den Trainingsdaten
#model = lss.fit(X_train,y_train)
#from sklearn.svm import SVC#
#from sklearn.preprocessing import StandardScaler
#import sklearn.datasets
#iris_dataset = sklearn.datasets.load_iris()
#X, y = iris_dataset['data'], iris_dataset['target']
# Zuerst importieren wir die Funktion
#from sklearn.tree import DecisionTreeRegressor

 #Dann definieren wir das Modell
#clf = DecisionTreeRegressor()

# Trainieren des Modells auf den Trainingsdaten
#model = clf.fit(X_train,y_train)
# Importieren der Funktion aus dem Package
from sklearn.metrics import mean_squared_error

# Berechnen der Vorhersage für die Testdaten
y_pred_train = model.predict(X_train)

# Berechnen des MSE für Lasso
mean_squared_error(y_train,y_pred_train)
# Importieren der Funktion aus dem Package
from sklearn.metrics import mean_squared_error

# Berechnen der Vorhersage für die Testdaten
y_pred_test = model.predict(X_test)

# Berechnen des MSE für Lasso
mean_squared_error(y_test,y_pred_test)
# Zuerst importieren wir die Funktion
from sklearn.linear_model import Lasso 

# Dann definieren wir das Modell
lss = Lasso()

# Trainieren des Modells
model_final = lss.fit(X.values,y.values)
# Vorbereitung des Validationdatensatz
validation_model = validation.drop(["Date","Id"],axis = 1).values
# Modell exportieren
validation["Sales"] = model_final.predict(validation_model)
submission = validation[["Id","Sales"]]
submission.to_csv("submission.csv", index = False)