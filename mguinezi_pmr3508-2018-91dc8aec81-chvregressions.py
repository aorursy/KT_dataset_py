import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
from geopy.distance import vincenty
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
!cd ../input/ && ls
Train = pd.read_csv("../input/californianhouses/train.csv",
          sep=r'\s*,\s*',
          engine='python',
          na_values="")
Test = pd.read_csv("../input/californianhouses/test.csv",
         sep=r'\s*,\s*',
         engine='python',
         na_values="")
Train.head()
plt.boxplot(Train.median_house_value)
plt.title("Median_house_value")
plt.show()
plt.close()
# Id não é um critério para estimar a variável desejada
Train = Train.drop(columns=['Id'])
direct = Train.filter(Train.columns)
mcorr = direct.corr()
mcorr
plt.title('Matriz de correlação')
sns.heatmap(mcorr)
scaler = MinMaxScaler()
selected_columns = ['median_income', 'total_rooms','population','median_age']
SC = scaler.fit_transform(Train[selected_columns])
x_train, x_test, y_train, y_test = train_test_split(SC, Train['median_house_value'], test_size=0.20)
def rmsle(y_test, y_pred):
    return np.sqrt(np.mean((np.log(y_pred+1) - np.log(y_test+1))**2))

reg = LinearRegression()
scorer = make_scorer(rmsle, greater_is_better=False)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
param_grid = dict(n_neighbors=list(range(1,15)))
neigh = KNeighborsClassifier()
grid_obj = GridSearchCV(neigh, param_grid, scoring=scorer, cv=5)
grid_obj.fit(x_train, y_train)
grid_obj.best_params_
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
las = Lasso()
param_grid = dict(alpha=np.divide(list(range(1,100)),100))
grid_obj = GridSearchCV(las, param_grid, scoring=scorer, cv=5)
grid_obj.fit(x_train, y_train)
grid_obj.best_params_
las = Lasso(alpha=0.21)
las.fit(x_train, y_train)
y_pred = las.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
rfc = RandomForestClassifier(n_estimators=50, max_depth=35, random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))


dfTest = Test.drop(['longitude', 'latitude', 'households', 'total_bedrooms'], axis=1)
dfTest.head()
dfTest.shape
selected_model = rfc
x_val_test = scaler.transform(dfTest[selected_columns])
y_val_test = selected_model.predict(x_val_test)
dfSave = pd.DataFrame(data={"Id" : dfTest["Id"], "median_house_value" : y_val_test})
pd.DataFrame(dfSave[["Id", "median_house_value"]], columns = ["Id", "median_house_value"]).to_csv("Output.csv", index=False)

