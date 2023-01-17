import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ebola = pd.read_csv("../input/ebolaaa/ebola.csv")
ebola.head()
ebola_filter = ebola[(ebola.Indicator == 'Cumulative number of confirmed Ebola cases')]
ebola_filter
ebola_filter.groupby(['Date']).sum()
plt.figure(figsize=(35,8))
plt.title("Количество заболевших")
sns.barplot(x=ebola_filter['value'], y=ebola_filter['Country'])
plt.show()
ebola_filter.to_csv("output.csv", index=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
model = pd.read_csv("../input/nedvijimosttt/Nedvijimost.csv")
model.describe()
model.isnull().sum()
model = model.fillna(model.mean())
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
s = (model.dtypes == 'object')
object_cols = list(s[s].index)
print("Категориальные атрибуты:")
print(object_cols)

label_model = model.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_model[col] = label_encoder.fit_transform(model[col])
label_model
features_1 = ['Район','Наличие агенства','Тип планировки','Количество комнат','Общая площадь (м2)']
X = label_model[features_1]
y = label_model['Стоимость (т.руб.)']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
val_X.reset_index(inplace = True)

output_1 = pd.DataFrame({'Район':val_X['Район'],'Наличие агенства':val_X['Наличие агенства'],'Тип планировки':val_X['Тип планировки'],'Количество комнат':val_X['Количество комнат'],
                         'Общая площадь (м2)':val_X['Общая площадь (м2)'],'Стоимость (т.руб.)':rf_val_predictions})
print(rf_val_mae)
print(output_1)
features_2 = ['Район','Тип планировки','Количество комнат','Общая площадь (м2)']
X = label_model[features_2]
y = label_model['Стоимость (т.руб.)']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
val_X.reset_index(inplace = True)

output_2 = pd.DataFrame({'Район':val_X['Район'],'Тип планировки':val_X['Тип планировки'],'Количество комнат':val_X['Количество комнат'],'Общая площадь (м2)':val_X['Общая площадь (м2)'],
                         'Стоимость (т.руб.)':rf_val_predictions})
print(rf_val_mae)
print(output_2)
features_3 = ['Состояние','Наличие агенства','Тип планировки','Количество комнат','Район','Общая площадь (м2)']
X = label_model[features_3]
y = label_model['Стоимость (т.руб.)']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
val_X.reset_index(inplace = True)

output_3 = pd.DataFrame({'Состояние':val_X['Состояние'],'Наличие агенства':val_X['Наличие агенства'],'Тип планировки':val_X['Тип планировки'],'Количество комнат':val_X['Количество комнат'],
                         'Район':val_X['Район'],'Общая площадь (м2)':val_X['Общая площадь (м2)'],'Стоимость (т.руб.)':rf_val_predictions})
print(rf_val_mae)
print(output_3)
output_3.to_csv('best_model.csv', index = False)