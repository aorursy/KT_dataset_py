# Bu uygulamada verilen ev örneklerinden ev fiyatı tahmin etmeye çalışacağız.
import pandas as pnds
dataFile = '../input/fiyat.csv'
data = pnds.read_csv(dataFile)
data.describe()
data.columns
featureSelecting = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[featureSelecting]
X.describe()
y = data.Price
y.describe()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error
fiyatTahmin = regressor.predict(X_test)
mean_absolute_error(y_test, fiyatTahmin)