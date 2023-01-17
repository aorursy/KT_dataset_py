import pandas as pd
from sklearn.preprocessing import Imputer
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
y = data['Price']
predictors = ['Rooms', 'BuildingArea', 'Landsize', 'YearBuilt']
x = data[predictors]
my_imp = Imputer()
imputed_x = my_imp.fit_transform(x)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

gbr = GradientBoostingRegressor()
gbr.fit(imputed_x, y)
my_plots = plot_partial_dependence(gbr, features = [1, 3], X=imputed_x, feature_names=predictors, grid_resolution=10)

titanic_data = pd.read_csv('../input/titanic/train.csv')
titanic_y = titanic_data.Survived
clf = GradientBoostingClassifier()
titanic_X_colns = ['PassengerId','Age', 'Fare',]
titanic_X = titanic_data[titanic_X_colns]
my_imputer = Imputer()
imputed_titanic_X = my_imputer.fit_transform(titanic_X)

clf.fit(imputed_titanic_X, titanic_y)
titanic_plots = plot_partial_dependence(clf, features=[1,2], X=imputed_titanic_X, 
                                        feature_names=titanic_X_colns, grid_resolution=8)
