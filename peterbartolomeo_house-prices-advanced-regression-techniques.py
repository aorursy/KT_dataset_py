import pandas as pd #Esta libreria se usó para leer los datos csv y guardar los resultados
from sklearn.linear_model import LinearRegression # Libreria para realizar regresiones lineales vista en clase


# Lectura de datos con pandas version web con los datos de kaggle 
entrenamiento = pd.read_csv('../input/train.csv')
prueba = pd.read_csv('../input/test.csv')
# Drop quita las columnas que no se van a ocupar y se crean variables dummies para que cuadre la información
entrenamiento = entrenamiento.drop(columns = ['Id','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'], axis = 1)
prueba = prueba.drop(columns = ['Id','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'], axis = 1)

# Drop quita las columnas que no se van a ocupar y se crean variables dummies para que cuadre la información
entrenamiento.fillna(0, inplace=True)
prueba.fillna(0, inplace=True)
# y es la variable que se va a predecir y X es la informacion que se ocupa para calcular Y
# aqui para X se elijen las variables desde la primera columna hasta una antes de la última
X = entrenamiento.iloc[:,:-1].values
# para Y solo se nececita la última columna
y = entrenamiento.iloc[:,-1].values

# se aplica la regresión lineal para hacer las predicciones de los datos de prueba
lin_reg = LinearRegression()
lin_reg.fit(X,y)
predicciones = lin_reg.predict(prueba.values)
# se le agregan las predicciones a la información
# y se eliminan las columnas sobrantes
archivo = pd.read_csv('../input/test.csv')
archivo['SalePrice'] = predicciones
archivo = archivo[['Id','SalePrice']]
# finalmente se exporta
archivo.to_csv('archivo.csv', sep=',', encoding='utf-8')
#se imprime el archivo
print(archivo)