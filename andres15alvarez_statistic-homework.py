import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
rows = len(train.dtypes.values)

i = 0

count = 0 #Almacenara la cantidad de datos tipo objeto, es decir las variables cualitativas

for i in range(rows):

    if(train.dtypes.values[i] == 'O'):

        count += 1

    i += 1

print("La cantidad de variables cualitativas es: " + str(count))
notnulos = train.notnull().sum()

cantnotnull = notnulos.sum()

cantotal = cantnotnull - train.Id.count()

print(cantotal)
counts = train['RoofStyle'].value_counts()

with plt.style.context('dark_background'):

    counts.plot(kind='bar', color='steelblue')

    plt.ylabel("Frecuency of Roof Styles")

    plt.title("Bar Chart of Roof Styles")
counts = train['RoofStyle'].value_counts()

porct =counts/1460*100

label = []

for i in range(len(counts)):

    label.append(counts.index[i] + " "+ '{0:.2f}%'.format(porct[i]))

sizes = [1141, 286, 13, 11, 7, 2]  

colors = ['steelblue', 'skyblue', 'navy', 'blue', 'red', 'green']

with plt.style.context('dark_background'):

	fig, ax = plt.subplots()

	ax.pie(sizes, colors=colors, shadow=False, startangle=0)

	ax.axis('equal') 

	ax.legend(label, shadow=True) 

	plt.title("Pie Chart of the Roof Styles")

	plt.show()
Yearsold = train.YrSold.value_counts().sort_index()

index = np.array(Yearsold.index, dtype=str)

with plt.style.context('dark_background'):

    plt.subplots()

    plt.plot(index, Yearsold, color='steelblue')

    plt.title("Year Sold")
datahist = train['1stFlrSF']

weights = np.ones_like(datahist)/float(len(datahist))

with plt.style.context('dark_background'):

    plt.subplots()

    datahist.hist(color='steelblue', weights=weights)

    plt.title('Relative Frecuency Histrogram of Superficie of the first floor')
#Vamos a colocar en una lista solo las variables numericas

numericVariables=[]

for i in range(rows):

    if train.dtypes.values[i] != 'O':

        numericVariables.append(train.dtypes.index[i])

numericVariables.pop(0)



#Ahora procedemos a calcular sus medidas

medias = np.array(train[numericVariables].mean().values)

medianas = np.array(train[numericVariables].median().values)

modas = np.array(train[numericVariables].mode().values[0])

varianzas = np.array(train[numericVariables].var().values)

dvestandar = np.array(train[numericVariables].std().values)

curtosis = np.array(train[numericVariables].kurtosis().values)

sesgo = np.array(train[numericVariables].skew().values)

tabla = pd.DataFrame({'Media':medias,'Mediana':medianas,'Moda':modas,

                      'varianza':varianzas,'Desviacion estandar':dvestandar,

                      'Curtosis':curtosis,'Sesgo':sesgo}, index=numericVariables)

tabla
from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_absolute_error

from scipy.stats import zscore

train.corr()
features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 

            'FullBath', 'TotRmsAbvGrd', 'GarageCars','GarageArea','Fireplaces','MasVnrArea']

#Verificamos que no tengan valores nulos

train[features].isnull().any()
train[train.MasVnrArea.isnull()]
train.MasVnrArea = train.MasVnrArea.fillna(0.0)

X = train[features]

y = train.SalePrice.values

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=44)

#Escogemos un 80% para el entrenamiento y 20% para la prueba

model = LinearRegression()

model.fit(XTrain, yTrain)
predict = model.predict(XTest)

mae = mean_absolute_error(y_true=yTest,y_pred=predict)

mse = mean_squared_error(y_true=yTest,y_pred=predict)

rmse = np.sqrt(mse)

r2 = model.score(XTrain,yTrain)

print("Error absoluto medio: ", str(mae))

print("Error Cuadratico Medio: ", str(mse))

print("Raiz del Error Cuadratico Medio: ", str(rmse))

print("Coeficiente de determinacion: ", str(r2))
#Escogemos las columnas de feature y el target para quitarles los valores atipicos

columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 

           'FullBath', 'TotRmsAbvGrd', 'GarageCars','GarageArea','Fireplaces','MasVnrArea', 'SalePrice']

df = train[columns] #df sera nuestro DataFrame auxiliar

z_scores = zscore(df)

abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3).all(axis=1)

train = df[filtered_entries]
#Ya tenemos el nuevo dataframe para el entrenamiento, asi que repetimos el proceso

train.OverallQual = np.power(train.OverallQual,2) #Realizamos una transformacion

X = train[features]

y = train.SalePrice.values

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=44)

model = LinearRegression()

model.fit(XTrain, yTrain)
predict = model.predict(XTest)

mae = mean_absolute_error(y_true=yTest,y_pred=predict)

mse = mean_squared_error(y_true=yTest,y_pred=predict)

rmse = np.sqrt(mse)

r2 = model.score(XTrain,yTrain)

print("Error absoluto medio: ", str(mae))

print("Error Cuadratico Medio: ", str(mse))

print("Raiz del Error Cuadratico Medio: ", str(rmse))

print("Coeficiente de determinacion: ", str(r2))