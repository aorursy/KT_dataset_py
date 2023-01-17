import pandas as pd



data_cars = pd.read_csv("../input/cars.csv")

data_cars.index = data_cars["name"]

# DataFrame.hist() invoca la función hist() de la libreria matplotlib.pyplot

data_cars.hist("mpg", bins=20) # Seleccionamos la columna del DataFrame a graficar y el número de columnas del histograma

print("Media (mpg): ", data_cars.mean()["mpg"])

print("Mediana (mpg): ",data_cars.median()["mpg"])

print("DesviaciónStd (mpg): ",data_cars.std()["mpg"])