import pandas as pd
import numpy as np
cars = pd.read_csv('../input/imports-85.data.txt') # imported the dataset
cars.head()
columnnames=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
           'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

cars = pd.read_csv('../input/imports-85.data.txt',names=columnnames)

cars=cars.replace("?",np.nan)
cars.isnull().sum()
# here we are converting the categorical one to numerical
convert = {"num-of-doors": {"four": 4, "two": 2}}
cars.replace(convert, inplace=True) 
import pandas as pd
from sklearn.preprocessing import Imputer

imp=Imputer(missing_values="NaN", strategy="mean" )
imp.fit(cars[["bore"]])
cars["bore"]=imp.transform(cars[["bore"]]).ravel()

imp=Imputer(missing_values="NaN", strategy="mean" )
imp.fit(cars[["stroke"]])
cars["stroke"]=imp.transform(cars[["stroke"]]).ravel()

imp=Imputer(missing_values="NaN", strategy="most_frequent" )
imp.fit(cars[["num-of-doors"]])
cars["num-of-doors"]=imp.transform(cars[["num-of-doors"]]).ravel()

imp=Imputer(missing_values="NaN", strategy="most_frequent" )
imp.fit(cars[["horsepower"]])
cars["horsepower"]=imp.transform(cars[["horsepower"]]).ravel()

imp=Imputer(missing_values="NaN", strategy="most_frequent" )
imp.fit(cars[["peak-rpm"]])
cars["peak-rpm"]=imp.transform(cars[["peak-rpm"]]).ravel()


cars.isnull().sum()
from fancyimpute import KNN    

X_filled_knn = KNN(k=3).complete(cars[['price']])

# try in jupyter the code will work
X_filled_knn = pd.DataFrame(X_filled_knn, columns = ['price'])
#try in jupyter the code will work
cars['price'] = np.round(X_filled_knn['price'], 0)
cars.isnull().sum()
#try in jupyter it will work