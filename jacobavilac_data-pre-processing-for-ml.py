import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# We load the data set
dataset = pd.read_csv ('../input/Preproc_Datos_Compras.csv')
print(dataset)
#Separating dependant variable from independant variable
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]
#Imputting missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x.values[:, 1:3])
x.iloc[:, 1:3] = imputer.transform(x.values[:, 1:3])
print("%.0f", x)
# Codificacion de variables categoricas
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto', n_values=x.values[:,0])
dummies = onehotencoder.fit_transform(x.values[:,0].reshape(-1,1)).toarray()

dummies = pd.DataFrame(dummies)
x_new = pd.concat([dummies,x.iloc[:,1:3]], axis=1 )
print(x_new)
from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print("%i", y)
# Transformación de escalas
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_new = sc_x.fit_transform(x_new)
y = sc_y.fit_transform(y.reshape(-1, 1))
print("%i", x_new)
print("%i", y)