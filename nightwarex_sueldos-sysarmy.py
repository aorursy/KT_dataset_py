import pandas as pd

import matplotlib.pyplot as plt
## Import dataset

data=pd.read_csv('../input/encuesta.csv',skiprows=8)    ## First rows are commentaries not data

data.drop_duplicates(keep = 'first', inplace = True)    ##remove duplicates
data.head()
##columns names

data.columns
data.describe()
mean_salario = data[['Salario mensual NETO (en tu moneda local)','Nivel de estudios alcanzado']].groupby('Nivel de estudios alcanzado').mean()

mean_salario
mean_salario.plot(kind='bar')