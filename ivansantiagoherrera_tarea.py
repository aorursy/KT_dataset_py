import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
file_datospath = '../input/DataSet_Analizar.csv'
data = pd.read_csv(file_datospath, encoding = 'unicode_escape')
#data.info()
# ARREGLO COLUMNA IMC
IMC_new = data['Peso (Kg)']/(data['Talla (m)']**2)
data['IMC'] = IMC_new
#####################################################

# ARREOGLO FILAS CON DOBLE VALOR

data = data.drop(index = [14,17,25],axis = 0)

# ARREGLO COLUMNAS TIPO OBJETO
categorical_cols = [cname for cname in data.columns if data[cname].dtype in ['object']]
print(categorical_cols)
for j in range(len(categorical_cols)):
    temp=list(data[categorical_cols[j]])
    temp1=[]
    for i in range(len(temp)):
        temp1.append(float(temp[i]))
    data[categorical_cols[j]]=temp1

categorical_cols = [cname for cname in data.columns if data[cname].dtype in ['object']]
numerical_cols = [cname for cname in data.columns if data[cname].dtype in ['int64','float64']]

print("Columnas tipo de dato Numerico: \n",numerical_cols)
print("\n Columnas tipo de dato 'Object':\n",categorical_cols)
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='constant')

data_numerical = data[numerical_cols]
data_imputer = pd.DataFrame(my_imputer.fit_transform(data_numerical))
data_imputer.columns = data_numerical.columns

plt.figure(figsize=(16,10))
sns.swarmplot(x = data_imputer['Estado civil'], y = data_imputer['Problemas_ Cardiovasculares'], hue=data_imputer['Estado civil'])

plt.figure(figsize=(16,10))
sns.swarmplot(x = data_imputer['N.Hijos'], y = data_imputer['Problemas_ Cardiovasculares'])
sns.swarmplot(x = data_imputer['N.Hijos'], y = data_imputer['Problemas_ Cardiovasculares'], hue=data_imputer['Estado civil'])


plt.figure(figsize=(16,10))
sns.swarmplot(x = data_imputer['Ejercicio Fi1co'], y = data_imputer['Actividad FisicaHoy'])