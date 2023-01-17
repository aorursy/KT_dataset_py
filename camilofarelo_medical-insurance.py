#Importar librerias importantes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df=pd.read_csv('../input/insurance/insurance.csv')#importar dataset
df.head()
df['region'].unique()
sns.violinplot(data=df,x='sex',y='charges')#Conociendo la relacion entre el sexo y la variable objetivo
sns.violinplot(data=df,x='region',y='charges')#Conociendo la relacion entre la region y la variable objetivo
sns.violinplot(data=df,x='smoker',y='charges')#Conociendo la relacion entre fumadores y la variable objetivo
corr=df.corr()
sns.heatmap(data=corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1)#Mapa de correlacion para conocer esta en las variables numericas
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
sns.pairplot(df)
smoker={'no':0,'yes':1}#transformacion de la variable categorica
df['smoker']=df['smoker'].map(smoker)
df['charges']=np.log(df['charges'])
df.head()
df.isnull().any()
plt.figure(figsize=(10,7))
sns.pairplot(df)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X=df[['age','bmi','children','smoker']].iloc[0:300]#Seleccionar las caracteristcas importantes para el train y test con unas 300 muestras
y=df[['charges']].iloc[0:300]#Variable objetivo
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2)#Separacion de los datos de train y test
modelo=Pipeline([('ss',StandardScaler()),
                 ('lr',LinearRegression())])#concatena los pasos de normalizacion y entrenamiento
modelo.fit(X_train,y_train)#Entrenar el modelo
print(f'El puntaje del modelo obtenido es {modelo.score(X_test,y_test):.2f}')
print(f"El intercepto es {modelo['lr'].intercept_}")
print(f"Los coeficientes del modelo son {modelo['lr'].coef_}")
from sklearn import metrics
y_pred=modelo.predict(X_test)
metrics.mean_squared_error(y_test,y_pred)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
def polmodel(grado=2):
  model=Pipeline([('pol',PolynomialFeatures(degree=grado)),('ss',StandardScaler()),('lr',LinearRegression())]).fit(X_train,y_train)
  return model 
polmodel().score(X_test,y_test)
polmodel(4).score(X_test,y_test)
polmodel(3).score(X_test,y_test)
from sklearn.model_selection import validation_curve
grados= np.arange(2,11)
train_scores, test_scores =validation_curve(polmodel(),X,y,param_name='pol__degree',param_range=grados,cv=5)
from sklearn.linear_model import SGDRegressor 
model2=Pipeline([('ss',StandardScaler()),('gd',SGDRegressor(max_iter=100,learning_rate='constant',eta0=0.01,alpha=0.5))]).fit(X_train,y_train)
model2.score(X_test,y_test)