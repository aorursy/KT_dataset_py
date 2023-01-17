# Importando Librerías
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Input data files are available in the "../input/" directory.
print(os.listdir("../input"))
seguros_ds = pd.read_csv("../input/insurance.csv")
seguros_ds.info()
#Vericando la distribucion de las variable numericas
with sns.plotting_context("paper",font_scale=2):
    d = sns.pairplot(seguros_ds[['age','bmi','children','charges']], 
                  hue='children',palette='Set3',size=4)
d.set(xticklabels=[]);
#Verificando las correlaciones entre las variables numericas
seguros_ds.corr()
#Matrix de correlation inicial
f, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(seguros_ds.corr(), vmax=.5, square=True);
#Previsualizacion los datos previo a cambios
seguros_ds.head(15)
#Variable a predecir
seguros_ds['charges'].describe()
#Escalando variables numéricas a rangos entre 0 y 1
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
seguros_ds[['age', 'children', 'bmi']] = mms.fit_transform(seguros_ds[['age', 'children', 'bmi']])
seguros_ds[['age', 'children', 'bmi']].describe()
#Actualizando valores de la variable Sex a numericos
seguros_ds['sex'] = seguros_ds['sex'].map( {'female': 1, 'male': 0,} ).astype(int)

#Conversión de los valores Si y No a 1 y 0
seguros_ds['smoker'] = seguros_ds['smoker'].map( {'yes': 1, 'no': 0,} ).astype(int)

seguros_ds[['sex','smoker']].describe()
#Validar las correlaciones
correlaciones = seguros_ds.corr()[1:]
correlaciones
#Variables predictivas
corReal=correlaciones[correlaciones >0.1][:]
corReal.dropna(axis=1,how='all',inplace=True)
corReal
#Remover la variables con baja correlacion
corReal.drop(['sex','children'],axis=1,inplace=True)
corReal.columns
#Variables correlacionadas
varPredictores=corReal.columns
varPredictores=varPredictores[:len(varPredictores)-1]
varPredictores
#Caculating VIF to check multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
y, X = dmatrices('charges ~ age+bmi+smoker', seguros_ds, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
#Definición de las variable predictores
X = seguros_ds[varPredictores]
Y = seguros_ds['charges']

from sklearn.model_selection import train_test_split
X_train, X_other, y_train, y_other = train_test_split(X,Y,test_size=0.35)
X_test, X_valid, y_test, y_valid = train_test_split(X_other,y_other,test_size=0.3)

print("Tamaño de los datasets:\nTrain: %d\nTest: %d\nValidation: %d" % (len(X_train), len(X_test), len(X_valid)) )
#Predecir charges
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

model1 = linear_model.LinearRegression()

prediccion = cross_val_predict(model1, X, Y, cv=2)
prediccion_score = cross_val_score(model1, X, Y, cv=2)
prediccion_score.mean()
#K-fold Cross validation function validate(model, X_train, y_train, k=10)
def validate(model1, X_train, y_train, k=8):
    result = 'K-fold cross validation:\n'
    scores = cross_val_score(estimator=model1,
                             X=X_train,
                             y=y_train,
                             cv=k,
                             n_jobs=1)
    for i, score in enumerate(scores):
        result += "Iteration %d:\t%.3f\n" % (i, score)
    result += 'CV accuracy:\t%.3f +/- %.3f' % (np.mean(scores), np.std(scores))
    return result
print(validate(model1, X_train, y_train))
#Curva de aprendizaje
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def learningCurve(model1, X_train, y_train, k=10):
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=model1,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 10),
                                   cv=k,
                                   n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(train_sizes, train_mean,
             color='orange', marker='o',
             markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='orange')

    plt.plot(train_sizes, test_mean,
             color='purple', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='purple')

    plt.grid()
    plt.xlabel('Numero de muestras de entrenamiento', fontsize=fsize)
    plt.ylabel('Veracidad', fontsize=fsize)
    plt.legend(loc='lower right')
    plt.ylim([0.4, 1.03])
    plt.tight_layout()
    plt.show()
#Visualización de la gráfica de la curva de entrenamiento y test
learningCurve(model1, X_train, y_train)
#Actualizando las variables de prediccion
corReal.drop(['bmi'],axis=1,inplace=True)
varPredictores2 = corReal.columns
varPredictores2 = varPredictores2[:len(varPredictores2)-1]
varPredictores2
X = seguros_ds[varPredictores2]
Y = seguros_ds['charges']

from sklearn.model_selection import train_test_split
X_train, X_other, y_train, y_other = train_test_split(X,Y,test_size=0.35)
X_test, X_valid, y_test, y_valid = train_test_split(X_other,y_other,test_size=0.3)

print("Tamaño de los datasets:\nTrain: %d\nTest: %d\nValidation: %d" % (len(X_train), len(X_test), len(X_valid)) )
model2 = linear_model.LinearRegression()

prediccion = cross_val_predict(model2, X, Y, cv=2)
prediccion_score = cross_val_score(model2, X, Y, cv=2)
prediccion_score.mean()
#Visualización de la gráfica de la curva de entrenamiento y test
learningCurve(model2, X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
model2.fit(X_train, y_train)

#Nueva prediccion
y_pred = model2.predict(X_test)

# Resultados
print('Coeficientes: \n', model2.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))