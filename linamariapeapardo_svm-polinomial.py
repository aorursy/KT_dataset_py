import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV # Para hacer validación cruzada
from sklearn.preprocessing import LabelEncoder # Change values of labels to 0 and 1
import seaborn as sns   
table = pd.read_csv('../input/genre-dataset/genre_dataset.txt')
table = table[(table.genre.str.contains("jazz and blues"))|
        (table.genre.str.contains("soul and reggae"))]

for i in range (4,len(table.columns)):
    col = table.iloc[:,[i]].values
    table.iloc[:,[i]] = scale(col)
table.head()
# Jazz es 0 en tanto Reggae es 1
le = LabelEncoder()
table['genre'] = le.fit_transform(table[['genre']])
table.head()
# Divide in training, test and validation sets
X_train ,X_test = train_test_split(table,test_size=0.2)

x_train = X_train.iloc[:,4:].values
y_train = X_train.iloc[:,0]
x_test = X_test.iloc[:,4:].values
y_test = X_test.iloc[:,0]
cls = svm.SVC(kernel='poly')
# Ajuste del grado del polinomio
smallgrid = {'degree': [1,2,3,4,5,6,7,8,9,10], 'C': [2]}
grid1 = GridSearchCV(cls,smallgrid)
grid1.fit(x_train,y_train)

# Graficar los resultados
grid_result1 = pd.DataFrame(grid1.cv_results_)
plt = grid_result1.plot(x ='param_degree', y='mean_test_score')
plt = plt.set_ylabel("Precisión")
# Ajuste de los parámetros del modelo en conjunto
param_grid = {'C': [0.1,1, 10, 100],'degree': [2,3,4,5]}
grid = GridSearchCV(cls,param_grid)
grid.fit(x_train,y_train)

pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_degree', columns='param_C')
sns.heatmap(pvt, annot=True)
# Ajuste del grado del polinomio
smallgrid2 = {'C': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],'degree': [3]}
grid2 = GridSearchCV(cls,smallgrid2)
grid2.fit(x_train,y_train)

# Graficar los resultados
grid_result2 = pd.DataFrame(grid2.cv_results_)
plt = grid_result2.plot(x ='param_C', y='mean_test_score')
plt = plt.set_ylabel("Precisión")
# Ajuste del grado del polinomio
smallgrid3 = {'gamma': [0.001, 0.01, 0.1], 'C': [2],'degree': [3]}
grid3 = GridSearchCV(cls,smallgrid3)
grid3.fit(x_train,y_train)

# Graficar los resultados
grid_result3 = pd.DataFrame(grid3.cv_results_)
plt3 = grid_result3.plot(x ='param_gamma', y='mean_test_score')
cls_final = svm.SVC(kernel='poly', C=2, degree=3, gamma=0.01)
# Entrenamiento
cls_final.fit(x_train,y_train)
# Test con datos de prueba
pred =  cls_final.predict(x_test)
print('accuracy:', metrics.accuracy_score(y_test,y_pred=pred))
print (metrics.classification_report(y_test,y_pred=pred))