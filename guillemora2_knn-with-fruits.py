# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fruits=pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
fruits.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler = MinMaxScaler()
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import minmax_scale
#from sklearn.preprocessing import MaxAbsScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer

scaler.fit(fruits.drop(['fruit_name','fruit_label','fruit_subtype'],axis=1))
scaled_features = scaler.transform(fruits.drop(['fruit_name','fruit_label','fruit_subtype'],axis=1))
df_feat = pd.DataFrame(scaled_features,columns=['mass','width','height','color_score'])
df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,fruits['fruit_label'],
                                                    test_size=0.30,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print((pred==y_test).mean())
import matplotlib.pyplot as plt
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))



plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

print('the best k='+str(error_rate.index(min(error_rate))+1))
ax = plt.axes(projection='3d')

#mass	width	height	color_score scaled_features

# Data for three-dimensional scattered points
zdata = scaled_features[:,1]
xdata = scaled_features[:,2]
ydata = scaled_features[:,3]
#scaled_features[:,3]

ax.scatter3D(xdata, ydata, zdata, c=fruits['fruit_label'], cmap='viridis');
#lets play with  plotly
import plotly.express as px

fig = px.scatter_3d(fruits, x='mass', y='width', z='height',
              color='color_score',symbol='fruit_name',opacity=0.7)
fig.show()

# Importamos desde scikit-learn nuestra clase de LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Reduciremos el dataset de 4 a 2 dimensiones usando el parámetro n_components
 #fruit_label	fruit_name	fruit_subtype	mass	width	height	color_score
especie=fruits['fruit_label'].unique()
lda_model = LinearDiscriminantAnalysis(n_components=2).fit(df_feat , fruits['fruit_label'])
datos_tx = lda_model.transform(df_feat ).T

# Y lo dibujamos, verde=setosa / rojo=virginica / negro=versicolor
plt.scatter(datos_tx[0], datos_tx[1], c=['green' if x==1 else 'red' if x==2 else'blue' if x==3 else 'black' for x in fruits['fruit_label']])
plt.show()
# Generamos un grid de valores para probar
XX, YY = np.meshgrid(np.linspace(datos_tx[0].min(), datos_tx[0].max(), 50), np.linspace(datos_tx[1].min(), datos_tx[1].max(), 50))
pos = np.vstack([XX.ravel(), YY.ravel()])

from sklearn.neighbors import KNeighborsClassifier

# Creamos el objeto knn donde definimos el valor de k=5
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(datos_tx.T, fruits['fruit_label'])
preds_knn = knn.predict(pos.T)
plt.scatter(pos[0], pos[1],c=['green' if x==1 else 'red' if x==2 else'blue' if x==3 else 'black' for x in preds_knn])
plt.show()

preds_vx = knn.predict(datos_tx.T)
print('Precisión: ' + str(np.array(preds_vx == fruits['fruit_label']).mean()))