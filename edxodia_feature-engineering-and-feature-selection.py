# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from functools import reduce 
import numpy as np

# definicion de corpus
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))
print(dictionary)
def vectorize(text): 
    vector = np.zeros(len(dictionary)) 
    for i, word in dictionary: 
        num = 0 
        for w in text: 
            if w == word: 
                num += 1 
        if num: 
            vector[i] = num 
    return vector

for t in texts: 
    print(vectorize(t))
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,1))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()
vect.vocabulary_ 
vect = CountVectorizer(ngram_range=(1,2))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()
vect.vocabulary_
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb')

n1, n2, n3, n4 = vect.fit_transform(['andersen', 'petersen', 'petrov', 'smith']).toarray()


euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)
## Algunos ejemplos utilizarán el dataset de la compañía Renthop, que se usa en la competencia 
## Two Sigma Connect: Consultas de listado de alquileres de Kaggle. 
## En esta tarea, debe predecir la popularidad de un nuevo listado de alquiler, es decir, 
### clasificar el listado en tres clases: `['low', 'medium' , 'high']`. Para evaluar las soluciones, 
### 
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Let's load the dataset from Renthop right away
with open('../input/twosigmaconnect/renthop_train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)
df.tail()
!pip install reverse_geocoder
import reverse_geocoder as revgc


revgc.search([df.latitude[1], df.longitude[2]])
df['dow'] = df['created'].apply(lambda x: pd.to_datetime(x).weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if pd.to_datetime(x).weekday() in (5, 6) else 0)
#df['is_weekend']
def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period 
    return np.cos(value), np.sin(value)
#import numpy as np
import matplotlib.pyplot as plt
xx=np.arange(0,24,1)
X=[]
Y=[]
for i in xx:
    x,y=make_harmonic_features(i)
    X.append(x)
    Y.append(y)
    
plt.plot(X, Y)
plt.show()
from scipy.spatial import distance
euclidean(make_harmonic_features(23), make_harmonic_features(1)) 
euclidean(make_harmonic_features(9), make_harmonic_features(11)) 
euclidean(make_harmonic_features(9), make_harmonic_features(21))
!pip install -q pyyaml ua-parser user-agents
import user_agents

ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'
ua = user_agents.parse(ua)

print('Is a bot? ', ua.is_bot)
print('Is mobile? ', ua.is_mobile)
print('Is PC? ',ua.is_pc)
print('OS Family: ',ua.os.family)
print('OS Version: ',ua.os.version)
print('Browser Family: ',ua.browser.family)
print('Browser Version: ',ua.browser.version)
import numpy as np
plt.style.use('seaborn-whitegrid')
size = 100
x = np.linspace(0, 10, size) 
y = x**2 + 10 - (20 * np.random.random(size))
plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
import lightgbm as lgb
overfit_model = lgb.LGBMRegressor(silent=False, min_child_samples=5)
overfit_model.fit(x.reshape(-1,1), y)
 
#predicted output from the model from the same input
prediction = overfit_model.predict(x.reshape(-1,1))

plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
plt.plot(x,prediction,color='r')
monotone_model = lgb.LGBMRegressor(min_child_samples=5, 
                                   monotone_constraints="1")
monotone_model.fit(x.reshape(-1,1), y)
#predicted output from the model from the same input
prediction = monotone_model.predict(x.reshape(-1,1))
plt.figure(figsize=(10,5))
plt.plot(x,y,'o')
plt.plot(x,prediction,color='r')
from sklearn.metrics import mean_squared_error as mse
 
size = 1000000
x = np.linspace(0, 10, size) 
y = x**2  -10 + (20 * np.random.random(size))
 
print ("MSE del modelo por defecto", mse(y, overfit_model.predict(x.reshape(-1,1))))
print ("MSE del modelo Monotono", mse(y, monotone_model.predict(x.reshape(-1,1))))
# EJEMPLO: usando Shapiro-Wilk - Test
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.stats import shapiro
import numpy as np

#Generamos 1000 números aleatorios [1,10]
data = beta(1, 10).rvs(1000).reshape(-1, 1)

# Realizamos la prueba de Shapiro-Wilk para la normalidad.
shapiro_stat,shapiro_p_value=shapiro(data)

#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')
#
shapiro_stat,shapiro_p_value=shapiro(StandardScaler().fit_transform(data))

# Con el valor p tendríamos que rechazar la hipótesis nula de normalidad de los datos.
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')
from sklearn.preprocessing import MinMaxScaler
shapiro_stat,shapiro_p_value=shapiro(MinMaxScaler().fit_transform(data))

# Con el valor p tendríamos que rechazar la hipótesis nula de normalidad de los datos.
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')

(data - data.min()) / (data.max() - data.min()) 
from scipy.stats import lognorm

data = lognorm(s=1).rvs(1000)
shapiro_stat,shapiro_p_value=shapiro(data)
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')
# graficando
plt.figure(figsize=(10,5))
plt.hist(data, bins=50)

shapiro_stat,shapiro_p_value=shapiro(np.log(data))
#conclusión
if shapiro_p_value > 0.05:
    print('con 95% de confianza los datos son similares a una distribución normal')
else:
    print('con 95% de confianza los datos NO son similares a una distribución normal')
# graficando
plt.figure(figsize=(10,5))
plt.hist(np.log(data), bins=50)
# ¡Dibujemos !
import statsmodels.api as sm


# Tomemos la característica de precio del dataset de Renthop y filtremos los valores más extremos para mayor claridad.
price = df.price[(df.price <= 20000) & (df.price > 500)]
price_log = np.log(price)

# usamos transformaciones
price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
sm.qqplot(price, loc=price.mean(), scale=price.std())
sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std())
sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std())
sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std())
rooms = df["bedrooms"].apply(lambda x: max(x, .5))
# Evitar la división por cero; .5 se elige más o menos arbitrariamente
df["price_per_bedroom"] = df["price"] / rooms
df["price_per_bedroom"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline  
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
data.head(n=10)
count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(10)
df1 = pd.DataFrame.from_dict(count1)
print(df1.head())
df1 = df1.rename(columns={0: "palabras non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(10)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "palabras spam", 1 : "count_"})
df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["palabras non-spam"]))
plt.xticks(y_pos, df1["palabras non-spam"])
plt.title('Palabras frecuentes en mensajes no-spam')
plt.xlabel('Palabras')
plt.ylabel('Numero')
plt.show()
df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["palabras spam"]))
plt.xticks(y_pos, df2["palabras spam"])
plt.title('Palabras frecuentes en mensajes spam ')
plt.xlabel('Palabras')
plt.ylabel('numero')
plt.show()

#usando CountVectorizer
#f = feature_extraction.text.CountVectorizer(stop_words = 'english')
#feat = feature_extraction.text.CountVectorizer(stop_words = 'english', ngram_range=(1,2)) ## usando n_gram?
feat = feature_extraction.text.CountVectorizer(stop_words = 'english')

X = feat.fit_transform(data["v2"])

np.shape(X)
data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.2, random_state=42)
print([np.shape(X_train), np.shape(X_test)])
# usamos Support Vector Machine de :https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svc = svm.SVC()
svc.fit(X_train, y_train)
score_train = svc.score(X_train, y_train)
score_test = svc.score(X_test, y_test)
# para validar debe usar una matriz de confusión usando el siguiente código:
matr_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = matr_confusion_test, columns = ['Prediccion spam', 'Prediccion no-spam'],
            index = ['Real spam', 'Real no-spam'])