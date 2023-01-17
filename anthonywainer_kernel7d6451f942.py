# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install --upgrade statsmodels

!sudo apt install -y graphviz

!pip install graphviz
import pickle

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, plot_confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold

from sklearn import datasets



import plotly.graph_objs as go

import ipywidgets as widgets



pd.set_option("display.max_columns", 50)



%matplotlib inline
train_url = '/kaggle/input/inf648-curso-de-aprendizaje-automtico/train.csv'

dataset=pd.read_csv(train_url)

print(dataset.shape)

dataset.head()
def deuda_union(x,y):

    if ((x==2) | (y==2)):

      return "2"

    elif ((x==0 & y==1) | (x==1 & y==0) | (x==1 & y==1)):

      return "1"

    else:

      return "0"
def preprocess_dataset(dataset):

  preprocessed_dataset = dataset.copy()



  if "ID" in preprocessed_dataset:

    preprocessed_dataset.drop(["ID"], axis=1, inplace=True)

    #ID no es predictivo



  preprocessed_dataset["Trabajo"].replace(to_replace =["emprendedor","propio-empleado"], value="independiente", inplace =True)

  preprocessed_dataset["Trabajo"].replace(to_replace =["administrador","operador-industria","tecnico","ejecutivo","servicio"], value="dependiente", inplace =True)

  preprocessed_dataset["Trabajo"].replace(to_replace =["ama-de-llaves","unemployed","retirado","estudiante"], value="no-asalariado", inplace =True)

    #Agrupamos los tipos de trabajo por tipo: dependiente, independiente, no-asalariado y desconocido



    #Mantenemos las clases del estado civil



  preprocessed_dataset["Grado Educacion"].replace(to_replace =["primaria.4a","primaria.6a","primaria.9a"], value="primaria",inplace =True)

    #Agrupamos los grados de instrcción primarios en sólo uno



  if "Credito por Default " in preprocessed_dataset:

    preprocessed_dataset.drop(["Credito por Default "], axis=1, inplace=True)

    #Esta información no aporta al análisis



  preprocessed_dataset["Dueda Casa"].replace({"no": "0", "si": "1", "desconocido":"2"}, inplace=True)

  preprocessed_dataset["Deuda Personal"].replace({"no": "0", "si": "1", "desconocido":"2"}, inplace=True)

  preprocessed_dataset.insert(6,"Deuda",preprocessed_dataset.apply(lambda x: deuda_union(int(x["Dueda Casa"]),int(x["Deuda Personal"])), axis=1)) 



  preprocessed_dataset.drop(["Dueda Casa","Deuda Personal"], axis=1, inplace=True)

  





  if "Contacto" in preprocessed_dataset:

    preprocessed_dataset.drop(["Contacto"], axis=1, inplace=True)

    #Esta información no aporta al análisis



  preprocessed_dataset["Mes"].replace(to_replace =["enero","febrero","marzo"], value="1T",inplace =True)

  preprocessed_dataset["Mes"].replace(to_replace =["abril","mayo","junio"], value="2T",inplace =True)

  preprocessed_dataset["Mes"].replace(to_replace =["julio","agosto","septiembre"], value="3T",inplace =True)

  preprocessed_dataset["Mes"].replace(to_replace =["octubre","noviembre","diciembre"], value="4T",inplace =True)

  #Agrupamos los meses por trimestres

 

  if "Dia Semana" in preprocessed_dataset:

    preprocessed_dataset.drop(["Dia Semana"], axis=1, inplace=True)

    #Esta información no aporta al análisis



  if "pdias" in preprocessed_dataset:

    preprocessed_dataset.drop(["pdias"], axis=1, inplace=True)

    #Esta información no aporta al análisis (la mayorpia es 999)



  if "Llamadas previas" in preprocessed_dataset:

    preprocessed_dataset.drop(["Llamadas previas"], axis=1, inplace=True)

    #Esta información no aporta al análisis (la mayorpia es 999)



  

  label_encoder = LabelEncoder()



  for column in preprocessed_dataset.columns:

    if not pd.api.types.is_numeric_dtype(preprocessed_dataset[column]):

      preprocessed_dataset[column] = label_encoder.fit_transform(preprocessed_dataset[column])



      print("Para la columna '{}', la codificación fue: {}".format(column, dict(enumerate(label_encoder.classes_))))

  

  return preprocessed_dataset
#Transformamos los datos a valores numéricos

preprocessed_dataset = preprocess_dataset(dataset)
preprocessed_dataset.head()
def create_bins(df, column, bins_dict):

  bins_list = bins_dict["bins_list"]

  bins_number = bins_dict["bins_number"]



  if (bins_list):

    data_for_bins = pd.cut(df[column], bins=bins_list, precision=0, duplicates="drop")



  else:

    data_for_bins = df[column]



    min_value = data_for_bins.min()

    max_value = data_for_bins.max()



    repetitions_of_min_value = len(data_for_bins[data_for_bins == min_value])

    repetitions_of_max_value = len(data_for_bins[data_for_bins == max_value])



    if repetitions_of_min_value > 1:

      #Si el valor mínimo se repite entonces agregamos un valor mínimo falso para balancear mejor los quantiles

      data_for_bins = data_for_bins.append(pd.Series([min_value - 1]), ignore_index=True)



    if repetitions_of_max_value > 1:

      #Si el valor máximo se repite entonces agregamos un valor máximo falso para balancear mejor los quantiles

      data_for_bins = data_for_bins.append(pd.Series([max_value + 1]), ignore_index=True)

    

    #Se usa duplicates="drop" por si los límites de los bins se repiten. No sucederá este caso para el primer y último bin 

    quantiles = pd.qcut(data_for_bins, q=bins_number, precision=0, duplicates="drop")

    

    data_for_bins = quantiles



    if repetitions_of_max_value > 1:

      #Eliminamos el valor máximo que introducimos

      data_for_bins = data_for_bins[:-1]

    

    if repetitions_of_min_value > 1:

      #Eliminamos el valor máximo que introducimos

      data_for_bins = data_for_bins[:-1]



  dictionary_of_intervals = dict()



  for index, interval in enumerate(data_for_bins.cat.categories):

    dictionary_of_intervals[interval] = int(index)

    

  df.reset_index(drop=True, inplace=True)



  bins_column = "Bins_for_" + column



  #df[bins_column] = data_for_bins

  df.drop([column], axis=1, inplace=True)



  df[bins_column + "_index"] = data_for_bins



  df.replace({bins_column + "_index": dictionary_of_intervals}, inplace=True)

  print("Bins Created")
bins_dict = {

    "Edad": {"bins_number": 0, "bins_list": [0,20,30,40,50,70,100]},

	  "Duracion": {"bins_number": 5, "bins_list": []},

	  "Llamadas": {"bins_number": 4, "bins_list": []},

	  "tasaVarEmp": {"bins_number": 5, "bins_list": []},

	  "indicador consumidor precio": {"bins_number": 5, "bins_list": []},

	  "indicador confianza consumidor": {"bins_number": 5, "bins_list": []},

	  "indicador macro": {"bins_number": 5, "bins_list": []},

	  "ind. Cuartil emp": {"bins_number": 5, "bins_list": []}

}
for column, bins_list in bins_dict.items():

  create_bins(preprocessed_dataset, column, bins_list)
preprocessed_dataset.head()
x = preprocessed_dataset.drop(["Target"], axis=1)



y = preprocessed_dataset["Target"]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
parameters_mdtc = {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 100}

pamameters_RF = {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 20, 'n_estimators': 500}

pamameters_XGC =  {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 50}
# preparar modelos para comparar

models = [

    ('TreeClassifier', DecisionTreeClassifier(**parameters_mdtc)),

    ('RandomForest', RandomForestClassifier(**pamameters_RF)),

    ('XGBClassifier', XGBClassifier(**pamameters_XGC))

]





# función para comparar modelos

def model_comparation():

    results = []

    names = []

    for name, model in models:

        kfold = KFold(n_splits=10)

        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

        results.append(cv_results)

        names.append(name)

        msg = "%s. accuracy score: %f, con std: (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

    return results, names
%%time

results, names = model_comparation()

# boxplot de comparación

fig = plt.figure()



ax = fig.add_subplot(111)

plt.title('Comparación')

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
test_url = '/kaggle/input/inf648-curso-de-aprendizaje-automtico/test.csv'

dataset_test=pd.read_csv(test_url)

print(dataset_test.shape)

dataset_test.head()
#Transformamos los datos a valores numéricos

preprocessed_dataset = preprocess_dataset(dataset_test)
for column, bins_list in bins_dict.items():

  create_bins(preprocessed_dataset, column, bins_list)
x_predict = preprocessed_dataset
model_xgb = XGBClassifier(**pamameters_XGC)

model_xgb.fit(x_train, y_train)

predict_xgc = model_xgb.predict(x_predict)
submission = pd.DataFrame({'ID':dataset_test.ID, 'Target':predict_xgc})
submission.to_csv('submission.csv', index=False)