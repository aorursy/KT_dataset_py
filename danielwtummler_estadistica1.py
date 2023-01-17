# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv")

#df.head()
#Replace de "-" y " "

df = df.replace("-", np.nan)

df = df.replace(" ", np.nan)



#Replace de los np.nan por moda

for col in df.columns:

    df[col] = df[col].replace(np.nan, stats.mode(df[col])[0][0])
#Cambio de tipo

for col in df.columns:

    try:

        df[col] = df[col].astype("float")

    except:

        df[col] = df[col].astype("object")
#Rango

for col in df.columns:

    if df[col].dtype == "float":

        print("Rango de la columna:", col, "es:", max(df[col]) - min(df[col]))
#Promedio o Moda

for col in df.columns:

    if df[col].dtype == "object":

        print("La moda de la columna", col, "es:", stats.mode(df[col])[0])

    else:

        print("El promedio de la columna", col, "es:", np.mean(df[col]))
#Rango, varianza, desviacion estandar

for col in df.columns:

    if df[col].dtype == "float":

        print(col)

        print("Rango:", max(df[col]) - min(df[col]))

        print("Varianza:", np.var(df[col]))

        print("Desviacion Estandar:", np.std(df[col]))

        print()
#Coeficiente de Asimetria, Curtosis

for col in df.columns:

    if df[col].dtype == "float":

        print("Columna: ", col)

        print("Coeficiente de asimetria:", stats.skew(df[col], axis = 0, bias = True))

        print("Curtosis:", stats.kurtosis(df[col], axis = 0, bias = True))

        print()
#Porcentaje de datos dentro de desviaciones standard

for col in df.columns:

    if df[col].dtype == "float":

        

        print("Columna:", col)

        mean = round(np.mean(df[col]), 2)

        std = round(np.std(df[col]), 2)

        

        print("Mean:", mean, "Std:", std, "Total:", df[col].shape[0])

        

        for i in range(1,3):

            

            print("Para", i, "std's")

            print("Porcentaje", round(df[df[col].between(mean - std*i, mean + std*i)].shape[0]*100/df.shape[0], 2))

            

        print("***********************************************************")

        print()    

    
#Grafica

for col in df.columns:

    if df[col].dtype == "float":

        

        mean = np.mean(df[col])

        std = np.std(df[col])

        

        skewness = stats.skew(df[col], axis = 0, bias = True)

        kurtosis = stats.kurtosis(df[col], axis = 0, bias = True)



        sns.distplot(df[col], hist=False, color="blue", kde_kws={'bw': 0.1})



        plt.plot([mean - 2*std for i in range(100)], np.linspace(0, 0.5, 100), color = "green", alpha = 0.4) #mean - 2*std

        plt.plot([mean - std for i in range(100)], np.linspace(0, 0.5, 100), color = "green", alpha = 0.4) #mean - std

        plt.plot([mean for i in range(100)], np.linspace(0, 0.5, 100), color = "red", alpha = 0.4) #mean

        plt.plot([mean + std for i in range(100)], np.linspace(0, 0.5, 100), color = "green", alpha = 0.4) #mean + std

        plt.plot([mean + 2*std for i in range(100)], np.linspace(0, 0.5, 100), color = "green", alpha = 0.4) #mean + 2*std

        

        if skewness > 0:

            if kurtosis > 0:

                plt.title("Skewness postivo, kurtosis positvo.")

            else:

                plt.title("Skewness postivo, kurtosis negativo.")

        else:

            if kurtosis > 0:

                plt.title("Skewness negativo, kurtosis positvo.")

            else:

                plt.title("Skewness negativo, kurtosis negativo.")



        plt.show()