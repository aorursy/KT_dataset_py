# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

#import csv as csv



data_path = "../input/"

#from numpy import genfromtxt

#my_data = genfromtxt(data_path+"train_ver2.csv", delimiter=",")

#my_data



#csv_file_object = csv.reader(open(data_path+"train_ver2.csv", "r")) 

#header = csv_file_object.next() 

#data=[] 



#for row in csv_file_object:

#    data.append(row)

#data = np.array(data) 

#data



with open(data_path+"train_ver2.csv", "r") as ins:

    array = []

for line in ins:

    array.append(line)



df = pd.DataFrame(array),

df

#data = pd.read_csv(data_path+"train_ver2.csv", keep_default_na=False, na_values=[""], low_memory=False)
data_path = "../input/"

Customer = pd.read_csv(data_path+"train_ver2.csv", usecols=['ncodpers'])



Month = pd.read_csv(data_path+"train_ver2.csv", usecols=['fecha_dato'], parse_dates=['fecha_dato'])

Month['month'] = Month['fecha_dato'].apply(lambda x: x.month)



#cols = ["ind_empleado","pais_residencia","sexo","age","ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","renta","segmento"]

#col_data = pd.read_csv(data_path+"train_ver2.csv", usecols = [cols])

#col_data = col_data.fillna(NA)