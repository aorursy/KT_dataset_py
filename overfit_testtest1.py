# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"));



# Any results you write to the current directory are saved as output.
limit_rows   = 7000000;

df_train     = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,

                                                    "age": str,  

                                                    "ind_nuevo": str,

                                                    "antiguedad": str,

                                                    "indrel": str,

                                                    "ult_fec_cli_1t": str,

                                                    "conyuemp": str,

                                                    "ind_actividad_cliente": str,

                                                    "indext":str}, 

                           na_values=[' NA','     NA'],keep_default_na=True,nrows=limit_rows);

# TODO: fix type of tipodom, cod_prov

# NOTICE: indrel_1mes'P is changed to 0

df_train["age"].fillna(-1,inplace=True);

df_train["ind_nuevo"].fillna(-1,inplace=True);

df_train["antiguedad"].fillna(-1,inplace=True);

df_train["indrel"].fillna(-1,inplace=True);

df_train["ind_actividad_cliente"].fillna(-1,inplace=True);

df_train["indrel_1mes"].fillna(-1,inplace=True);

df_train["indrel_1mes"] = df_train["indrel_1mes"].replace("P","0");

df_train["age"] = df_train["age"].astype('int32');

df_train["ind_nuevo"] = df_train["ind_nuevo"].astype('int8');

df_train["antiguedad"] = df_train["antiguedad"].astype('int32');

df_train["indrel"] = df_train["indrel"].astype('int8');

df_train["ind_actividad_cliente"] = df_train["ind_actividad_cliente"].astype('int8');

df_train["indrel_1mes"] = df_train["indrel_1mes"].astype(float).astype('int8');
print(df_train.dtypes);
print(df_train.iloc[0]);
print("Number of rows in train : ", df_train.shape[0])
unique_ncodpers = df_train.ncodpers.unique();

print("Number of customers in train : ", len(set(unique_ncodpers)));
example1 = df_train.loc[df_train['ncodpers'] == (unique_ncodpers[2])];

from IPython.display import HTML

HTML(example1.to_html())
# check for change in columns

df_train_x_part = df_train [["ncodpers", "ind_empleado", "pais_residencia", "sexo", "ind_nuevo", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente", "renta"]];

list_changed_ncodpers = list();



num = 19;

ones = np.ones(num);

temp = unique_ncodpers[0:1000];

for i in temp:

    example = df_train_x_part.loc[df_train['ncodpers'] == i];

    for j in range (0, num):

        if (pd.value_counts(example.iloc[:, j]).size > 1):

            list_changed_ncodpers.append(i);

            break;

print (len(list_changed_ncodpers))    



example2 = df_train.loc[df_train['ncodpers'] == (list_changed_ncodpers[1])];

from IPython.display import HTML

HTML(example2.to_html())


print (example1.iloc[:, 5])

print (type(example1.iloc[1, 5]))

print (pd.value_counts(example1.iloc[:, 5]))