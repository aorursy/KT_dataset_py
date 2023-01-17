# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv")

pd.set_option("display.max_columns", 200)

data.head()
data["('D6', 'anonymized_role')"].value_counts()
data["('P8', 'degreee_level')"].value_counts()
data["('P10', 'job_situation')"].value_counts()
data["('P22', 'most_used_proggraming_languages')"].value_counts()
data["('P35', 'data_science_plataforms_preference')"].value_counts()
salario_converter = {

    'de R$ 1.001/mês a R$ 2.000/mês': 1500., 

    'de R$ 2.001/mês a R$ 3000/mês': 2500.,

    'de R$ 4.001/mês a R$ 6.000/mês': 5000., 

    'de R$ 6.001/mês a R$ 8.000/mês': 7000.,

    'de R$ 3.001/mês a R$ 4.000/mês': 3500.,

    'de R$ 8.001/mês a R$ 12.000/mês': 10000.,

    'de R$ 12.001/mês a R$ 16.000/mês': 14000.,

    'Menos de R$ 1.000/mês': 1000.,

    'de R$ 16.001/mês a R$ 20.000/mês': 18000.,

    'de R$ 20.001/mês a R$ 25.000/mês': 22500., 

    'Acima de R$ 25.001/mês': 25000.

}



data['salario_numerico'] = data["('P16', 'salary_range')"].map(salario_converter)
data.groupby(["('P22', 'most_used_proggraming_languages')"])['salario_numerico'].mean().sort_values()
data.groupby(["('D6', 'anonymized_role')"])['salario_numerico'].mean().sort_values()
fig,ax = plt.subplots(1,1, figsize=(20,10))

plote = sns.barplot(x="('D6', 'anonymized_role')", y="salario_numerico", data=data)

plote.set_xticklabels(plote.get_xticklabels(), rotation=90)
ds = data[data["('D6', 'anonymized_role')"] == 'Data Scientist/Cientista de Dados']

fig,ax = plt.subplots(1,1, figsize=(15,5))

plt.hist(ds["salario_numerico"], bins=7)

plt.axvline(ds['salario_numerico'].mean(), color='k', linewidth=5)

plt.axvline(ds['salario_numerico'].median(), color='g', linewidth=5)
fig,ax = plt.subplots(1,1, figsize=(8,8))

degree = data.groupby(["('D6', 'anonymized_role')", "('D3', 'anonymized_degree_area')"]).size().unstack()#.fillna(0)

degree = degree.div(degree.sum(axis=1), axis=0)

sns.heatmap(degree, cmap='viridis')
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ["('P17', 'time_experience_data_science')"]

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

data_t = ohe.fit_transform(data[cols_selected+['salario_numerico']])



#scaler = MaxAbsScaler()

X = data_t.iloc[:,:-1]

y = data_t.iloc[:, -1].fillna(data_t.iloc[:, -1].mean()) / data_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
data.groupby("('P17', 'time_experience_data_science')")['salario_numerico'].mean().sort_values()
from category_encoders import OneHotEncoder



#scaler = MaxAbsScaler()

X = data[data.filter(regex='P20', axis=1).columns]

y = data.iloc[:, -1].fillna(data.iloc[:, -1].mean()) / data.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
data.groupby(["('P19', 'is_data_science_professional')"])['salario_numerico'].mean()
ds_pro = data[data["('P19', 'is_data_science_professional')"] == 1]

ds_pro["('D6', 'anonymized_role')"].value_counts()
ds_pro = data[data["('P19', 'is_data_science_professional')"] == 1]

ds_pro["('P5', 'living_state')"].value_counts()
not_ds_pro = data[data["('P19', 'is_data_science_professional')"] == 0]

not_ds_pro["('P5', 'living_state')"].value_counts()
ds_pro["('P1', 'age')"].hist(figsize=(10,5), bins=20)
ds_pro.groupby(["('P8', 'degreee_level')"])['salario_numerico'].mean().sort_values()
ds_pro.groupby(["('P8', 'degreee_level')"])['salario_numerico'].size().sort_values()