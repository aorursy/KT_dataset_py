# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("/kaggle/input/mental-health-in-tech-survey/survey.csv")
df = df.dropna(how='any')

CUANTITATIVA= df[["Age","no_employees"]]
CUALITATIVA= df[["Gender","self_employed","work_interfere","anonymity","leave"]]
import seaborn as sns 
import matplotlib.pyplot as plt

df["Age"].sort_values().unique()
df.loc[~df["Age"] .between(18,72),"Age"] =np.nan

df0 = df[df["treatment"]=='No']
df1 = df[df["treatment"]=='Yes']

mean_df1 = df1["Age"].mean()
mean_df0 = df0["Age"].mean()


df1["Age"]=df1["Age"].replace(np.nan, mean_df1)
df0["Age"]=df0["Age"].replace(np.nan, mean_df0)


df["Gender"] = df["Gender"].apply(lambda x: x.strip())
female = 'Female', 'female', 'F', 'f', 'Woman', 'Female', 'Female (trans)'
male = 'Male', 'male', 'M', 'm', 'Make', 'Male', 'Cis Male', 'Man'


df["Gender"]=df["Gender"].replace(female, 'Female')
df["Gender"]=df["Gender"].replace(male, 'Male')

df.loc[(df["Gender"] != "Female") & (df["Gender"] != "Male"), "Gender"] = "Other" 
df_new =df[["Age","Gender","treatment","family_history","no_employees","mental_health_consequence","mental_vs_physical","obs_consequence","tech_company","seek_help","remote_work","self_employed","phys_health_interview","state","benefits"]]

edad_JOVENES=len(df_new[(df_new["mental_health_consequence"]=="No")& (df_new["Age"]<21)])
edad_ADULTOS=len(df_new[(df_new["mental_health_consequence"]=="No")& (df_new["Age"]>=21)& (df_new["Age"]<50)])
edad_MAYORES=len(df_new[(df_new["mental_health_consequence"]=="No")& (df_new["Age"]>=50)])

etiqueta =  ['JOVENES', 'ADULTOS','MAYORES']
total = [edad_JOVENES,edad_ADULTOS,edad_MAYORES]
colores = ['pink','magenta',"orange"]
plt.pie(total, labels = etiqueta, colors=colores ,shadow = True, explode = (0.0, 0.3, 0.0), autopct = '%1.1f%%',textprops={'fontsize':14})  
plt.show() 
NO1=len(df_new[(df_new["mental_vs_physical"]=="No")& (df_new["tech_company"]=="No")])
YES1=len(df_new[(df_new["mental_vs_physical"]=="Yes")& (df_new["tech_company"]=="Yes")])

etiquetas=["NO","YES"]
valor=[NO1,YES1]
plt.bar(etiquetas,valor, color=["blue","red"])
plt.show()
NO1=len(df_new[(df_new["tech_company"]=="No")& (df_new["seek_help"]=="No")])
YES1=len(df_new[(df_new["tech_company"]=="Yes")& (df_new["seek_help"]=="Yes")])

etiquetas=["NO","YES"]
valor=[NO1,YES1]
plt.bar(etiquetas,valor, color=["orange","green"])
plt.show()
valores=df_new[(df_new["Gender"]=="Female") & (df_new["tech_company"]=="Yes")]["tech_company"].value_counts()
valores1=df_new[(df_new["Gender"]=="Male") & (df_new["tech_company"]=="Yes")]["tech_company"].value_counts()

etiqueta=["Mujeres","Hombres"]

total = [valores,valores1]
colores = ['magenta',"orange"]
plt.pie(total, labels = etiqueta, colors=colores ,shadow = True, explode = (0.0, 0.3), autopct = '%1.1f%%',textprops={'fontsize':14})  
plt.show() 
edad_JOVENES=len(df_new[(df_new["treatment"]=="Yes")& (df_new["self_employed"]=="Yes")& (df_new["tech_company"]=="Yes") & (df_new["Age"]>18)])
edad_ADULTOS=len(df_new[(df_new["remote_work"]=="Yes")& (df_new["self_employed"]=="Yes")& (df_new["tech_company"]=="Yes")& (df_new["Age"]<=30)& (df_new["Age"]<50)])
edad_MAYORES=len(df_new[(df_new["remote_work"]=="Yes")& (df_new["self_employed"]=="Yes")& (df_new["tech_company"]=="Yes")& (df_new["Age"]>=50)])

etiqueta =  ['JOVENES', 'ADULTOS','MAYORES']
total = [edad_JOVENES,edad_ADULTOS,edad_MAYORES]
colores = ['pink','c',"g"]
plt.pie(total, labels = etiqueta, colors=colores ,shadow = True, explode = (0.1, 0.3, 0.1), autopct = '%1.1f%%',textprops={'fontsize':20})  
plt.show() 
valores=len(df_new[(df_new["remote_work"]=="No")& (df_new["Gender"]=="Male")& (df_new["phys_health_interview"]=="No")])
valores1=len(df_new[(df_new["remote_work"]=="No")& (df_new["Gender"]=="Female")& (df_new["phys_health_interview"]=="No")])

etiquetas=['MASCULINO','FEMENINO']
valor=[valores,valores1]
plt.bar(etiquetas,valor, color=["c","pink"])
plt.show()
valores=len(df_new[(df_new["family_history"]=="Yes")& (df_new["treatment"]=="Yes")])
valores1=len(df_new[(df_new["family_history"]=="Yes")& (df_new["treatment"]=="No")])

etiquetas=['Con Tratamiento','Sin Tratamiento']
valor=[valores,valores1]
plt.bar(etiquetas,valor, color=["m","c"])
plt.show()
fig, ax = plt.subplots()
etiquetas=["Masculino","Femenino"]

valores=len(df_new[(df_new["no_employees"]== "More than 1000") & (df_new["tech_company"]== "Yes") & (df_new["Gender"]== "Male")])
valores1=len(df_new[(df_new["no_employees"]== "More than 1000") & (df_new["tech_company"]== "Yes") & (df_new["Gender"]== "Female")])

x = np.arange(len(etiquetas))  # the label locations
ancho=0.6

valor=[valores,valores1]

plt.bar(etiquetas, valor, ancho, color=["c","m"])
ax.set_xticks(x)
ax.set_xticklabels(etiquetas)
plt.show()
