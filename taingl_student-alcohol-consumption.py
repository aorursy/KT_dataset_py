# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualize
import matplotlib.pyplot as plt # visualize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_por = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-por.csv")
df_por.head()
df_por.columns
df_por.describe()
df_maths = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")
df_maths.head()
df_maths.columns
df_maths.describe()
df_por.groupby("Dalc").mean()
df_maths.groupby("Dalc").mean()
df_por.groupby("Walc").mean()
df_maths.groupby("Walc").mean()
df_class = pd.merge(df_por, df_maths, how='inner', on= ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
print((df_class.shape)) # 382 students
df_all  = pd.concat([df_por,df_maths],ignore_index=True).drop_duplicates().reset_index(drop=True)
df_all.columns
df_all
#Les attributs "payé" est spécifique au cours plutôt que spécifique à l'élève, 
#je l'élimine donc de la liste des attributs
#df_all.drop(['paid'], axis=1)
plt.figure(figsize=(15,15))
sns.heatmap(df_all.corr(),annot = True,fmt = ".2f",cbar = True)
plt.xticks(rotation=90)
plt.yticks(rotation = 0)
# La consommation d'alcool dans la semaine entière : lundi au dimanche
df_all['total_consumption'] = df_all['Dalc'] + df_all['Walc']
df_all.groupby("total_consumption").mean()
dalc_sum = df_all['Dalc'].value_counts(normalize = True)*100
dalc_sum
# Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high) 
plt.title('Partition de niveau de consommation d\'alcool dans les jours de la semaine dans les 2 classes')
dalc_sum.plot.pie(figsize=(8, 8),autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(["Niveau 1","Niveau 2","Niveau 3","Niveau 4","Niveau 5"])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
walc_sum = df_all['Walc'].value_counts(normalize = True)*100
walc_sum
# Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) 
plt.title('Partition de niveau de consommation d\'alcool pendant les week-end dans les 2 classes')
walc_sum.plot.pie(figsize=(8, 8), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(["Niveau 1","Niveau 2","Niveau 3","Niveau 4","Niveau 5"])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
# age et absence
df_all[['age','absences']].head(n=10)
# le graphe sur la moyenne de la colonne age et absence
means_a_abs = df_all[['age','absences']].groupby(['age']).mean()
means_a_abs.plot.bar()
# selection des colonnes à étudier
a_trav_fail_free_go_abs = df_all[['age','traveltime','failures','freetime','goout','absences','studytime','total_consumption']]
all_columns = a_trav_fail_free_go_abs.groupby("age").mean()
all_columns
conso10 = a_trav_fail_free_go_abs[a_trav_fail_free_go_abs['total_consumption'] == 10] 
conso10[['age','absences','freetime','goout','failures']].groupby(['age']).mean().plot.bar(title = " Observation pour une consommation très élevée dans toute la semaine ")
total_consumption = a_trav_fail_free_go_abs[a_trav_fail_free_go_abs['total_consumption'] < 11] 
age_alc = total_consumption.groupby("total_consumption").mean()
age_alc
age_alc[['absences','goout','freetime']].plot(kind ='bar',title = "Lien entre les abscences à l'école, temps libre après l'école et les sorties avec les ami(e)s selon la consommation d'alcool dans la semaine")
df_all_family = df_all[['school', 'address', 'famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','famsup','nursery','higher','internet','famrel','Dalc','Walc','total_consumption' ]]
df_all_family
# Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
plt.title("Partition des élèves les 2 classes selon l'éducation de la mère")
df_all.Medu.value_counts(normalize=True).plot.pie(figsize=(5, 5), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(["none", "primary education (4th grade)", "5th to 9th grade", "2nd education","higher education"])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
medu_df = df_por.groupby(['Medu']).mean()
plt.rcParams['figure.figsize'] = (20, 10)
medu_df[['G1','G2','G3']].plot()
# Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
plt.title("Partition des élèves les 2 classes selon l'éducation du père")
df_all.Fedu.value_counts(normalize=True).plot.pie(figsize=(5, 5), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(["none", "primary education (4th grade)", "5th to 9th grade", "2nd education","higher education"])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
fedu_df = df_por.groupby(['Fedu']).mean()
plt.rcParams['figure.figsize'] = (20, 10)
fedu_df[['G1','G2','G3']].plot()
# Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
plt.title("Partition des élèves les 2 classes selon le travail de la mère")
df_all.Mjob.value_counts(normalize=True).plot.pie(figsize=(5, 5), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(['teacher', 'health','services', 'at_home', 'other'])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
mjob_df = df_all.groupby(['Mjob']).mean()
plt.rcParams['figure.figsize'] = (20, 10)
mjob_df[['G1','G2','G3']].plot(kind='bar')
# Fjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
plt.title("Partition des élèves les 2 classes selon le travail de la père")
df_all.Fjob.value_counts(normalize=True).plot.pie(figsize=(5, 5), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(['teacher', 'health','services', 'at_home', 'other'])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
fjob_df = df_all.groupby(['Fjob']).mean()
plt.rcParams['figure.figsize'] = (20, 10)
fjob_df[['G1','G2','G3']].plot(kind='bar')
# famsup - family educational support (binary: yes or no) 
plt.title("Partition des élèves les 2 classes selon s'il existe un soutien éducatif familial")
df_all.famsup.value_counts(normalize=True).plot.pie(figsize=(5, 5), autopct = lambda x: str(round(x, 2)) + '%')
plt.legend(['yes', 'no'])
plt.rcParams['figure.figsize'] = (15, 5)
plt.show()
rural_data = df_all[df_all['address']== 'R']
urban_data = df_all[df_all['address']== 'U']
#rural area
rural_df = rural_data[['sex', 'total_consumption']]
rural_df_h = rural_df[rural_df['sex'] == 'M']
rural_df_h = rural_df_h.groupby(['total_consumption']).count()
rural_df_h.plot(kind='bar', title = "Consommation d'alcool chez les hommes")
rural_df = rural_data[['sex', 'total_consumption']]
rural_df_f = rural_df[rural_df['sex'] == 'F']
rural_df_f = rural_df_f.groupby(['total_consumption']).count()
rural_df_f.plot(kind='bar', title = "Consommation d'alcool chez les femmes")
df_all = df_all[['sex', 'total_consumption']]
df_all_h = df_all[df_all['sex'] == 'M']
df_all_h = df_all.groupby(['total_consumption']).count()
df_all_h.plot(kind='bar', title = "Consommation d'alcool chez les hommes")
df_all = df_all[['sex', 'total_consumption']]
df_all_f = df_all[df_all['sex'] == 'F']
df_all_f = df_all.groupby(['total_consumption']).count()
df_all_f.plot(kind='bar', title = "Consommation d'alcool chez les femmes")
