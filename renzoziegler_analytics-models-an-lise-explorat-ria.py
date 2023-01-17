# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import pylab

import seaborn as sns

import datetime as dt

sns.set_style("whitegrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')

data.head()

#Leia o dataset, e atribua-o a uma variavel
data.info()

#Nenhuma coluna tem valores nulos
data.isnull().any()
data.columns = ['patient_id',

                 'appointment_id',

                 'gender','schedule_day',

                 'appointment_day',

                 'age',

                 'neighborhood',

                 'scholarship',

                 'hypertension',

                 'diabetes',

                 'alcoholism',

                 'handicap',

                 'sms_received',

                 'no_show']

data.info()
sum(data.duplicated())

#Nenhum valor duplicado
data['gender'].value_counts()
data['no_show'].value_counts()
data.describe()
#Existe algum registro com idade -1!!!

data[data['age']< 0]
data = data.drop(data[data['age']< 0].index)
#E o handicap?

data.handicap.value_counts()
#O que seria handicap acima de 1?? Como não há uma explicacao clara, e o número de registros é pequeno,

#vamos eliminar esses registros

data = data.drop(data[data['handicap'] > 1].index)
print(type(data['schedule_day'][0]))

print(type(data['appointment_day'][0]))
data['waitdays'] = pd.to_datetime(data['appointment_day'])- pd.to_datetime(data['schedule_day'])

data.head()
data['waitdays'] = data['waitdays'].apply(lambda x: x.days)

data['waitdays'].describe()
# Vamos eliminar valores menores que -1

data.drop(data[data.waitdays < -1].index, inplace=True)
#Vamos eliminar as variaveis que não vamos utilizar

#Ids não são relevantes para nossas análises (patient_id e appointment_id)

data.drop(['patient_id', 'appointment_id'], axis = 1, inplace = True)
#Vamos transformar as datas em tipo data, e extrair o dia da semana

data['appointment_day'] = data['appointment_day'].apply(lambda x : x.replace('T00:00:00Z', ''))

data['appointment_day'] = pd.to_datetime(data['appointment_day'])



data['appointment_weekday'] = data['appointment_day'].apply(lambda x : dt.datetime.strftime(x, '%A'))

data['appointment_month'] = data['appointment_day'].apply(lambda x : dt.datetime.strftime(x, '%B'))
#Vamos acertar agora os valores da coluna no_show

#Vamos substituir os valores texto (Yes, No) por números (1, 0)

data['no_show_str'] = data['no_show']

data['no_show'] = data['no_show'].replace({'Yes' : 1, 'No' : 0})

data.no_show.value_counts()
#proporção de no show

data['no_show'].value_counts(normalize = True)
#número de pacientes que não apareceram

len(data[data['no_show'] == 1].index)
sns.countplot(x = 'no_show', data = data)
data.hist(figsize=(15, 8))
gender_noshow = data.groupby('gender').sum()['no_show']

gender_noshow.plot.pie(figsize=(5,5),title = 'No show por gênero')
#Idade por no show

age_noshow = data.groupby('age').sum()['no_show']
age_noshow.plot()
age_percentage = age_noshow/data['age'].value_counts()
age_percentage.plot()
age_percentage = age_percentage[0:90]

age_percentage.plot()
data.groupby('no_show')['age'].mean()
fig = sns.FacetGrid(data, hue ='no_show', aspect = 4)

fig.map(sns.kdeplot, 'age', shade = True)

fig.add_legend()
def show_no_show_trend(dataset, attribute, fit_reg = True):

    '''Prints a chart with no_show_rate explanation

    Syntax: show_no_show_trend(dataframe, attribute), where:

        attribute = the string representing the attribute;

        dataframe = the current dataframe;

    '''

    return sns.lmplot(data = dataset, x = attribute, y = 'no_show_rate', fit_reg = fit_reg, legend = True, height=8, aspect=2)    



def show_attribute_statistics(attribute, dataframe, scale = 0.06, sorter = False, verticalLabel = False):

    '''Prints basic statistics from the attribute also plotting the basic chart. 

    Syntax: show_attribute_statistics(dataframe, attribute), where:

        attribute = the string representing the attribute;

        dataframe = the current dataframe;

        scale = what's the scale you want to converto;

        sorter = array representing the sort reindex;

    '''

    

    # grouping by the patients by attribute and see if there is any interesting data related to their no showing

    # also stripping unwanted attributes with crosstab - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html

    dataset = pd.crosstab(index = dataframe[attribute], columns = dataframe.no_show_str).reindex(sorter).reset_index() if sorter else pd.crosstab(index = dataframe[attribute], columns = dataframe.no_show_str).reset_index()

    

    # replacing all none values with zero, since it's the count of patients on that categorie

    dataset['No'].fillna(value=0, inplace=True)

    dataset['Yes'].fillna(value=0, inplace=True)



    # let's also record the rate of no-showing base on the attribute

    dataset["no_show_rate"] = dataset['Yes'] / (dataset['No'] + dataset['Yes'])

    dataset.no_show_rate.fillna(value=0.0, inplace=True)



    dataset["no_show_rate_value"] = dataset["no_show_rate"] * 100 

    dataset.no_show_rate_value.fillna(value=0.0, inplace=True)

    

    # plotting our data

    plt.figure(figsize=(30, 10))



    # scale data if needed

    dataset['No'] = dataset['No'] * scale

    dataset['Yes'] = dataset['Yes'] * scale



    # line chart

    plt.plot(dataset.no_show_rate_value.values, color="r")



    # bar chart

    plt.bar(dataset[attribute].unique(), dataset['No'].values, bottom = dataset['Yes'].values)

    plt.bar(dataset[attribute].unique(), dataset['Yes'].values)



    # configs

    if (verticalLabel):

        plt.xticks(rotation='vertical')

        

    plt.subplots_adjust(bottom=0.15)

    plt.xlabel(attribute, fontsize=16)

    plt.ylabel(f"amount of patients (scaled 1 to {scale * 100}%)", fontsize=16)

    plt.legend(["not attended rate", "attended", "not attended"], fontsize=14)



    plt.title("amount of patient by no show appointment groupped by %s" % attribute)



    plt.show();

    

    return dataset
age_dataset = show_attribute_statistics("age", data);

show_no_show_trend(age_dataset, "age");
wait_noshow = data.groupby('waitdays').sum()['no_show']

wait_noshow.plot()
wait_percentage = wait_noshow/data['waitdays'].value_counts()

wait_percentage.plot()
wait_percentage = wait_percentage[0:40]

wait_percentage.plot()
appointment_waiting_days_dataset = show_attribute_statistics("waitdays", data)

show_no_show_trend(appointment_waiting_days_dataset, "waitdays")
#Melhor criar categorias para facilitar a visualização

categories = pd.Series(['same day: 0', 'week: 1-7', 'month: 8-30', 'quarter: 31-90', 'semester: 91-180', 'a lot of time: >180'])

data['waitdays_cat'] = pd.cut(data.waitdays, bins = [-1, 0, 7, 30, 90, 180, 500], labels=categories)

waiting_days_categories_dataset = show_attribute_statistics("waitdays_cat", data, 0.005)

show_no_show_trend(waiting_days_categories_dataset, "waitdays_cat", False)
#No show e diabetes

sns.countplot(x='no_show', hue='diabetes', data=data)
fig, ax =plt.subplots(2,2, figsize=(15,10))

sns.countplot(x='no_show', hue='scholarship', data=data, ax=ax[0][0]).set_title('No-show vs Scholarship')

sns.countplot(x='no_show', hue='handicap', data=data, ax=ax[0][1]).set_title('No-show vs Handicap')

sns.countplot(x='no_show', hue='hypertension', data=data, ax=ax[1][0]).set_title('No-show vs Hipertension')

sns.countplot(x='no_show', hue='alcoholism', data=data, ax=ax[1][1]).set_title('No-show vs Alcoholism')
received_sms_dataset = show_attribute_statistics("sms_received", data, 0.005)

show_no_show_trend(received_sms_dataset, "sms_received")
#Qual a frequência de consultas por dia da semana

sns.countplot(x='appointment_weekday', data=data)

plt.xticks(rotation=15)
appointment_week_day_dataset = show_attribute_statistics("appointment_weekday", data, 0.005, ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

show_no_show_trend(appointment_week_day_dataset, "appointment_weekday", False)