# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





#plotly.offline doesn't push your charts to the clouds

import plotly.offline as pyo

import plotly.graph_objs as go

pyo.offline.init_notebook_mode()



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('/kaggle/input/medicalappointmentnoshown/KaggleV2-May-2016.csv')



#browse sample of data values and formats of each feature. 

df.head()
#browse data frame columns data types

df.info()



#print out statistical details of the numeric data.

df.describe()
#Rename Column name

df.rename(columns = {'ApointmentData':'AppointmentData',

                         'Alcoolism': 'Alchoholism',

                         'HiperTension': 'Hypertension',

                         'Handcap': 'Handicap'}, inplace = True)



print(df.columns)
#check number of not showing up patinets to an appointment on scale of 100

#group by no-show column

no_show_percentage = pd.DataFrame(df.groupby(["No-show"])["PatientId"].count())

print(no_show_percentage)

#calculate percentage of show up and no show and store it in column No-Show

no_show_percentage["No-show"] = no_show_percentage["PatientId"] / sum(no_show_percentage["PatientId"]) * 100

print(no_show_percentage)

no_show_percentage.drop(columns="PatientId", inplace=True)

print(no_show_percentage)

#plot the dataframe 

no_show_percentage.plot.bar(figsize=(10,5))

plt.ylim(top=100)

plt.title("Medical Appointments",{'fontsize': 20},pad=20)

plt.xlabel("Appointment Status")

plt.xticks(np.arange(2), ('Show-Up', 'No-Show'), rotation=0)

plt.legend(["Appointment Status Rate"])