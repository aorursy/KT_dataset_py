import numpy as np

import pandas as pd 

from control import *

from clean import *

from format_data import *
data = pd.read_csv("/kaggle/input/characteristics-corona-patients/Characteristics_Corona_patients version5 6-7-20.csv",

                   parse_dates=['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"])
data.shape
[i for i in data.columns]
[Control.examining_values_by_col ([data], [""],j ) for j in data.columns ]
f = Format.category_col_to_num_col(data, "severity_illness")



data["severity_illness_infectious_person"] = data["severity_illness_infectious_person"].replace(f)
for j in  [ "sex", "treatment", "country" ]:

    l = Format.category_col_to_num_col(data, j)

    print(l)
#  Format

def turn_hotvec(df, input_col):

    name_col = get_name_of_categorized_value(df, input_col)

    for j in name_col:

        name = j.replace(" ", "_")

        df[input_col + "_"+ name] = df[input_col][df[input_col].notnull()].apply(lambda i: 1 if j in i else 0)

        print(df[input_col + "_"+ name].value_counts())
%%time

Format.turn_hotvec(data, 'symptoms')

Format.turn_hotvec(data, 'background_diseases')
data = data.drop(['city', 'infection_place','region', 'symptoms', 'region', 

           'symptoms_no_symptom', 'background_diseases', 'date_onset_symptoms',  'deceased_date',

                  'released_date',  'return_date',  "infected_by",'background_diseases_', "confirmed_date",], axis=1)
data = data.drop([] , axis=1)
[Control.examining_values_by_col ([data], [""],j ) for j in data.columns ]

data.shape
[i for i in data.columns]
data.to_csv("Characteristics_Corona_patients_version_6 - 6-7-2020.csv", index = False)

data.to_csv()