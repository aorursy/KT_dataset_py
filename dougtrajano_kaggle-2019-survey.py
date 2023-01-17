# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Graphics

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from plotly.offline import init_notebook_mode, iplot

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as offline

py.init_notebook_mode(connected=True)

init_notebook_mode(connected=True)

offline.init_notebook_mode()

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file = "/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv"

df_multiple_choice = pd.read_csv(file)

print(df_multiple_choice.shape)

df_multiple_choice.head()
file = "/kaggle/input/kaggle-survey-2019/questions_only.csv"

df_questions = pd.read_csv(file)

print(df_questions.shape)

df_questions.head()
file = "/kaggle/input/kaggle-survey-2019/survey_schema.csv"

df_survey_schema = pd.read_csv(file)

print(df_survey_schema.shape)

df_survey_schema.head()
file = "/kaggle/input/kaggle-survey-2019/other_text_responses.csv"

df_other_text = pd.read_csv(file)

print(df_other_text.shape)

df_other_text.head()
def get_country_codes(country_list, countries_errors=None):

    import pycountry

    code_list = []

    for country in country_list:

        try:

            code = pycountry.countries.get(name=country).alpha_3

        except:

            try:

                code = pycountry.countries.search_fuzzy(country)

                if len(code) == 1:

                    code = code[0].alpha_3

                else:

                    if countries_errors != None:

                        if country in countries_errors.keys():

                            code = countries_errors[country]

                        else:

                            print("{} isn't in countries_errors".format(country))

                            code = np.nan

                    else:

                        print("Error to get code for {}".format(country))

                        code = np.nan

            except:

                if countries_errors != None:

                    if country in countries_errors.keys():

                        code = countries_errors[country]

                    else:

                        print("{} isn't in countries_errors".format(country))

                        code = np.nan

                else:

                    print("Error to get code for {}".format(country))

                    code = np.nan

        finally:

            code_list.append(code)

    return code_list



countries_errors = {

    "South Korea": "KOR",

    "Iran, Islamic Republic of...": "IRN",

    "Hong Kong (S.A.R.)": "HKG",

    "Taiwan": "TWN"

}



df_temp = pd.DataFrame(df_multiple_choice["Q3"][1:])

df_temp.rename(columns={"Q3": "country"}, inplace=True)

df_temp["respondents"] = 1

df_temp = df_temp.groupby("country").sum()

df_temp.reset_index(inplace=True)

df_temp.sort_values(by="respondents", inplace=True, ascending=False)

df_temp["country_code"] = get_country_codes(df_temp["country"], countries_errors)



print(df_temp.shape)

df_temp.head()
import plotly.graph_objects as go



fig = go.Figure(data=go.Choropleth(

    locations = df_temp['country_code'],

    z = df_temp['respondents'],

    text = df_temp['country'],

    colorscale = 'Blues',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Qty respondents'

))



fig.update_layout(

    title_text="Kaggle's respondents in the world.",

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
df_temp = df_multiple_choice[1:]["Q2"].value_counts()



fig = {

  "data": [

    {

      "values": df_temp.values,

      "labels": df_temp.index,

      "domain": {"x": [0, .48]},

      "hole": .7,

      "type": "pie"

    },

    

    ],

  "layout": {

        "title":"Gender distribution",

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)
df_temp = df_multiple_choice[1:]["Q1"].value_counts()

df_temp = pd.DataFrame(df_temp)

df_temp.reset_index(inplace=True)

df_temp.sort_values(by="index", inplace=True)

df_temp.set_index("index", inplace=True)

df_temp.index.names = [""]

df_temp = df_temp["Q1"]



fig = {

  "data": [

    {

      "y": df_temp.values,

      "x": df_temp.index,

      "type": "bar"

    },

    

    ],

  "layout": {

        "title":"Age distribution",

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)