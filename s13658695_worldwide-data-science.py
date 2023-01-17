import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import os

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

SurveySchema_2019 = pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv")

questions_only_2019= pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv")

multipleChoiceResponses_2019 = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses_2019 = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")
multipleChoice_2019_df = multipleChoiceResponses_2019.copy()

multipleChoice_2019_df = multipleChoice_2019_df.drop([0])
multipleChoice_2019_df
multipleChoice_2019_df['Q4'].replace({

    'Some college/university study without earning a bachelor’s degree': 'College with no degree',

    'I prefer not to answer':'no answer',

    'No formal education past high school':'high school'}, inplace=True)
degree_type = list(multipleChoice_2019_df['Q4'].value_counts().keys())

degree_count = list(multipleChoice_2019_df['Q4'].value_counts().values)
from random import randint

color_list = []

for i in range(10):

    color_list.append('#%06X' % randint(0, 0xFFFFFF))
df=pd.DataFrame({'degree_type': degree_type,

                 'Count': degree_count})



plt.figure(figsize=(30,30))

plt.bar('degree_type', 'Count', width=0.5, color = color_list,bottom=None, 

        align='center', data= df)

plt.title("Education Background", fontsize =30, pad=30)

plt.xlabel('Education',fontsize = 25, labelpad=30)

plt.ylabel('Count', fontsize = 25, labelpad=30)

plt.xticks(fontsize=18,rotation=45)

plt.yticks(fontsize=18)

plt.grid(color='g', linestyle='-', linewidth=0.5)

plt.show()
import plotly.express as px

fig = px.bar(df, x="degree_type", y="Count",color='Count',title='good',)

fig.update_layout(

    title={

        'text': "Plot Title",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
'''



SurveySchema_2018 = pd.read_csv("../input/kaggle-survey-2018/SurveySchema.csv")

freeFormResponses_2018 = pd.read_csv("../input/kaggle-survey-2018/freeFormResponses.csv")

multipleChoiceResponses_2018 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")



conversionRates_2017 = pd.read_csv("../input/kaggle-survey-2017/conversionRates.csv")

freeformResponses_2017 = pd.read_csv("../input/kaggle-survey-2017/freeformResponses.csv")

multipleChoiceResponses_2017 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv")

schema_2017 = pd.read_csv("../input/kaggle-survey-2017/schema.csv")

'''