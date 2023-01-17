import pandas  as pd

import seaborn as sns

import plotly.express    as px

import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')
# Read and inspect data

raw_data = pd.read_csv('../input/violence-against-women-and-girls/makeovermonday-2020w10/violence_data.csv')

raw_data.head()
print('Dataset contains data from {} countries'.format(raw_data.Country.nunique()))
raw_survey_df = raw_data.pivot_table(index=['Country','Gender','Demographics Question','Demographics Response'],columns=['Question'], values=['Value'])

raw_survey_df
# Reset columns

survey_df = raw_survey_df.T.reset_index(drop=True).T.reset_index()



# Rename columns

survey_df.columns = ['country',

                     'gender',

                     'demographics_question',

                     'demographics_response',

                     'violence_any_reason',

                     'violence_argue',

                     'violence_food',

                     'violence_goingout',

                     'violence_neglect',

                     'violence_sex',

                    ]
survey_df
# Examine Violence x gender

fig = px.box(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),

            x      = 'country',

            y      = 'violence_any_reason',

            color  = 'gender',

            title  = '% of Respondents that agree with violence for any surveyed reason across Country and Gender',

            color_discrete_sequence = ['#4a00ba','#00ba82'],

            height = 650

        )



fig.update_xaxes(title='Country')

fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')

fig.show()
# Examine Violence x Age group

fig = px.bar(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),

            x      = 'country',

            y      = 'violence_any_reason',

            color = 'demographics_response',

            title  = '% of Violence for any surveyed reason across Country and Age Group ',

            height = 650

        )



fig.update_xaxes(title='Country')

fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')

fig.show()
# Examine Correlations

plt.figure(figsize=(10,10))

sns.heatmap(survey_df.iloc[:,4:].corr(),

            square=True,

            linewidths=.5,

            cmap=sns.diverging_palette(10, 220, sep=80, n=7),

            annot=True,

           )

plt.title('Correlation Across Different Violence Questions')

plt.show()