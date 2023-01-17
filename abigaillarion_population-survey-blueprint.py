import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



survey_data = pd.read_csv('../input/cps_2016-08.csv',

                          usecols=[0, 3, 11, 16, 17, 18, 32, 43, 45, 47, 48, 51, 52, 65,

                                   69, 91, 189, 210, 220, 222, 226, 229, 242, 243, 244])

survey_data = survey_data.rename(

    columns={'HRHHID':'household_id', 'HURESPLI':'person_id', 'HRINTSTA':'interview',

             'GESTFIPS':'state', 'HRHTYPE':'household_type', 'HRNUMHOU':'household_count', 

             'HEFAMINC':'household_income', 'PESEX':'sex', 'PRTAGE':'age',

             'PTDTRACE':'race', 'PRCITSHP':'citizenship', 'PEEDUCA':'education',

             'PEMARITL':'marriage', 'PEMLR':'work_status', 'PEIO1COW':'work_type',

             'PRMJIND1':'work_industry', 'PEHRUSLT':'work_hours', 'PEERNPER':'pay_period',

             'PEERNHRY':'pay_type', 'PRERNHLY':'hourly_wage', 'PESCHENR':'school_status',

             'PESCHFT':'school_hours', 'PESCHLVL':'school_type'})

survey_data = survey_data[survey_data.interview == 1]



np.set_printoptions(suppress=True)
# households in Census Bureau population survey (51,075 rows)

survey_household = survey_data[['household_id', 'state', 'household_type',

                                'household_count', 'household_income']]

survey_household = survey_household.drop_duplicates('household_id')



# distribution of household type in population survey

household_type = np.asarray(

    survey_household.groupby('household_type').household_type.count())

household_type = np.round(np.divide(household_type, sum(household_type)) * 100, 2)



# distribution of household count in population survey

household_count = np.asarray(

    survey_household.groupby('household_count').household_count.count())

household_count[9] = sum(household_count[9:])

household_count = np.round(np.divide(household_count[:10],

                                     sum(household_count)) * 100, 2)



# distribution of household income in population survey

household_income = np.asarray(

    survey_household.groupby('household_income').household_income.count())

household_income[0] = sum(household_income[0:3])

household_income[3] = sum(household_income[3:6])

household_income[6] = sum(household_income[6:8])

household_income[8] = sum(household_income[8:10])

household_income = np.delete(household_income, [1,2,4,5,7,9])

household_income = np.round(np.divide(household_income,

                                      sum(household_income)) * 100, 2)



axis_one = np.arange(1, 11)



trace_type = go.Bar(x=axis_one, y=household_type, name='Type')

trace_count = go.Bar(x=axis_one, y=household_count, name='Count')

trace_income = go.Bar(x=axis_one, y=household_income, name='Income')



figure = tools.make_subplots(rows=1, cols=3, shared_yaxes=True, print_grid=False,

                             subplot_titles=('Household Type', 'Household Count',

                                             'Household Income'))

figure['layout'].update(xaxis1=dict(tickmode='linear'), xaxis2=dict(tickmode='linear'),

                        xaxis3=dict(tickmode='linear'), showlegend=False, height=400,

                        yaxis1=dict(showgrid=False, ticksuffix='%'))

figure.append_trace(trace_type, 1, 1)

figure.append_trace(trace_count, 1, 2)

figure.append_trace(trace_income, 1, 3)



iplot(figure)
# individuals in Census Bureau population survey (131,759 rows)

survey_individual = survey_data[['household_id', 'person_id', 'state', 'sex', 'age',

                                 'race', 'citizenship', 'education', 'marriage',

                                 'work_status', 'work_type', 'work_industry',

                                 'work_hours', 'pay_period', 'pay_type', 'hourly_wage',

                                 'school_status', 'school_hours', 'school_type']]



# distribution of sex in population survey

sex = np.asarray(survey_individual.groupby('sex').sex.count())

sex = np.round(np.divide(sex, sum(sex)) * 100, 2)

sex = np.append(sex, [0] * 8)



# distribution of age in population survey

age_count = np.asarray(survey_individual.groupby('age').age.count())

age_count[80] = sum(age_count[80:])

age = np.add.reduceat(age_count, range(0, 79, 9))

age = np.append(age, age_count[80])

age = np.round(np.divide(age, sum(age)) * 100, 2)



# distribution of race in population survey

race = np.asarray(survey_individual.groupby('race').race.count())

race[8] = sum(race[8:15])

race[15] = sum(race[15:])

race = race[[0,1,2,3,4,5,6,7,8,15]]

race = np.round(np.divide(race, sum(race)) * 100, 2)



trace_sex = go.Bar(x=axis_one, y=sex, name='Sex')

trace_age = go.Bar(x=axis_one, y=age, name='Age')

trace_race = go.Bar(x=axis_one, y=race, name='Race')



figure = tools.make_subplots(rows=1, cols=3, shared_yaxes=True, print_grid=False,

                             subplot_titles=('Individual Sex', 'Individual Age',

                                             'Individual Race'))

figure['layout'].update(xaxis1=dict(tickmode='linear'), xaxis2=dict(tickmode='linear'),

                        xaxis3=dict(tickmode='linear'), showlegend=False, height=400,

                        yaxis1=dict(showgrid=False, ticksuffix='%'))

figure.append_trace(trace_sex, 1, 1)

figure.append_trace(trace_age, 1, 2)

figure.append_trace(trace_race, 1, 3)



iplot(figure)
# distribution of citizenship in population survey

citizenship = np.asarray(survey_individual.groupby('citizenship').citizenship.count())

citizenship = np.round(np.divide(citizenship, sum(citizenship)) * 100, 2)

citizenship = np.append(citizenship, [0] * 5)



# distribution of education in population survey

education = np.asarray(survey_individual[survey_individual.education > 0]

                       .groupby('education').education.count())

education[0] = sum(education[0:2])

education[2] = sum(education[2:4])

education[4] = sum(education[4:8])

education[10] = sum(education[10:12])

education = np.delete(education, [1,3,5,6,7,11])

education = np.round(np.divide(education, sum(education)) * 100, 2)



# distribution of marriage in population survey

marriage = np.asarray(survey_individual[survey_individual.marriage > 0]

                            .groupby('marriage').marriage.count())

marriage[0] = sum(marriage[0:2])

marriage = np.delete(marriage, 1)

marriage = np.round(np.divide(marriage, sum(marriage)) * 100, 2)

marriage = np.append(marriage, [0] * 5)



trace_citizenship = go.Bar(x=axis_one, y=citizenship, name='Citizenship')

trace_education = go.Bar(x=axis_one, y=education, name='Education')

trace_marriage = go.Bar(x=axis_one, y=marriage, name='Marriage')



figure = tools.make_subplots(rows=1, cols=3, shared_yaxes=True, print_grid=False,

                             subplot_titles=('Individual Citizenship',

                                             'Individual Education',

                                             'Individual Marriage'))

figure['layout'].update(xaxis1=dict(tickmode='linear'), xaxis2=dict(tickmode='linear'),

                        xaxis3=dict(tickmode='linear'), showlegend=False, height=400,

                        yaxis1=dict(showgrid=False, ticksuffix='%'))

figure.append_trace(trace_citizenship, 1, 1)

figure.append_trace(trace_education, 1, 2)

figure.append_trace(trace_marriage, 1, 3)



iplot(figure)
# distribution of work status in population survey

work_status = np.asarray(survey_individual[survey_individual.work_status > 0]

                         .groupby('work_status').work_status.count())

work_status = np.round(np.divide(work_status, sum(work_status)) * 100, 2)

work_status = np.append(work_status, [0] * 1)



# distribution of work type in population survey

work_type = np.asarray(survey_individual[survey_individual.work_type > 0]

                       .groupby('work_type').work_type.count())

work_type = np.round(np.divide(work_type, sum(work_type)) * 100, 2)



# distribution of work industry in population survey

work_industry = np.asarray(survey_individual[survey_individual.work_industry > 0]

                       .groupby('work_industry').work_industry.count())

work_industry = np.round(np.divide(work_industry, sum(work_industry)) * 100, 2)



axis_two = np.arange(1, 9)

axis_three = np.arange(1, 15)



trace_status = go.Bar(x=axis_two, y=work_status, name='Status')

trace_type = go.Bar(x=axis_two, y=work_type, name='Type')

trace_industry = go.Bar(x=axis_three, y=work_industry, name='Industry')



figure = tools.make_subplots(rows=1, cols=3, shared_yaxes=True, print_grid=False,

                             subplot_titles=('Work Status', 'Work Type', 'Work Industry'))

figure['layout'].update(xaxis1=dict(domain=[0.0, 0.232], tickmode='linear'),

                        xaxis2=dict(domain=[0.3, 0.532], tickmode='linear'),

                        xaxis3=dict(domain=[0.6, 1.0], tickmode='linear'),

                        showlegend=False, height=400,

                        yaxis1=dict(showgrid=False, ticksuffix='%'))

figure.append_trace(trace_status, 1, 1)

figure.append_trace(trace_type, 1, 2)

figure.append_trace(trace_industry, 1, 3)

figure['layout']['annotations'][0].update(x=0.116)

figure['layout']['annotations'][1].update(x=0.416)

figure['layout']['annotations'][2].update(x=0.8)



iplot(figure)