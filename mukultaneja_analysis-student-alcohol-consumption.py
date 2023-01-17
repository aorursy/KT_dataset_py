# loading require libraries

import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

%matplotlib inline



data = pd.read_csv('../input/student-mat.csv')



# dividing data into two sets based on address such as rural / urban

rural_data = data[data['address'] == 'R']

urban_data = data[data['address'] == 'U']



# Length of rural - 88 rows

# Length of urban - 307
# Daily / Weekly consumption in rural areas



fig, ax = plt.subplots(1, 2)

rural_data.groupby(['Dalc']).size().plot(kind='bar', ax=ax[0], title='Dalc measure in rural area')

rural_data.groupby(['Walc']).size().plot(kind='bar', ax=ax[1], title='Walc measure in rural area')
# Daily / Weekly consumption in urban areas



fig, ax = plt.subplots(1, 2)

urban_data.groupby(['Dalc']).size().plot(kind='bar', ax=ax[0], title='Dalc measure in urban area')

urban_data.groupby(['Walc']).size().plot(kind='bar', ax=ax[1], title='Walc measure in urban area')
# Daily alcohol consumption across sex in rural areas



fig, ax = plt.subplots(1, 2)



rural_df = rural_data[['sex', 'Dalc']]

rural_df = rural_df[rural_df['sex'] == 'M']

rural_df.groupby(['sex', 'Dalc']).size().plot(kind='bar', ax=ax[0], title="consumption across sex")



rural_df = rural_data[['sex', 'Dalc']]

rural_df = rural_df[rural_df['sex'] == 'F']

rural_df.groupby(['sex', 'Dalc']).size().plot(kind='bar', ax=ax[1], title="consumption across sex")
# Daily alcohol consumption across sex in urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['sex', 'Dalc']]

urban_df = urban_df[urban_df['sex'] == 'M']

urban_df.groupby(['sex', 'Dalc']).size().plot(kind='bar', ax=ax[0], title="consumption across sex")



urban_df = urban_data[['sex', 'Dalc']]

urban_df = urban_df[urban_df['sex'] == 'F']

urban_df.groupby(['sex', 'Dalc']).size().plot(kind='bar', ax=ax[1], title="consumption across sex")
# age wise distribution of Daily consumption in rural areas



rural_df = rural_data[['age', 'Dalc']]

rural_df['total'] = 1

rural_df = rural_df.groupby(['age', 'Dalc']).sum().reset_index()

pd.pivot_table(rural_df, index=['age'], columns=['Dalc'], values=['total']).fillna(0).plot(kind='bar', stacked=True,

                                                                                          title="age wise distribution")
# age wise distribution of Daily consumption in urban areas



urban_df = urban_data[['age', 'Dalc']]

urban_df['total'] = 1

urban_df = urban_df.groupby(['age', 'Dalc']).sum().reset_index()

pd.pivot_table(urban_df, index=['age'], columns=['Dalc'], values=['total']).fillna(0).plot(kind='bar', stacked=True,

                                                                                          title="age wise distribution")
# total grades in rural and urban areas



fig, ax = plt.subplots(1, 2)



rural_df = rural_data[['G1', 'G2', 'G3']]

rural_df['total'] = rural_df.apply(sum, axis=1)

rural_df['total'].plot(ax=ax[0], title="students grades - rural areas")



urban_df = urban_data[['G1', 'G2', 'G3']]

urban_df['total'] = urban_df.apply(sum, axis=1)

urban_df['total'].plot(ax=ax[1], title="students grades - urban areas")
# past failures and passing ratios in rural and urban areas



fig, ax = plt.subplots(1, 2)



rural_failure_ratio = (1.0 * rural_data['failures'].sum()) / len(rural_data) * 100



urban_failure_ratio = (1.0 * urban_data['failures'].sum()) / len(urban_data) * 100



failure_df = pd.DataFrame({'rural' : [rural_failure_ratio,],

                           'urban' : [urban_failure_ratio,]})



rural_passing_ratio = (1 - (1.0 * rural_data['failures'].sum()) / len(rural_data)) * 100



urban_passing_ratio = (1 - (1.0 * urban_data['failures'].sum()) / len(urban_data)) * 100



passing_df = pd.DataFrame({'rural' : [rural_passing_ratio,],

                           'urban' : [urban_passing_ratio,]})





failure_df.plot(kind='bar', legend=True, ax=ax[0], title="failure ratio rural vs urban")



passing_df.plot(kind='bar', ax=ax[1], legend=True, title="passing ratio rural vs urban")
# impact of Dalc over failures in rural areas



rural_df = rural_data[['failures', 'Dalc']]

rural_df['total'] = 1

rural_df = rural_df.groupby(['Dalc', 'failures']).sum().reset_index()

pd.pivot_table(rural_df, index=['Dalc'], columns=['failures'], values=['total']).fillna(0).plot(kind='bar',

                                                                                                stacked=True,

                                                                                                title="Dalc imapact over failures")
# impact of Dalc over failures in urban areas



urban_df = urban_data[['failures', 'Dalc']]

urban_df['total'] = 1

urban_df = urban_df.groupby(['Dalc', 'failures']).sum().reset_index()

pd.pivot_table(urban_df, index=['Dalc'], columns=['failures'], values=['total']).fillna(0).plot(kind='bar',

                                                                                                stacked=True,

                                                                                                title="Dalc imapact over failures")
# trend across Dalc and failures in rural areas



rural_df = rural_data[['failures', 'age', 'Dalc']]

rural_df.groupby(['age']).sum().plot(title="trend across Dalc and failures")
# trend across Dalc and failures in urban areas



urban_df = urban_data[['failures', 'age', 'Dalc']]

urban_df.groupby(['age']).sum().plot(title="trend across Dalc and failures")
# trend across Dalc and studytime in rural areas



rural_df = rural_data[['studytime', 'Dalc']]

rural_df.groupby(['Dalc']).sum().plot(title="trend across Dalc and studytime")
# trend across Dalc and studytime in urban areas



urban_df = urban_data[['studytime', 'Dalc']]

urban_df.groupby(['Dalc']).sum().plot(title="trend across Dalc and studytime")
# trend across studytime and failures in rural areas



rural_df = rural_data[['studytime', 'failures']]

rural_df.groupby(['studytime']).sum().plot(title="trend across studytime and failures")
# trend across studytime and failures in urban areas



urban_df = urban_data[['studytime', 'failures']]

urban_df.groupby(['studytime']).sum().plot(title="trend across studytime and failures")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## rural areas



fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'failures', 'age']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs age vs failures")



rural_df_T = rural_df[rural_df['Pstatus'] == 'T']

rural_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs age vs failures")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## rural areas



fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'studytime', 'age']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs age vs studytime")



rural_df_T = rural_df[rural_df['Pstatus'] == 'T']

rural_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs age vs studytime")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## rural areas



fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'G1', 'G2', 'G3', 'age']]



rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs age vs grades")



rural_df_T = rural_df[rural_df['Pstatus'] == 'T']

rural_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs age vs grades")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)

urban_df = urban_data[['Pstatus', 'failures', 'age']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs failures vs age")



urban_df_T = urban_df[urban_df['Pstatus'] == 'T']

urban_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs failures vs age")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Pstatus', 'studytime', 'age']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs studytime vs age")



urban_df_T = urban_df[urban_df['Pstatus'] == 'T']

urban_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs studytime vs age")
## lets focus over support for education now from parents / school

## using Pstatus field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)

urban_df = urban_data[['Pstatus', 'G1', 'G2', 'G3', 'age']]



urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Pstatus', 'age']).sum().plot(ax=ax[0], title="Pstatus vs grades vs age")



urban_df_T = urban_df[urban_df['Pstatus'] == 'T']

urban_df_T.groupby(['Pstatus', 'age']).sum().plot(ax=ax[1], title="Pstatus vs grades vs age")
## lets focus over support for education now from parents / school

## using Medu / Fedu field here to show impact of this over performance

## rural areas



fig, ax = plt.subplots(1, 2)



rural_df = rural_data[['Medu', 'G1', 'G2', 'G3']]

rural_df.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Medu vs grades")



rural_df = rural_data[['Fedu', 'G1', 'G2', 'G3']]

rural_df.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Medu vs grades")
## lets focus over support for education now from parents / school

## using Pstatus / Medu / Fedu field here to show impact of this over performance

## rural areas





fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'Medu', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Pstatus vs Medu vs grades")



rural_df = rural_data[['Pstatus', 'Fedu', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Pstatus vs Fedu vs grades")
## lets focus over support for education now from parents / school

## using Pstatus / Medu / Fedu field here to show impact of this over performance

## rural areas





fig, ax = plt.subplots(1, 2)





rural_df = rural_data[['Pstatus', 'Medu', 'G1', 'G2', 'G3']]

rural_df_T = rural_df[rural_df['Pstatus'] == 'T']

rural_df_T.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Pstaus vs Medu vs grades")



rural_df = rural_data[['Pstatus', 'Fedu', 'G1', 'G2', 'G3']]

rural_df_T = rural_df[rural_df['Pstatus'] == 'T']

rural_df_T.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Pstaus vs Fedu vs grades")
## lets focus over support for education now from parents / school

## using Medu / Fedu field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Medu', 'G1', 'G2', 'G3']]

urban_df.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Medu vs grades")



urban_df = urban_data[['Fedu', 'G1', 'G2', 'G3']]

urban_df.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Fedu vs grades")
## lets focus over support for education now from parents / school

## using Pstatus / Medu / Fedu field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Pstatus', 'Medu', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Pstatus vs Medu vs grades")



urban_df = urban_data[['Pstatus', 'Fedu', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Pstatus vs Fedu vs grades")
## lets focus over support for education now from parents / school

## using Pstatus / Medu / Fedu field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)





urban_df = urban_data[['Pstatus', 'Medu', 'G1', 'G2', 'G3']]

urban_df_T = urban_df[urban_df['Pstatus'] == 'T']

urban_df_T.groupby(['Medu']).sum().plot(ax=ax[0], legend=True, title="Pstaus vs Medu vs grades")



urban_df = urban_data[['Pstatus', 'Fedu', 'G1', 'G2', 'G3']]

urban_df_T = urban_df[urban_df['Pstatus'] == 'T']

urban_df_T.groupby(['Fedu']).sum().plot(ax=ax[1], legend=True, title="Pstatus vs Fedu vs grades")
## lets focus over support for education now from parents / school

## using Mjob / Fjob field here to show impact of this over performance

## rural areas





fig, ax = plt.subplots(1, 2)



rural_df = rural_data[['Mjob', 'G1', 'G2', 'G3']]

rural_df.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Mjob vs grades")



rural_df = rural_data[['Fjob', 'G1', 'G2', 'G3']]

rural_df.groupby(['Fjob']).sum().plot(ax=ax[1], legend=True, title="Fjob vs grades")
## lets focus over support for education now from parents / school

## using Mjob / Fjob field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'Mjob', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Mjob vs grades")



rural_df = rural_data[['Pstatus', 'Fjob', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'A']

rural_df_A.groupby(['Fjob']).sum().plot(ax=ax[1], legend=True, title="Fjob vs grades")
## lets focus over support for education now from parents / school

## using Pstatus Mjob / Fjob field here to show impact of this over performance

## rural areas



fig, ax = plt.subplots(1, 2)

rural_df = rural_data[['Pstatus', 'Mjob', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'T']

rural_df_A.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Pstatus vs Mjob vs grades")



rural_df = rural_data[['Pstatus', 'Fjob', 'G1', 'G2', 'G3']]

rural_df_A = rural_df[rural_df['Pstatus'] == 'T']

rural_df_A.groupby(['Fjob']).sum().plot(ax=ax[1], legend=False, title="Pstatus vs Fjob vs grades")
## lets focus over support for education now from parents / school

## using Mjob / Fjob field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Mjob', 'G1', 'G2', 'G3']]

urban_df.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Mjob vs grades")



urban_df = urban_data[['Fjob', 'G1', 'G2', 'G3']]

urban_df.groupby(['Fjob']).sum().plot(ax=ax[1], legend=True, title="Fjob vs grades")
## lets focus over support for education now from parents / school

## using Pstatus Mjob / Fjob field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Pstatus', 'Mjob', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Pstatus vs Mjob vs grades")



urban_df = urban_data[['Pstatus', 'Fjob', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'A']

urban_df_A.groupby(['Fjob']).sum().plot(ax=ax[1], legend=True, title="Pstatus vs Fjob vs grades")
## lets focus over support for education now from parents / school

## using Pstatus Mjob / Fjob field here to show impact of this over performance

## urban areas



fig, ax = plt.subplots(1, 2)



urban_df = urban_data[['Pstatus', 'Mjob', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'T']

urban_df_A.groupby(['Mjob']).sum().plot(ax=ax[0], legend=True, title="Pstatus vs Mjob vs grades")



urban_df = urban_data[['Pstatus', 'Fjob', 'G1', 'G2', 'G3']]

urban_df_A = urban_df[urban_df['Pstatus'] == 'T']

urban_df_A.groupby(['Fjob']).sum().plot(ax=ax[1], legend=True, title="Pstatus vs Fjob vs grades")
## lets focus over support for education now from parents / school

## using guardian field here to show impact of this over performance

## rural areas / urban areas



fig, ax  = plt.subplots(1, 2)



rural_df = rural_data[['guardian', 'G1', 'G2', 'G3']]

rural_df.groupby(['guardian']).sum().plot(ax=ax[0], title="guardian vs grades", legend=True)



urban_df = urban_data[['guardian', 'G1', 'G2', 'G3']]

urban_df.groupby(['guardian']).sum().plot(ax=ax[1], title="guardian vs grades", legend=True)
## lets focus over support for education now from parents / school

## using school support field here to show impact of this over performance

## rural areas / urban areas



fig, ax  = plt.subplots(1, 2)



rural_df = rural_data[['schoolsup', 'G1', 'G2', 'G3']]

rural_df.groupby(['schoolsup']).sum().plot(ax=ax[0], title="school support vs grades")



urban_df = urban_data[['schoolsup', 'G1', 'G2', 'G3']]

urban_df.groupby(['schoolsup']).sum().plot(ax=ax[1], title="school support vs grades")
## lets focus over support for education now from parents / school

## using family support field here to show impact of this over performance

## rural areas / urban areas



fig, ax  = plt.subplots(1, 2)



rural_df = rural_data[['famsup', 'G1', 'G2', 'G3']]

rural_df.groupby(['famsup']).sum().plot(ax=ax[0], title="Family support vs grades")



urban_df = urban_data[['famsup', 'G1', 'G2', 'G3']]

urban_df.groupby(['famsup']).sum().plot(ax=ax[1], title="Family support vs grades")
## lets focus over support for education now from parents / school

## using higher field here to show impact of this over performance

## rural areas / urban areas



fig, ax  = plt.subplots(1, 2)



rural_df = rural_data[['higher']]

rural_df['count'] = 1

rural_df.groupby(['higher']).sum().plot(ax=ax[0], title="higher study vs count")



urban_df = urban_data[['higher']]

urban_df['count'] = 1

urban_df.groupby(['higher']).sum().plot(ax=ax[1], title="higher study vs count")