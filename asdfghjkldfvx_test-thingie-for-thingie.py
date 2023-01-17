import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

%matplotlib inline



survey_results = pd.read_csv('../input/survey_results_public.csv')

survey_results_schema = pd.read_csv('../input/survey_results_schema.csv')
def splitDataFrameList(df, target_column, separator):

    def splitListToRows(row, row_accumulator, target_column, separator):

        val = row[target_column]

        if type(val) is float:

            val = str(val)

        split_row = val.split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df



def remove_nulls(data, field):

    return data[data[field].isnull() == False]





def get_question(column_name):

    for i, column in enumerate(survey_results_schema.Column):

        if column == column_name:

            return survey_results_schema.Question[i]

    return ''
survey_results.VersionControl.value_counts().plot.bar(width=0.9);
survey_results.TabsSpaces.value_counts().plot.bar(width=0.9);
survey_results.PronounceGIF.value_counts().plot.bar(width=0.9);
# survey_results.groupby('Country').PronounceGIF.value_counts()

survey_results[survey_results.Country == 'Estonia'].groupby('Country').PronounceGIF.value_counts()
plt.figure(figsize=(10, 5));

non_null_ide = remove_nulls(survey_results, 'IDE')



splitDataFrameList(non_null_ide, 'IDE', '; ').IDE.value_counts().plot.bar(width=0.9);
print(get_question('HaveWorkedLanguage'))



plt.figure(figsize=(10, 5));

non_null_ide = remove_nulls(survey_results, 'HaveWorkedLanguage')



splitDataFrameList(non_null_ide, 'HaveWorkedLanguage', '; ').HaveWorkedLanguage.value_counts().plot.bar(width=0.9);
print(get_question('Salary') + '\n\n' + get_question('HoursPerWeek'))



filtered_survey = remove_nulls(remove_nulls(survey_results, 'Salary'), 'HoursPerWeek')



filtered_survey.plot.scatter('Salary', 'HoursPerWeek', alpha=0.2);