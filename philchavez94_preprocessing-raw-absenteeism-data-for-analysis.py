import pandas as pd
raw_csv_file = pd.read_csv('../input/employee-absenteeism-prediction/Absenteeism-data.csv')
raw_csv_file
df = raw_csv_file.copy()
df = df.drop(['ID'], axis = 1)
reason_columns = pd.get_dummies(df['Reason for Absence'])
reason_columns
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
reason_columns
df.columns.values
df = df.drop(['Reason for Absence'], axis = 1)
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)

reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)

reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)

reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
reason_type_4
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
df
df.columns.values
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns.values
df.columns = column_names
df.head()
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours']
column_names_reordered
df = df[column_names_reordered]
df.head()
df_reason_mod = df.copy()
type(df_reason_mod['Date'][0])
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')
df_reason_mod['Date']
list_months = []
for i in range(df_reason_mod.shape[0]):

    list_months.append(df_reason_mod['Date'][i].month)
len(list_months)
df_reason_mod['Month Value'] = list_months
df_reason_mod.head()
df_reason_mod['Date'][699].weekday()
df_reason_mod['Date'][699]
def date_to_weekday(date_value):

    return date_value.weekday()
df_reason_mod['Day of Week'] = df_reason_mod['Date'].apply(date_to_weekday)
df_reason_mod.head()
df_reason_mod.drop('Date', axis = 1)
df_reason_mod.columns
columns_reorderd_2 = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of Week', 'Date',

       'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',

       'Pets', 'Absenteeism Time in Hours']
columns_reorderd_2
df_reason_mod = df_reason_mod[columns_reorderd_2]
df_reason_mod
df_reason_mod = df_reason_mod.drop('Date', axis = 1)
df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod
df_reason_date_mod['Education'].value_counts()
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0,2:1,3:1,4:1})
df_reason_date_mod
df_reason_date_mod['Education'].value_counts()
df_preprocessed = df_reason_date_mod.copy()
df_preprocessed
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)