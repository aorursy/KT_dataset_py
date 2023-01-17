#Import libraries

import numpy as np

import pandas as pd



#Read in data

df = pd.read_csv("../input/pga-tour-20102018-data/PGA_Data_Historical.csv")



#Number of Rows

print("Number of Rows: " + str(len(df)))

#Number of Columns

print("Number of Columns: " + str(len(df.columns)))
#Transpose based on key value pairs

wide_df = df.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()



#Number of Rows

print("Number of Rows: " + str(len(wide_df)))

#Number of Columns

print("Number of Columns: " + str(len(wide_df.columns)))
#%% Missing value exploration



number_non_missing_row = wide_df.apply(lambda x: x.count(), axis=1)



number_of_columns = len(wide_df.columns)

                                  

#Creates data frame of above

missing_row_value_df = pd.DataFrame({'Player Name': wide_df['Player Name'],'Season': wide_df['Season'],

                                     'number_non_missing_row': number_non_missing_row, 

                                     'number_of_columns': number_of_columns, 

                                     'percent_missing_row':(1 - (number_non_missing_row/number_of_columns)) * 100})



#Removes number_non_missing_row & number_of_columns 

missing_row_value_df.drop(columns=['number_non_missing_row','number_of_columns'],inplace=True)



#Sorts values low percentage to high

missing_row_value_df.sort_values('percent_missing_row', inplace=True)



#Plot missing values in rows

rowplot = missing_row_value_df.plot(x='Player Name', y='percent_missing_row', style='o', 

                                         title='Percent of Statistics Missing For Each Player Per Week', legend=False)

rowplot.axes.get_xaxis().set_visible(False)

rowplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row

print('Percent of Statistics Missing For Each Player:\n' + str(missing_row_value_df['percent_missing_row'].describe()))



missing_row_value_df['percent_missing_row'] = missing_row_value_df['percent_missing_row'].astype(float)



#Missing data columns data frame

missing_column_value_df = pd.DataFrame({'column_name': wide_df.columns,

                                 'percent_missing': wide_df.isnull().sum() *100 / len(wide_df)})



#Sorts data low percentage to high

missing_column_value_df.sort_values('percent_missing', inplace=True)



#Plot missing values in rows

columnplot = missing_column_value_df.plot(x='column_name', y='percent_missing', style='o', 

                                   title='Percent of Players Missing For Each Statistic', legend=False)

columnplot.axes.get_xaxis().set_visible(False)

columnplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row column

print('Percent of Players Missing For Each Statistic:\n' + str(missing_column_value_df['percent_missing'].describe()))



missing_column_value_df['percent_missing'] = missing_column_value_df['percent_missing'].astype(float)
#%% Drops rows with large amounts of missing data



# Sets threshold on data with 40% missing values

row_threshold = len(wide_df.columns) - (0.4 * len(wide_df.columns))



# Drops rows with more than 40% missing data

df_row_threshold = wide_df.dropna(thresh=row_threshold)
#%% Missing value exploration after row threshold



number_non_missing_row = df_row_threshold.apply(lambda x: x.count(), axis=1)



number_of_columns = len(df_row_threshold.columns)

                                  

#Creates data frame of above

missing_row_value_df = pd.DataFrame({'Player Name': df_row_threshold['Player Name'],'Season': df_row_threshold['Season'],

                                     'number_non_missing_row': number_non_missing_row, 

                                     'number_of_columns': number_of_columns, 

                                     'percent_missing_row':(1 - (number_non_missing_row/number_of_columns)) * 100})



#Removes number_non_missing_row & number_of_columns 

missing_row_value_df.drop(columns=['number_non_missing_row','number_of_columns'],inplace=True)



#Sorts values low percentage to high

missing_row_value_df.sort_values('percent_missing_row', inplace=True)



#Plot missing values in rows

rowplot = missing_row_value_df.plot(x='Player Name', y='percent_missing_row', style='o', 

                                         title='Percent of Statistics Missing For Each Player Per Week', legend=False)

rowplot.axes.get_xaxis().set_visible(False)

rowplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row

print('Percent of Statistics Missing For Each Player:\n' + str(missing_row_value_df['percent_missing_row'].describe()))



missing_row_value_df['percent_missing_row'] = missing_row_value_df['percent_missing_row'].astype(float)



#Missing data columns data frame

missing_column_value_df = pd.DataFrame({'column_name': df_row_threshold.columns,

                                 'percent_missing': df_row_threshold.isnull().sum() *100 / len(df_row_threshold)})



#Sorts data low percentage to high

missing_column_value_df.sort_values('percent_missing', inplace=True)



#Plot missing values in rows

columnplot = missing_column_value_df.plot(x='column_name', y='percent_missing', style='o', 

                                   title='Percent of Players Missing For Each Statistic', legend=False)

columnplot.axes.get_xaxis().set_visible(False)

columnplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row column

print('Percent of Players Missing For Each Statistic:\n' + str(missing_column_value_df['percent_missing'].describe()))



missing_column_value_df['percent_missing'] = missing_column_value_df['percent_missing'].astype(float)

#%% Drops all distance analysis columns and explore missing values



df_column_threshold = df_row_threshold.drop(df_row_threshold.filter(regex='Distance Analysis').columns, axis=1)



number_non_missing_row = df_column_threshold.apply(lambda x: x.count(), axis=1)



number_of_columns = len(df_column_threshold.columns)

                                  

#Creates data frame of above

missing_row_value_df = pd.DataFrame({'Player Name': df_column_threshold['Player Name'],'Season': df_column_threshold['Season'],

                                     'number_non_missing_row': number_non_missing_row, 

                                     'number_of_columns': number_of_columns, 

                                     'percent_missing_row':(1 - (number_non_missing_row/number_of_columns)) * 100})



#Removes number_non_missing_row & number_of_columns 

missing_row_value_df.drop(columns=['number_non_missing_row','number_of_columns'],inplace=True)



#Sorts values low percentage to high

missing_row_value_df.sort_values('percent_missing_row', inplace=True)



#Plot missing values in rows

rowplot = missing_row_value_df.plot(x='Player Name', y='percent_missing_row', style='o', 

                                         title='Percent of Statistics Missing For Each Player Per Week', legend=False)

rowplot.axes.get_xaxis().set_visible(False)

rowplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row

print('Percent of Statistics Missing For Each Player:\n' + str(missing_row_value_df['percent_missing_row'].describe()))



missing_row_value_df['percent_missing_row'] = missing_row_value_df['percent_missing_row'].astype(float)



#Missing data columns data frame

missing_column_value_df = pd.DataFrame({'column_name': df_column_threshold.columns,

                                 'percent_missing': df_column_threshold.isnull().sum() *100 / len(df_column_threshold)})



#Sorts data low percentage to high

missing_column_value_df.sort_values('percent_missing', inplace=True)



#Plot missing values in rows

columnplot = missing_column_value_df.plot(x='column_name', y='percent_missing', style='o', 

                                   title='Percent of Players Missing For Each Statistic', legend=False)

columnplot.axes.get_xaxis().set_visible(False)

columnplot.set_ylabel('Percent (%)')



#Summary statistics of percent missing row column

print('Percent of Players Missing For Each Statistic:\n' + str(missing_column_value_df['percent_missing'].describe()))



missing_column_value_df['percent_missing'] = missing_column_value_df['percent_missing'].astype(float)
#Perent of missing values in entire dataframe

print(str((df_column_threshold.isnull().sum().sum()) / (len(df_column_threshold.columns)*len(df_column_threshold)) * 100) + '%')