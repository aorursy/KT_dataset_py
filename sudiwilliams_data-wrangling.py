#Import pandas library
import pandas as pd
#Save your dataframe into a variable so that you can keep working with it
Military_Expenditure = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv")
#The shape attribute returns a python tool
#The first value is the number of rows and the second value is the number of columns
Military_Expenditure.shape
#DataFrame, index dtype, column dtypes, non-null values and memory usage
Military_Expenditure.info()
#Head gives us the first five rows
#A pandas dataframe has three components: index, column, and value(body of the dataframe).
Military_Expenditure.head()
Name = Military_Expenditure[['Name', 'Code']]
Name
#loc matches the index label and iloc matches the index position
Military_Expenditure.loc[[0, 1, 2]]
rows_columns = Military_Expenditure.loc[0:10, ['Name', 'Code', 'Type']]
rows_columns.head
#Multiple Criteria Filtering
filter_list = ['Regions Clubbed Economically', 'Semi Autonomous Region', 'Regions Clubbed Geographically']
Military_Expenditure[Military_Expenditure.Type.isin(filter_list)]
#This data set has a column named Indicator Name that should be removed since it adds no value to the data set
Military_Expenditure.drop(['Indicator Name'], axis='columns', inplace=True)
Military_Expenditure.head()
#When column headers are values and not variables the data set needs to be melted.
Military_Expenditure= Military_Expenditure.melt(id_vars=['Name', 'Code', 'Type'],
                                         var_name='Year', value_name='Expenditure_USD')
Military_Expenditure.head()
#Wrong data type is assigned to a feature
Military_Expenditure.dtypes
#Convert
Military_Expenditure[["Year"]] = Military_Expenditure[["Year"]].astype("int")
#List the columns after the conversion
Military_Expenditure.dtypes
Military_Expenditure.head()
#Count missing values in each column
for column in Military_Expenditure.columns.values.tolist():
    print(column)
    print (Military_Expenditure[column].value_counts())
    print("")
#Replace all NaN elements with 0s
Military_Expenditure.fillna(value=0, inplace=True)
Military_Expenditure.head()