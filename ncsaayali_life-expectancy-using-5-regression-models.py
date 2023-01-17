import pandas as pd

Life_Expectancy_Data = pd.read_csv("../input/life-expectancy/Life Expectancy Data.csv")

Life_Expectancy_Data
Life_Expectancy_Data.info()
Life_Expectancy_Data.columns
Life_Expectancy_Data.columns = Life_Expectancy_Data.columns.str.strip()

Life_Expectancy_Data.columns
Life_Expectancy_Data.columns = Life_Expectancy_Data.columns.str.replace (" ", "_")

Life_Expectancy_Data.columns
Life_Expectancy_Data.rename(columns = {'HIV/AIDS': 'HIV'}, inplace = True)

Life_Expectancy_Data.columns
Life_Expectancy_Data.replace(regex=["CÃ´te d'Ivoire"], value='Ivory Coast')
# Displaying the count of Null values in each column

Life_Expectancy_Data.isnull().sum()
# Creating multiple dataframes based upon Country column

data1=Life_Expectancy_Data[['Country','GDP']]

data2=Life_Expectancy_Data[['Country','Population']]

data3=Life_Expectancy_Data[['Country','Schooling']]

data4=Life_Expectancy_Data[['Country','Alcohol']]

data5=Life_Expectancy_Data[['Country','BMI']]

data6=Life_Expectancy_Data[['Country','Total_expenditure']]

data7=Life_Expectancy_Data[['Country','thinness__1-19_years']]

data8=Life_Expectancy_Data[['Country','Adult_Mortality']]

data9=Life_Expectancy_Data[['Country','Polio']]

data10=Life_Expectancy_Data[['Country','Diphtheria']]

data11=Life_Expectancy_Data[['Country','Hepatitis_B']]
# Grouping and displaying the dataframe with the mean values of the specific country column

grouped1 = data1.groupby('Country').mean()

grouped2 = data2.groupby('Country').mean()



print(grouped1)

print(grouped2)
# Grouping by country and replacing NULL values with country specific mean values



data1["GDP"] = data1.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data2["Population"] = data2.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data3["Schooling"] = data3.groupby("Country").transform(lambda x: x.fillna(0))

data4["Alcohol"] = data4.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data5["BMI"] = data5.groupby("Country").transform(lambda x: x.fillna(0))

data6["Total expenditure"] = data6.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data7["thinness__1-19_years"] = data7.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data8["Adult Mortality"] = data8.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data9["Polio"] = data9.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data10["Diphtheria"] = data10.groupby("Country").transform(lambda x: x.fillna(x.mean()))

data11["Hepatitis_B"] = data10.groupby("Country").transform(lambda x: x.fillna(x.mean()))



print(data1.loc[data1['Country'] == 'Somalia'])

print(data2.loc[data2['Country'] == 'Iraq'])

print(data3.loc[data3['Country'] == 'Cook Islands'])

print(data4.loc[data4['Country'] == 'Iraq'])

print(data5.loc[data5['Country'] == 'South Sudan'])

print(data6.loc[data6['Country'] == 'Algeria'])

print(data7.loc[data7['Country'] == 'Algeria'])

print(data8.loc[data8['Country'] == 'Iraq'])

print(data9.loc[data9['Country'] == 'Iraq'])

print(data10.loc[data10['Country'] == 'Iraq'])

print(data11.loc[data11['Country'] == 'Switzerland'])
# Merging the pre-processed dataset

merged_data_frame = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11],axis=1)

processed_data=merged_data_frame[['Country', 'GDP', 'Population', 'Schooling','Alcohol','BMI','Total_expenditure','thinness__1-19_years','Adult_Mortality','Polio','Diphtheria','Hepatitis_B']]

pre_processed_data=processed_data.iloc[:,10:24] 

print(pre_processed_data)
# Replacing the null values in the data dataframe with zero

pre_processed_data.fillna(0,inplace=True)
# Re-checking for NULL values

pre_processed_data.isnull().sum()
dataframe=Life_Expectancy_Data[['Country','Year', 'Status', 'Life_expectancy','infant_deaths','percentage_expenditure', 'Measles','under-five_deaths','HIV','thinness_5-9_years','Income_composition_of_resources']]

print(dataframe)
# Merging the two dataframes into single dataset

combined_data = pd.merge(pre_processed_data, dataframe, on='Country')

combined_data
# Checking for null values in the final dataset

combined_data.isnull().sum()
# Replacing the NULL values with the Zero value

combined_data.fillna(0,inplace=True)
# Re-checking the final dataset for NULL values

combined_data.isnull().sum()
# Drop "Income_composition_of_resources" column and "thinness_5-9_years"

combined_data = combined_data.drop(columns=["Income_composition_of_resources", "thinness_5-9_years"])
# Renaming the thinness__1-19_years column

combined_data = combined_data.rename(columns={'thinness__1-19_years': 'thinness_1_to_19_years'})
# Displaying the final dataset columns

combined_data.columns
# Displaying the count of number of different values present in the "Status" column

combined_data["Status"].value_counts()
# Categorizing the "Status" column using the dictionary country_status

country_status = {"Status":     {"Developed": 1, "Developing": 2}}
# Replacing the "Status" column values in the original dataset with the categorized dictionary values

combined_data.replace(country_status, inplace=True)

combined_data.head()
# Loading the external dataset

df = pd.read_excel('https://query.data.world/s/5ilfdggga4cv3psg2os532gesrtfzd')

df
df.columns
# Renaming the columns in the external dataset

df = df.rename(columns={'Country Name': 'Country', 'Country Code': 'Country_Code'})