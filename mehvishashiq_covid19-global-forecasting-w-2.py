import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB
train_file_path = '/kaggle/input/covid19-global-forecasting-week-2/train.csv'

test_file_path = '/kaggle/input/covid19-global-forecasting-week-2/test.csv'



train_dataset = pd.read_csv(train_file_path)

test_dataset = pd.read_csv(test_file_path)

print("Understanding of Train Dataset:\n\n\n")



print('Train Dataset has following states:\n')

province_state = train_dataset['Province_State'].unique()

print(province_state)





print('\n\n\nTrain Dataset has following Country Region:\n')

country_region = train_dataset['Country_Region'].unique()

print(country_region)





print('\n\n\nTrain Dataset has records of following dates:\n')

dates = train_dataset['Date'].unique()

print(dates)

#convert to mm/dd/yyyy

train_dataset['Date'] = pd.to_datetime(train_dataset['Date'])

print('\nTrain Dataset has following Date Range:')

print(pd.date_range(start=train_dataset['Date'].min(), end=train_dataset['Date'].max()))











print("Understanding of Test Dataset:\n\n\n")



print('Test Dataset has following states:\n')

province_state = test_dataset['Province_State'].unique()

print(province_state)





print('\n\n\nTest Dataset has following Country Region:\n')

country_region = test_dataset['Country_Region'].unique()

print(country_region)





print('\n\n\nTest Dataset has records of following dates:\n')

dates = test_dataset['Date'].unique()

print(dates)

#convert to mm/dd/yyyy

test_dataset['Date'] = pd.to_datetime(test_dataset['Date'])

print('\nTest Dataset has following Date Range:')

print(pd.date_range(start=test_dataset['Date'].min(), end=test_dataset['Date'].max()))











print("Train Dataset Graphical Representation of Counrtry Region w.r.t. Confirmed Cases")

show_cumulatively = train_dataset.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()

plt.figure(figsize=(20,10))

#sns.set()

sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cumulatively[show_cumulatively['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(50))





print("Train Dataset Graphical Representation of Counrtry Region w.r.t. Fatalities")

plt.figure(figsize=(20,10))

sns.barplot(x='Fatalities',y='Country_Region',data=show_cumulatively[show_cumulatively['Fatalities'] != 0].sort_values(by='Fatalities',ascending=False).head(50))
print('Those Country Regions of Train Dataset whose Confirmed Cases have Fatalities')

non_fatalities_train_df = train_dataset[train_dataset['Fatalities'] != 0] 

non_fatalities_train_df[['Country_Region','Date','ConfirmedCases','Fatalities']]
print('Those Country Regions whose Confirmed Cases have not Fatalities')

non_fatalities_train_df = train_dataset[train_dataset['Fatalities'] == 0] 

non_fatalities_train_df[['Country_Region','Date','ConfirmedCases','Fatalities']]
print('The value count of Country Regions of Non-Null Province States in Train Dataset')

print(train_dataset[~train_dataset['Province_State'].isnull()]['Country_Region'].value_counts())





print('\n\n\nThe value count of Country Regions of Null Province States in Train Dataset')

print(train_dataset[train_dataset['Province_State'].isnull()]['Country_Region'].value_counts())
print('The value count of Country Regions of Non-Null Province States in Test Dataset')

print(test_dataset[~test_dataset['Province_State'].isnull()]['Country_Region'].value_counts())





print('\n\n\nThe value count of Country Regions of Null Province States in Test Dataset')

print(test_dataset[test_dataset['Province_State'].isnull()]['Country_Region'].value_counts())
print("Train dataset before pre-processing:\n")

print(train_dataset.head())



train_dataset = train_dataset.fillna('Enpyty_value')



print("\n\n\nTrain dataset after pre-processing:\n")

print(train_dataset.head())
print("Test dataset before pre-processing:\n")

print(test_dataset.head())



test_dataset = test_dataset.fillna('Enpyty_value')



print("\n\n\nTest dataset after pre-processing:\n")

print(test_dataset.head())
labelEncoder = LabelEncoder()

train_dataset['Date'] = pd.to_datetime(train_dataset['Date']).dt.strftime("%m%d").astype(int) 

train_dataset['Date'] -= 122 

test_dataset['Date'] = pd.to_datetime(test_dataset['Date']).dt.strftime("%m%d").astype(int) 

test_dataset['Date'] -= 122 



train_dataset.Province_State = labelEncoder.fit_transform(train_dataset.Province_State)

train_dataset.Country_Region = labelEncoder.fit_transform(train_dataset.Country_Region)



test_dataset.Province_State = labelEncoder.fit_transform(test_dataset.Province_State)

test_dataset.Country_Region = labelEncoder.fit_transform(test_dataset.Country_Region)





print('\n\n\nTrain Dataset After Encoding')

print(train_dataset.head(5))

      



print('\n\n\nTest Dataset After Encoding')

print(test_dataset.head(5))



#We don't need to convert it into vector because it is alreayd in vector form. See following



print(train_dataset.head())

print(test_dataset.head())



X = train_dataset[['Province_State','Country_Region','Date']]

y = train_dataset[['ConfirmedCases','Fatalities']]



classifier = BinaryRelevance(GaussianNB())



# train

classifier.fit(X, y[['Fatalities']])



# predict

predictions_fatalities = classifier.predict(test_dataset[['Province_State','Country_Region','Date']])



# train

classifier.fit(X, y[['ConfirmedCases']])



# predict

predictions_confirmed_cases = classifier.predict(test_dataset[['Province_State','Country_Region','Date']])

output_confirmed_cases_df = pd.DataFrame(data=predictions_confirmed_cases.toarray())

output_fatalities_df = pd.DataFrame(data=predictions_fatalities.toarray())



output_confirmed_cases_df = output_confirmed_cases_df.rename(columns={0: "ConfirmedCases"})

output_fatalities_df = output_fatalities_df.rename(columns={0: "Fatalities"})
test_dataset.ForecastId
result = pd.concat([test_dataset.ForecastId,output_confirmed_cases_df, output_fatalities_df], axis=1)

# result.index.names = ['indexes']

# result.columns.name = result.index.name

# result.index.name = None

result

result.to_csv('submission.csv', index=False)
