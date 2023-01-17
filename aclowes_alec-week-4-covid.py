import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv", parse_dates=['Date'])
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", parse_dates=['Date'])
countryinfo = pd.read_csv(
    "../input/countryinfo/covid19countryinfo.csv", 
    parse_dates=['quarantine', 'schools', 'publicplace', 'nonessential', 'gathering']
)
countryinfo['pop'] = countryinfo['pop'].apply(lambda x: float(str(x).replace(',', '')))
states = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv", parse_dates=['date'])
state_pops = pd.read_csv("../input/us-state-populations-2018/State Populations.csv")
state_abbr = pd.read_csv("../input/state-abbreviations/state_abbrev.csv")
# pick out regions with more than 1000 cases on April 12th
filtered = (
    train[
        (train["Date"] == "2020-04-12") &
        (train['ConfirmedCases'] > 1000)
    ]
)[['Country_Region', 'Province_State']]
print(f'Found {len(filtered)} matching regions')

dataset = train.merge(
    filtered,
    on=['Country_Region', 'Province_State']
)[train['ConfirmedCases'] > 100]

dataset[train["Date"] == "2020-04-12"]\
.sort_values(by=['ConfirmedCases'], ascending=False).head()
# merge datasets
merged = dataset.merge(
    countryinfo, 
    left_on=['Country_Region', 'Province_State'],
    right_on=['country', 'region'],
    how='left'
).merge(
    state_pops,
    left_on=['Province_State'],
    right_on=['State'],
    how='left'
).merge(
    state_abbr,
    left_on=['Province_State'],
    right_on=['State'],
    how='left'
).merge(
    states,
    left_on=['Abbreviation', 'Date'],
    right_on=['state', 'date'],
    how='left'
)
# print(merged[merged["Country_Region"] == "Italy"].head(10))
merged['pop'].fillna(merged['2018 Population'], inplace=True)
merged['tests'].fillna(merged['posNeg'], inplace=True)
merged['Province_State'].fillna('', inplace=True)
# these are all shut as well if there is a quarantine
merged['nonessential'].fillna(merged['quarantine'], inplace=True)
merged['publicplace'].fillna(merged['quarantine'], inplace=True)
merged['gathering'].fillna(merged['quarantine'], inplace=True)
merged['schools'].fillna(merged['quarantine'], inplace=True)
merged = merged[[
    'Country_Region', 'Province_State', 'Date', 'ConfirmedCases', 'density',
    'tests', 'pop', 'quarantine', 'schools', 'publicplace', 'nonessential', 'gathering']]
merged['CasesPerK'] = merged['ConfirmedCases'] / merged['pop'] * 1000
merged['TestsPerK'] = merged['tests'] / merged['pop'] * 1000
merged['TestsPositive'] = merged['ConfirmedCases'] / merged['tests']
merged = merged[(merged["tests"] > 100) & (merged['ConfirmedCases'] > 100)]
merged[merged["Date"] == "2020-04-12"]\
.sort_values(by=['ConfirmedCases'], ascending=False).head(10)
# the daily percent increase in confirmed cases
by_ctry_prov = merged.groupby(['Country_Region','Province_State'])[['CasesPerK']]
merged[['DeltaCasesPerK']]= by_ctry_prov.transform(lambda x: x.diff().fillna(0))
by_ctry_prov = merged.groupby(['Country_Region','Province_State'])[['CasesPerK']]
period = 1
merged[['AvgDeltaCasesPerK']]= by_ctry_prov.transform(lambda x: x.diff(periods=period).fillna(0) / period)
# the days since restriction was added
merged['quarantine_days'] = (merged['Date'] - merged['quarantine']).transform(lambda x: x.days).fillna(0)
merged['schools_days'] = (merged['Date'] - merged['schools']).transform(lambda x: x.days).fillna(0)
merged['publicplace_days'] = (merged['Date'] - merged['publicplace']).transform(lambda x: x.days).fillna(0)
merged['nonessential_days'] = (merged['Date'] - merged['nonessential']).transform(lambda x: x.days).fillna(0)
merged['gathering_days'] = (merged['Date'] - merged['gathering']).transform(lambda x: x.days).fillna(0)
final = merged[[
    'Country_Region', 'Province_State', 'Date', 'DeltaCasesPerK',
    'CasesPerK', 'TestsPerK', # 'AvgDeltaCasesPerK',
    'quarantine_days', 'schools_days', 'publicplace_days', 'nonessential_days',
    'gathering_days', 'TestsPositive'
]]
# final[final["Date"] == "2020-04-12"].head(10)
final[final["Country_Region"] == "Italy"].head(20)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
y = final['DeltaCasesPerK']
X = final
train_X, test_X, train_y, test_y = train_test_split(
    X.as_matrix(), y.as_matrix(), test_size=0.25)
train_index = train_X[:,:4]
train_X = train_X[:,4:]
test_index = test_X[:,:4]
test_X = test_X[:,4:]
my_model = XGBRegressor(n_estimators=100, max_depth=3)
# was 0.0698 before parameters, bounces around with each run
my_model.fit(train_X, train_y, verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
output = pd.DataFrame(data=test_X, columns=X.columns[4:])
output['DeltaCasesActual'] = test_y
output['DeltaCasesPrediction'] = predictions
output['Country_Region'] = test_index[:,0]
output['Province_State'] = test_index[:,1]
output['Date'] = test_index[:,2]
output.head(10)
# plot cases (y) by feature value (X)
import numpy as np

# 'Country_Region', 'Province_State', 'Date', 'DeltaCasesPerK',
# 'AvgDeltaCasesPerK',
features = ('CasesPerK', 'TestsPerK',
    'quarantine_days', 'schools_days', 'publicplace_days', 'nonessential_days',
    'gathering_days', 'TestsPositive')

maximums = [5, 25, 100, 100, 100, 100, 100, 1]

for index, maximum in enumerate(maximums):
    feature_X = np.array([[.5, 2.5, 20, 20, 20, 20, 20, 0.2]] * 100)
    feature_X[:, index] = np.arange(0, maximum, maximum/100)
    feature_y = my_model.predict(feature_X)
    plt.figure(figsize=(14,2))    
    plt.subplot(1,2,1)
    plt.plot(feature_X[:, index], feature_y)
    plt.title(features[index])
    plt.ylabel('DeltaCases')
    plt.show()