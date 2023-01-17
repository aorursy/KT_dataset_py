# import standard libraries
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from statistics import mean
# read in data (provided via course website)
data_dir = '/kaggle/input/coronavirusdataset'
case = pd.read_csv(os.path.join(data_dir, 'Case.csv'), na_values='-')
patient_route = pd.read_csv(os.path.join(data_dir, 'PatientRoute.csv'))
region = pd.read_csv(os.path.join(data_dir, 'Region.csv'))

# Subsampling patient route to improve map visualization performance
patient_route = patient_route.sample(750, random_state=1)

# count based on province only
case_province = pd.DataFrame(case['confirmed'].groupby(case['province']).sum())

# get average geocode info
# gc = case[['latitude', 'longitude']].groupby(case['province']).mean()

# get average geocode info from Region dataset (which is more complete)
gc = region[['latitude', 'longitude']].groupby(region['province']).mean()

# combine into single dataset
df_case = case_province.join(gc)

# center of lon & lat average
center = df_case.mean()
# generate map object for cases
fig = px.scatter_mapbox(df_case,
                        lat='latitude', lon='longitude', 
                        size='confirmed', size_max=50,
                        hover_name=df_case.index)

# update layout
fig.update_layout(mapbox_style= "carto-positron", 
                  mapbox_zoom=6, 
                  mapbox_center_lat = center['latitude'],
                  mapbox_center_lon = center['longitude'],
                  
                  margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.line_mapbox(patient_route, 
                     lat='latitude', lon='longitude', 
                     color='patient_id')

fig.update_layout(mapbox_style= "carto-positron", 
                  mapbox_zoom=6, 
                  mapbox_center_lat = center['latitude'],
                  mapbox_center_lon = center['longitude'],
                  margin={"r":0,"t":0,"l":0,"b":0})

fig.update_layout(showlegend=False)

fig.show()
region_counts = region[['nursing_home_count','kindergarten_count', 'elementary_school_count','university_count']]
region_counts = region_counts.groupby(region['province']).sum()
region_counts = region_counts.join(gc)
region_counts['province'] = region_counts.index
region_melt = pd.melt(region_counts, id_vars=['province', 'latitude', 'longitude'])
fig = px.scatter_mapbox(region_melt, 
                        lat='latitude', lon='longitude',
                        color='variable', opacity=0.3,
                        size='value', size_max=50)

fig.update_layout(mapbox_style= "carto-positron", 
                  mapbox_zoom=6, 
                  mapbox_center_lat = center['latitude'],
                  mapbox_center_lon = center['longitude'],
                  margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# read in data
time = pd.read_csv(os.path.join(data_dir, 'Time.csv'))

time_age = pd.read_csv(os.path.join(data_dir, 'TimeAge.csv'))
time_gender = pd.read_csv(os.path.join(data_dir, 'TimeGender.csv'))
time_province = pd.read_csv(os.path.join(data_dir, 'TimeProvince.csv'))
time_cumulative = time[['test', 'confirmed']].cumsum()
time_cumulative['date'] = time['date']
time_cumulative_melt = pd.melt(time_cumulative, id_vars=['date'])
# fig = px.line(time_cumulative_melt, x = 'date', y = 'value', color = 'variable')
# fig.show()

# time_cumulative['confirmed_ratio'] = time_cumulative['confirmed'] / time_cumulative['test']
# fig = px.line(time_cumulative, x = 'date', y = 'confirmed_ratio')
# fig.show()

fig = px.bar(time, x = 'date', y = 'confirmed')
fig.show()

# fig = px.line(time_cumulative, x = 'date', y = 'confirmed')
# fig.show()
time_age_cumulative = time_age[['confirmed', 'deceased']].groupby(time_age['age']).cumsum()
time_age_cumulative = time_age[['date', 'age']].join(time_age_cumulative)
time_age_cumulative_melt = pd.melt(time_age_cumulative, id_vars=['date', 'age'])
fig = px.line(time_age_cumulative_melt, x = 'date', y = 'value', color = 'variable', facet_col='age', facet_col_wrap=5)
fig.show()
time_gender_cumulative = time_gender[['confirmed', 'deceased']].groupby(time_gender['sex']).cumsum()
time_gender_cumulative = time_gender[['date', 'sex']].join(time_gender_cumulative)
time_gender_cumulative_melt = pd.melt(time_gender_cumulative, id_vars=['date', 'sex'])
fig = px.line(time_gender_cumulative_melt, 
              x = 'date', y = 'value', 
              color = 'sex', facet_row='variable')
fig.update_yaxes(matches=None)
fig.show()
SearchTrend = pd.read_csv(os.path.join(data_dir, 'SearchTrend.csv'))
search = SearchTrend.drop('date', axis=1)
search.describe()
search.skew()
search.kurt()
search_melt = pd.melt(search)
search_melt.head()
fig = px.box(search_melt, y = 'value', color = 'variable', log_y=True)

fig.show()
fig = px.scatter_matrix(search)
fig.show()
weather = pd.read_csv(os.path.join(data_dir, 'Weather.csv'))

#print(weather)

weather.index = weather['date']
weather = weather.drop('date', axis=1)
temp = weather[['avg_temp', 'min_temp', 'max_temp']]
precip = weather[['precipitation', 'avg_relative_humidity']]
wind = weather[['max_wind_speed', 'most_wind_direction']]

search.index = SearchTrend['date']
#print(search.index)
temp_trend = search.join(temp)
print(temp_trend)
temp_trend_melt = pd.melt(temp_trend, id_vars=['cold', 'flu', 'pneumonia', 'coronavirus'],
                          var_name = 'weather_var', value_name='weather_val')

temp_trend_melt2 = pd.melt(temp_trend_melt, id_vars = ['weather_var', 'weather_val'], 
                           var_name = 'search_var', value_name='search_val')

fig = px.scatter(temp_trend_melt2, x = 'weather_val', y = 'search_val', 
                 facet_col = 'weather_var', facet_row = 'search_var')
fig.update_yaxes(matches=None)
fig.show()
precip_trend = search.join(precip)

precip_trend_melt = pd.melt(precip_trend, id_vars = ['cold', 'flu', 'pneumonia', 'coronavirus'],
                           var_name = 'precip_var', value_name = 'precip_val')

precip_trend_melt2 = pd.melt(precip_trend_melt, id_vars = ['precip_var', 'precip_val'],
                            var_name = 'search_var', value_name = 'search_val')

fig = px.scatter(precip_trend_melt2, x = 'precip_val', y = 'search_val',
                facet_col = 'precip_var', facet_row = 'search_var')

fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
fig.show()
wind_search = search.join(wind)

wind_search_melt = pd.melt(wind_search, id_vars = ['cold', 'flu', 'pneumonia', 'coronavirus'],
                          var_name = 'wind_var', value_name = 'wind_val')

wind_search_melt2 = pd.melt(wind_search_melt, id_vars = ['wind_var', 'wind_val'],
                           var_name = 'search_var', value_name = 'search_val')

fig = px.scatter(wind_search_melt2, x = 'wind_val', y = 'search_val', 
                 facet_col = 'wind_var', facet_row = 'search_var')

fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
fig.show()
wind_search_melt2
# Loading additional tables
patient_info = pd.read_csv(os.path.join(data_dir, 'PatientInfo.csv'))

# Total confirmed cases in Korea
numConfirmedCases = case['confirmed'].sum();
display('Total confirmed cases: ', numConfirmedCases)
# Model 1: Perform SVM on data

patient_info['province_cat_codes'] = patient_info.province.astype('category').cat.codes
patient_info['city_cat_codes'] = patient_info.city.astype('category').cat.codes
patient_info['sex_cat_codes'] = patient_info.sex.astype('category').cat.codes
patient_info['age_cat_codes'] = patient_info.age.astype('category').cat.codes
patient_info['state_cat_codes'] = patient_info.state.astype('category').cat.codes
patient_info['disease_cat_codes'] = patient_info.disease.astype('category').cat.codes

# Combine isolated and recovered patients into a single category to represent all living patients
patient_info.loc[
   (patient_info['state_cat_codes'] == 2)
   , 'state_cat_codes'] = 1

# The accuracy went up when I removed those patients whose birth year we didn't know
#patient_info = patient_info[patient_info['birth_year'].notnull()]

# There are way more trues than falses. So filter down
temp = patient_info.loc[patient_info['state_cat_codes'] == 1]

temp3 = patient_info.loc[patient_info['state_cat_codes'] == 0]

temp2 = temp.sample(n=len(temp3), random_state=1)

frames = [temp2, temp3]
balanced_patient_info = pd.concat(frames)
cor = balanced_patient_info.corr()
cor_target = abs(cor["state_cat_codes"])
relevant_features = cor_target[cor_target>0.5]

X = balanced_patient_info[['province_cat_codes','age_cat_codes', 'disease_cat_codes']]
y = balanced_patient_info['state_cat_codes']
accuracies = list()
precisions = list()
recalls = list()
# Perform 2000 MC reps. There is no change after this to in the first two decimal places
for i in range(2000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_train = y_train.astype(np.int8)

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_test = y_test.to_numpy()
    temp = np.vstack((y_test,y_pred)).transpose()

    accuracies.append(metrics.accuracy_score(y_test, y_pred))
    precisions.append(metrics.precision_score(y_test, y_pred))
    recalls.append(metrics.recall_score(y_test, y_pred))

print("Accuracy: ",mean(accuracies))
print("Precision: ",mean(precisions))
print("Recall: ",mean(recalls))
# Model 2: Determine if there is a correlation between the number of places and individual has gone and how many people they infect

# Question: Which individuals affected the most other people?
infectors = patient_info.groupby('infected_by').count().reset_index()
infectors['num_patients_infected'] = infectors['patient_id']
patientCausesOfInfection = infectors[['infected_by','num_patients_infected']].sort_values(by=['num_patients_infected'],ascending=False)

# Top 3 culprits
print('Top 3 Infectors in Korea')
display(patientCausesOfInfection[0:3])

print('Bottom 3 Infectors in Korea (must have infected at least one person)')
display(patientCausesOfInfection[-1:-4:-1])

# Top culprit
topInfected_by = patientCausesOfInfection.head(1)
topInfected_by = topInfected_by['infected_by'].to_numpy()[0]

# Information about the top culprit
print('Top infector in Korea')
display(patient_info[patient_info['patient_id'] == topInfected_by])
print('')

# All patients affected by top culprit
print('All patients affected by top infector')
display(patient_info[patient_info['infected_by'] == topInfected_by])

patient_route = pd.read_csv(os.path.join(data_dir, 'PatientRoute.csv'))

# Percentage of patients we know the routes for
print('Percentage of patients we have routes for: ', len(patient_route['patient_id'].value_counts()) / patient_info.shape[0] * 100)

# This was another possible route: to track the top and bottom infectors and compare differences between them. 
# We moved on from this idea because the route data was too sparse and proved to be too much of a challenge

# Unable to track first and third because they have no routes
# top3 = patientCausesOfInfection.head(3)
# firstInfected_by = top3['infected_by'].to_numpy()[0]
# secondInfected_by = top3['infected_by'].to_numpy()[1]
# thirdInfected_by = top3['infected_by'].to_numpy()[2]
# lastInfected_by = patientCausesOfInfection['infected_by'].to_numpy()[-3]

# Who infected the second top?
# temp = patient_info[patient_info['patient_id'] == secondInfected_by]
# Nobody knows! It's NaN

# Join patient route to info
patient_info_route = patient_info.merge(patient_route, on='patient_id', how='left')

# Count up the number of routes. Counting the number of times a patient ID appears works because every row is a route
num_places = patient_info_route['patient_id'].value_counts()
infectors = patient_info['infected_by'].value_counts()

# Merge number of places into the table
temp1 = num_places.keys().to_frame(name='patient_id').to_numpy()
temp2 = num_places.to_frame(name='num_places').to_numpy()
temp2 = pd.DataFrame(np.concatenate((temp1,temp2), axis=1))
temp3 = temp2.rename(columns={0: "patient_id", 1: "num_places"})

filtered_patient_info_route = patient_info_route.merge(temp3, on='patient_id', how='left')

# Merge number of infected by individuals into the table
temp1 = infectors.keys().to_frame(name='patient_id').to_numpy()
temp2 = infectors.to_frame(name='infected').to_numpy()
temp2 = pd.DataFrame(np.concatenate((temp1,temp2), axis=1))
temp3 = temp2.rename(columns={0: "patient_id", 1: "infected"})

filtered_patient_info_route = filtered_patient_info_route.merge(temp3, on='patient_id', how='left')

# Remove all those that don't have route
filtered_patient_info_route = filtered_patient_info_route[filtered_patient_info_route['type'].notnull()]

# Remove all those that don't have infected
filtered_patient_info_route = filtered_patient_info_route[filtered_patient_info_route['infected'].notnull()]

fig = plt.figure()
ax = plt.subplot(111)
filtered_patient_info_route.plot(x='num_places',y='infected',ax=ax, kind='scatter')

# Question: Which cases had the biggest outbreak?

case_infection_case = case.groupby('infection_case').sum().reset_index()
confirmed_by_infection_case = case_infection_case[['infection_case','confirmed']].sort_values(by=['confirmed'],ascending=False)

# Top 5 culprits:
confirmed_by_infection_case['percentage_of_total'] = (confirmed_by_infection_case['confirmed'] / case['confirmed'].sum()) * 100
display(confirmed_by_infection_case[0:5])
# Question: How many patients do we have data on and what is the largest infection case associated with this?

# What percentage of patients do we have info for?
display('Percentage of patients we have info for: ', patient_info.shape[0] / numConfirmedCases * 100)

# Highest infection case type
patient_info_infection_case = patient_info.groupby('infection_case').count().reset_index()
patient_info_infection_case['num_patients'] = patient_info_infection_case['patient_id']
patientCausesOfInfection = patient_info_infection_case[['infection_case','num_patients']].sort_values(by=['num_patients'],ascending=False)
display(patientCausesOfInfection)
# Question: Who was the first infector in Korea?

display(patient_info.sort_values('confirmed_date').head(1))