DIR_PATH = '/kaggle/input/coronavirusdataset/'
# List of files used

import os



file_paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_paths.append(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
%config InlineBackend.figure_format = 'retina'

%matplotlib inline



import os;

import numpy as np;

import pandas as pd;

import seaborn as sns;

import folium;

import matplotlib.pyplot as plt;

import matplotlib.ticker as ticker



sns.set_style('darkgrid')
# Importing data: PopulationDistribution

pop_dist = pd.read_csv('/kaggle/input/geolocation-population-distribution-of-south-kr/PopulationDistribution.csv')

pop_dist = pop_dist.iloc[:, np.r_[0, 1, 3:12, 14:15, 27]]

pop_dist.columns = ['location', 'total', '0s', '10s', '20s', '30s', '40s', '50s',

                   '60s', '70s', '80s', 'male_total', 'female_total']

pop_dist.head()
# Tag value on bars

def show_values_on_bars(axs, h_v="v", space=0.4, modh=0, modv=0):

    def _show_on_single_plot(ax):

        if h_v == 'v':

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height() + float(modv)

                value = int(p.get_height())

                ax.text(_x, _y, value, ha='center') 

        elif h_v == 'h':

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height() - float(modh)

                value = int(p.get_width())

                ax.text(_x, _y, value, ha='left')



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
# Importing data: Gender

gender = pd.read_csv(os.path.join(DIR_PATH, 'TimeGender.csv'))

gender.head(2)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17, 7), gridspec_kw={'width_ratios': [1, 2]})

sns.set_palette(['#66b3ff','#ff9999'])



# Donut plot of confirmed cases by gender

ax1.title.set_text('Confirmed Cases ({0})'.format(gender.iloc[-1, 0]))

ax1.pie(gender.confirmed[-2:], labels=['male', 'female'], autopct='%.1f%%',

        startangle=90, pctdistance=0.85)

ax1.add_artist(plt.Circle((0, 0), 0.7, fc='white'))



# Change in time of confirmed cases

ax2.title.set_text('Confirmed Cases by Gender')

sns.lineplot(data=gender, x='date', y='confirmed', hue='sex', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.xticks(rotation=45, ha='right')



plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Growth rate of confirmed cases (Index - male: even, female: odd)

ax1.title.set_text('Growth Rate of Confirmed Cases by Gender')

gender['growth_rate'] = gender.groupby('sex')[['confirmed']].pct_change()

sns.lineplot(data=gender, x='date', y='growth_rate', hue='sex', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Decease rate of confirmed cases

ax2.title.set_text('Decease Rate of Confirmed Cases by Gender')

# Limiting y axis range to reduce fluctuations in graph

ax2.set(ylim=(-0.05, 0.5))

gender['decease_rate'] = gender.groupby('sex')[['deceased']].pct_change()

sns.lineplot(data=gender, x='date', y='decease_rate', hue='sex', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.xticks(rotation=45, ha='right')



plt.show()
# Importing data: Age

age = pd.read_csv(os.path.join(DIR_PATH, 'TimeAge.csv'))

print('Unique items: {0}'.format(len(age['age'].unique())))

age.head(9)
sns.set_palette('deep')

pop_dist_age = pop_dist.iloc[0, 2:11].str.replace(',', '')



# Population distribution by age

plt.figure(figsize=(7, 7))

plt.title('Age Distribution in South Korea')

plt.pie(pop_dist_age, labels=pop_dist_age.index, 

        autopct='%.1f%%', startangle=90, pctdistance=0.85)

plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Confirmed cases by age

ax1.title.set_text('Confirmed Cases of COVID-19')

sns.barplot(data=age[-9:], x='age', y='confirmed', ax=ax1)



# Create new column of total people in that age group

pop_dist_age = pop_dist.iloc[0, 2:11].str.replace(',', '')

age['age_total'] = np.tile(pop_dist_age, len(age) // len(pop_dist_age) + 1)[:len(age)]



# Create proportion column

age['prop_total'] = age['confirmed'] / age['age_total'].astype(float)



# Proportion of confirmed cases by age to total people in age group

ax2.title.set_text('Confirmed Cases of COVID-19 (Out of total age group)')

sns.barplot(data=age[-9:], x='age', y='prop_total', ax=ax2)



plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Confirmed cases by age

ax1.title.set_text('Confirmed Cases by Age')

sns.lineplot(data=age, x='date', y='confirmed', hue='age', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Deceased cases by age

ax2.title.set_text('Deceased Cases of Confirmed Cases by Age')

sns.lineplot(data=age, x='date', y='deceased', hue='age', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.xticks(rotation=45, ha='right')



plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Growth rate of confirmed cases

ax1.title.set_text('Growth Rate of Confirmed Cases by Age')

age['growth_rate'] = age.groupby('age')[['confirmed']].pct_change()

sns.lineplot(data=age, x='date', y='growth_rate', hue='age', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Decease rate of confirmed cases

ax2.title.set_text('Decease Rate of Confirmed Cases by Age')

age['decease_rate'] = age.groupby('age')[['deceased']].pct_change()

sns.lineplot(data=age, x='date', y='decease_rate', hue='age', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.xticks(rotation=45, ha='right')



plt.show()
# Importing data: Location

location = pd.read_csv(os.path.join(DIR_PATH, 'TimeProvince.csv'))

prov_num = len(location['province'].unique())

print(f'There are {prov_num} provinces in this dataset')



# Latest data of confirmed cases by province

loc_latest = location.iloc[-prov_num:]

loc_latest = loc_latest.sort_values('confirmed', ascending=False).reset_index(

                        drop=True).drop('time', axis=1)

loc_latest
# Latest number of confirmed & released & deceased people

fig, ax1 = plt.subplots(figsize=(15, 7))

ax1.title.set_text('COVID-19 Patients by Province')

sns.set_color_codes("pastel")

sns.barplot(data=loc_latest, x='confirmed', y='province',  label='confirmed',

            color='b', ci=None, estimator=sum)

sns.barplot(data=loc_latest, x='released', y='province', label='released',

            color='r', ci=None, estimator=sum)

sns.barplot(data=loc_latest, x='deceased', y='province', label='deceased',

            color='g', ci=None, estimator=sum)

ax1.legend(loc='lower right', frameon=True)

fig.show()
# Confirmed cases in each province (accumulated)

rows = int(prov_num / 2 + 1)

fig, ax = plt.subplots(rows, 2, figsize=(20, 6 * rows))

fig.subplots_adjust(hspace=.3)



for i, province in enumerate(loc_latest['province']):

    r, c = int(i / 2), i % 2

    sns.lineplot(data=location[location['province'] == province],

                 x='date', y='confirmed', ax=ax[r, c])

    ax[r, c].set_title(f'Confirmed Cases in {province}')

    ax[r, c].xaxis.set_major_locator(ticker.MultipleLocator(base=6))

    plt.setp(ax[r, c].xaxis.get_majorticklabels(), rotation=30, ha='right')



fig.delaxes(ax[rows - 1][rows * 2 - prov_num])

fig.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))

location['growth_rate'] = location.groupby('province')[['confirmed']].pct_change()



# Growth rate of confirmed cases in Daegu

ax1.set_title('Growth rate of confirmed cases (Daegu)')

sns.lineplot(data=location[location['province'] == 'Daegu'], x='date', y='growth_rate', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')



# Growth Rate of confirmed cases in Gyeonggi-do

ax2.set_title('Growth rate of confirmed cases (Gyeonggi-do)')

sns.lineplot(data=location[location['province'] == 'Gyeonggi-do'], x='date', y='growth_rate', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')



fig.show()
# Proportion out of total confirmed cases by province

loc_latest['proportion'] = round(loc_latest['confirmed'] / sum(loc_latest['confirmed']) * 100, 2)



# Combine provinces that consists less than 2% of total cases

loc_latest.loc['17',:] = loc_latest.iloc[4:, :].sum()

loc_latest.loc['17',['date', 'province']] = ['2020-03-30', 'Others']



sns.set_palette('deep')

loc_latest_w_etc = loc_latest.iloc[[0, 1, 2, 3, 17], [1, 5]]



# COVID-19 distribution by province

plt.figure(figsize=(7, 7))

plt.title('COVID-19 Distribution by Province')

plt.pie(loc_latest_w_etc['proportion'], labels=loc_latest_w_etc['province'], 

        autopct='%.1f%%', startangle=90, pctdistance=0.85)

plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))

plt.show()
# Importing data: Region

region = pd.read_csv(os.path.join(DIR_PATH, 'Region.csv'))

region = region.drop('nursing_home_count', axis=1)

# region = region.drop(['latitude', 'longitude', 'nursing_home_count'], axis=1)

# Drop column with same value and sort by academy_ratio

region_overview = region[region['province'] == region['city']].drop('city',

                  axis=1).drop(243).sort_values('academy_ratio', 

                  ascending=False).reset_index(drop=True)

region_overview.head()
# Add latitude and longtitude

loc_latest = loc_latest.merge(

    region_overview[['province', 'latitude','longitude']],

    on = 'province')

loc_latest['latitude'] = loc_latest['latitude'].astype(float)

loc_latest['longitude'] = loc_latest['longitude'].astype(float)

loc_latest.head()
# COVID-19 infection distribution

map_southKR = folium.Map(location=[35.9, 128], tiles="cartodbpositron",

                         zoom_start=7, max_zoom=9, min_zoom=5)

folium.Choropleth(geo_data='/kaggle/input/geolocation-population-distribution-of-south-kr/province_geo.json', 

                  fill_color='#ffff66', line_opacity=0.5, fill_opacity=0.3).add_to(map_southKR)



for i in range(0, len(loc_latest)):

    folium.Circle(location=[loc_latest.iloc[i]['latitude'], loc_latest.iloc[i]['longitude']],

                  tooltip="<h5 style='text-align:center;font-weight: bold'>" + 

                  loc_latest.iloc[i]['province'] + "</h5><hr style='margin:10px;'>" +

                  "<ul style='align-item:left;padding-left:20px;padding-right:20px'>" +

                  "<li>Confirmed: " + str(loc_latest.iloc[i]['confirmed']) + "</li>" +

                  "<li>Deaths: " + str(loc_latest.iloc[i]['deceased']) + "</li>" +

                  "<li>Mortality Rate: " + str(round(loc_latest.iloc[i]['deceased'] /

                                                     (loc_latest.iloc[i]['confirmed'] + .000001) * 100, 2)) + 

                  "%</li></ul>",

                  radius=int((np.log(loc_latest.iloc[i]['confirmed'])))*5000,

                  color='#ff3333',

                  fill_color='#ff0000',

                  fill=True).add_to(map_southKR)



map_southKR
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Academy ratio of each province

ax1.title.set_text('Academy Ratio of Each Province')

sns.barplot(data=region_overview, x='province', y='academy_ratio', ax=ax1)

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')



region_overview = region_overview.sort_values('elderly_population_ratio', 

                                              ascending=False).reset_index(drop=True)



# Elderly population ratio of each province

ax2.title.set_text('Elderly Population Ratio of Each Province')

sns.barplot(data=region_overview, x='province', y='elderly_population_ratio', ax=ax2)

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')



plt.show()
# DataFrame only with province and population value

pop_dist_prov = pop_dist.copy(deep=True)

pop_dist_prov['total'] = pop_dist_prov['total'].str.replace(',', '').astype(int)

pop_dist_prov = pop_dist_prov.sort_values('total', ascending=False).reset_index(

    drop=True).drop(pop_dist_prov.columns[2:13], axis=1)

by_i_case = pop_dist_prov.loc[[7, 6, 1, 2, 8, 3, 4, 5, 17, 11, 15], :]

by_i_case['location'] = ['Daegu','Gyeongsangbuk-do','Gyeonggi-do','Seoul',

                 'Chungcheongnam-do','Busan','Gyeongsangnam-do','Incheon',

                 'Sejong','Chungcheongbuk-do','Ulsan']



# Province population ordered by infection cases

plt.figure(figsize=(10, 5))

plt.title('Province Population (Order by infection cases)')

sns.barplot(data=by_i_case, x='location', y='total')

plt.xticks(rotation=30, ha='right')

plt.show()
# Importing data: Time

time = pd.read_csv(os.path.join(DIR_PATH, 'Time.csv'))

time.head()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Number of tests conducted

ax1.title.set_text('Total COVID-19 Tests Conducted')

sns.lineplot(data=time, x='date', y='test', label='total', ax=ax1)

sns.lineplot(data=time, x='date', y='confirmed', color='red', label='positive', ax=ax1)

sns.lineplot(data=time, x='date', y='negative', color='green', label='negative', ax=ax1)



ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

ax1.set(ylabel='count')

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Positive & Released & Deceased cases

ax2.title.set_text('Patient Count')

sns.lineplot(data=time, x='date', y='confirmed', label='positive', ax=ax2)

sns.lineplot(data=time, x='date', y='released', label='released', ax=ax2)

sns.lineplot(data=time, x='date', y='deceased', label='deceased', ax=ax2)



ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

ax2.set(ylabel='count')

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')



# Draw vertical line in patient count graph

ax2.axvline('2020-03-10', 0, 10000, color='red', linestyle='dotted')



plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

time['p_growth_rate'] = time['confirmed'].pct_change()

time['n_growth_rate'] = time['negative'].pct_change()



# Growth rate of positive cases

ax1.set_title('Positive Case Growth Rate')

sns.lineplot(data=time, x='date', y='p_growth_rate', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')



# Growth rate of negative cases

ax2.set_title('Negative Case Growth Rate')

sns.lineplot(data=time, x='date', y='n_growth_rate', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')



fig.show()
# Proportion by total population

time_f = time.tail(1)

time_latestT = time_f.test.values[0]

time_latestP = time_f.confirmed.values[0]

time_latestN = time_f.negative.values[0]

pop_total = int(pop_dist.iat[0, 1].replace(',', ''))



print('Percentage of people tested out of total population: {0}%\n'.format(round(time_latestT / pop_total * 100, 2)) + 

      'Percentage of positive cases out of people tested: {0}%\n'.format(round(time_latestP / time_latestT * 100, 2)) + 

      'Percentage of negative cases out of people tested: {0}%'.format(round(time_latestN / time_latestT * 100, 2)))
# Importing data: Search Trend

searchtrend = pd.read_csv(os.path.join(DIR_PATH, 'SearchTrend.csv'))

searchtrend.head()
searchTrend_2020 = searchtrend.iloc[1461:, :]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))



# Search proportion of keywords related to COVID-19

ax1.title.set_text('Keyword Search Trend')

for keyword in searchTrend_2020.iloc[:, 1:].columns:

    sns.lineplot(data=searchTrend_2020, x='date', y=keyword, label=keyword, ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

ax1.set(ylabel='percentage')

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')



# Search proportion of keywords related to COVID-19 except coronavirus

ax2.title.set_text('Keyword Search Trend (excluding "coronavirus")')

sns.lineplot(data=searchTrend_2020, x='date', y='cold', label='cold', ax=ax2)

sns.lineplot(data=searchTrend_2020, x='date', y='flu', label='flu', ax=ax2)

sns.lineplot(data=searchTrend_2020, x='date', y='pneumonia', label='pneumonia', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))

ax2.set(ylabel='percentage')

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')



fig.show()
# Importing data: Patient info

patientinfo = pd.read_csv(os.path.join(DIR_PATH, 'PatientInfo.csv'))

p_total = len(patientinfo)

print('People mainly got infected by {0} ways'.format(len(patientinfo['infection_case'].unique())) + 

      ' and had {0} contacts per person in average.'.format(round(patientinfo['contact_number'].mean(), 2)))

print('There are {0} patient data in this set.'.format(p_total))



# Convert to Int64 to remove decimals and leave NaN

patientinfo['infected_by'] = patientinfo['infected_by'].astype('Int64')



# Show transpose of a matrix for better visualization

patientinfo.head().T
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))



# Where/How confirmed patients got infected

ax1.title.set_text('Route of Infection')

sns.countplot(data=patientinfo, y='infection_case', 

              order=patientinfo['infection_case'].value_counts().index, ax=ax1)

show_values_on_bars(ax1, 'h', 10, 0.25)



# Infection order of a patient

ax2.title.set_text('Infection Order of a Patient (excluding NaN)')

sns.countplot(data=patientinfo, x='infection_order',

              order=patientinfo['infection_order'].value_counts().index, ax=ax2)

show_values_on_bars(ax2, 'v', modv=0.2)



plt.show()
transmit_order = patientinfo['infected_by'].value_counts().iloc[:10].index



# Top 10 patients who transmitted COVID-19 to others

fig, ax1 = plt.subplots(figsize=(10, 5))

plt.title('Top 10 patients who transmitted COVID-19')

sns.countplot(data=patientinfo, x='infected_by', order=transmit_order, ax=ax1)

plt.xticks(rotation=30, ha='right')

fig.show()
# Information of top 10 COVID-19 carriers

transmit_order_df = patientinfo.loc[patientinfo['patient_id'].isin(transmit_order)]

transmit_order_df.T
# Days took to release prior positive patients (Exclude NaN values)

patientinfo_release = pd.DataFrame()

patientinfo_release['c_date'] = pd.to_datetime(patientinfo['confirmed_date'], format='%Y-%m-%d')

patientinfo_release['r_date'] = pd.to_datetime(patientinfo['released_date'], format='%Y-%m-%d')

patientinfo_release['days_took'] = (patientinfo_release['r_date']

                                    - patientinfo_release['c_date']).dt.days.astype('Int64')

patientinfo_release = patientinfo_release.dropna()



plt.figure(figsize=(10, 3))

plt.title('Days took to get released')

sns.boxplot(x=patientinfo_release['days_took'])

# sns.swarmplot(x=patientinfo_release['days_took'], color='.25')

plt.show()
p_nosymp = patientinfo['symptom_onset_date'].isna().sum()



# Proportion of patients with/without symptom

plt.figure(figsize=(7, 7))

plt.title('Patients with Symptom')

plt.pie([p_total - p_nosymp, p_nosymp], labels=[f'Yes ({p_total - p_nosymp})', f'No ({p_nosymp})'], 

        autopct='%.1f%%', pctdistance=0.85)

plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))

plt.show()
# Importing data: Patient route

patientroute = pd.read_csv(os.path.join(DIR_PATH, 'PatientRoute.csv'))

patientroute.head()
patientroute_top_log = pd.DataFrame(patientroute['patient_id'].value_counts().head(10))

print('There are {0} patients\' route data.'.format(len(patientroute['patient_id'].unique())))

patientroute_top_place = pd.DataFrame(patientroute['type'].value_counts().head(10))

patientroute_top_place.reset_index(level=0, inplace=True)

patientroute_top_place.columns = ['type', 'count']

patientroute_top_place
fig, ax1 = plt.subplots(figsize=(10, 5))



ax1.title.set_text('Top 10 Places COVID-19 Patients Visited')

sns.barplot(data=patientroute_top_place, x='type', y='count', ax=ax1)

show_values_on_bars(ax1, 'v', modv=20)

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.show()
# Route of top 10 patients who spread COVID-19

patient_routes = []



for i in range(len(transmit_order)):

    a = []

    tmp_route = patientroute.loc[patientroute['patient_id'] == 

                                 transmit_order[i]].reset_index(drop=True)

    for j in range(len(tmp_route)):

        a.append(tuple([tmp_route.loc[j].latitude, tmp_route.loc[j].longitude]))

    patient_routes.append(a)



print('Saved in \'patient_routes\'')
## Going to fix... please leave a comment if you know what's causing this problem



route_southKR = folium.Map(location=[36.5, 128], tiles="cartodbpositron",

                         zoom_start=8, min_zoom=5)

folium.Choropleth(geo_data='/kaggle/input/geolocation-population-distribution-of-south-kr/province_geo.json',

                  fill_color='#ffff66', line_opacity=0.5, fill_opacity=0.3).add_to(route_southKR)



for i in range(len(patient_routes)):

    for places in patient_routes[i]:

        folium.Marker(places).add_to(route_southKR)

    ran_c = list(np.random.choice(range(256), size=3))

    folium.PolyLine(patient_routes[i], color='#%02x%02x%02x' % (ran_c[0], ran_c[1], ran_c[2])).add_to(route_southKR)



route_southKR