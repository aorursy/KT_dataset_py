# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# * Which hours of the day do the most accidents occur during?
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
# change 0 as 24
accidents_by_hour["f1_"][accidents_by_hour["f1_"] == 0] = 24

fig, ax = plt.subplots()
bar = ax.bar(accidents_by_hour["f1_"], accidents_by_hour["f0_"], color = 'grey')

ax.set_xlabel('Hour of the day [1-24]')
ax.set_ylabel('Fatalities count')
ax.set_xticks(accidents_by_hour["f1_"])

#highlight the time with most fatalities
bar[0].set_color('r')

plt.show()
# * Which state has the most hit and runs?
#    * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
# Instead of registration_state_name (where the vehicle involved is from) I am using here the state where the event happened
query = """SELECT COUNT(hit_and_run) as count, 
                  state_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY state_number
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
hit_and_run_by_state
# States code were obtained from http://www.trb.org/Main/Blurbs/173591.aspx
state_code = [(1, "Alabama"),
(2, "Alaska"),
(4, "Arizona"),
(5, "Arkansas"),
(6, "California"),
(8, "Colorado"),
(9, "Connecticut"),
(10, "Delaware"),
(11, "District of Columbia"),
(12, "Florida"),
(13, "Georgia"),
(15, "Hawaii"),
(16, "Idaho"),
(17, "Illinois"),
(18, "Indiana"),
(19, "Iowa"),
(20, "Kansas"),
(21, "Kentucky"),
(22, "Louisiana"),
(23, "Maine"),
(24, "Maryland"),
(25, "Massachusetts"),
(26, "Michigan"),
(27, "Minnesota"),
(28, "Mississippi"),
(29, "Missouri"),
(30, "Montana"),
(31, "Nebraska"),
(32, "Nevada"),
(33, "New Hampshire"),
(34, "New Jersey"),
(35, "New Mexico"),
(36, "New York"),
(37, "North Carolina"),
(38, "North Dakota"),
(39, "Ohio"),
(40, "Oklahoma"),
(41, "Oregon"),
(42, "Pennsylvania"),
(43, "Puerto Rico"),
(44, "Rhode Island"),
(45, "South Carolina"),
(46, "South Dakota"),
(47, "Tennessee"),
(48, "Texas"),
(49, "Utah"),
(50, "Vermont"),
(52, "Virgin Islands"), 
(51, "Virginia"),
(53, "Washington"),
(54, "West Virginia"),
(55, "Wisconsin"),
(56, "Wyoming")]
state_code_number, state_code_name = zip(*state_code)
state_code_number, state_code_name
# add column for state name
new_col = [state_code_name[state_code_number.index(i)] for i in hit_and_run_by_state["state_number"].values]
hit_and_run_by_state = hit_and_run_by_state.assign(state_name = new_col)
# normalize count of hit and run
new_col = hit_and_run_by_state["count"].values/max(hit_and_run_by_state["count"].values)
hit_and_run_by_state = hit_and_run_by_state.assign(count_normalized = new_col)
#thanks: https://stackoverflow.com/questions/7586384/color-states-with-pythons-matplotlib-basemap
#https://www.kaggle.com/mknawara/regression-challenge-day-5
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))

map = Basemap(projection='cyl', 
            lat_0=46.2374,
            lon_0=2.375,
            resolution='h',
            llcrnrlon=-130, llcrnrlat=24,
            urcrnrlon=-62, urcrnrlat=51)

map.readshapefile('../input/cb_2016_us_state_500k', name='states', drawbounds=False)

map.drawcoastlines()
map.drawcountries(linewidth=2)
map.drawmapboundary()
map.drawstates()

ax = plt.gca()

# color each state based on the count of hit and run
for state_name in hit_and_run_by_state['state_name']:
    for idx, info in enumerate(map.states_info):
        if info['NAME'] == state_name:
            val = hit_and_run_by_state[hit_and_run_by_state['state_name'] == state_name]['count_normalized'].values
            # The color map is bounded between 0 and 1, as count_normalized
            col = cm.gray_r(val[0])[:3] # only rgb and not alpha value
            poly = Polygon(map.states[idx], facecolor=col, edgecolor=None)
            ax.add_patch(poly)
            
plt.show()
#registered automobiles from https://www.fhwa.dot.gov/policyinformation/statistics/2010/mv1.cfm
registered_vehicles = [('Alabama', 2211550), 
('Alaska', 228407), 
('Arizona', 2201251), 
('Arkansas', 945198), 
('California', 17977605), 
('Colorado', 1890748), 
('Connecticut', 1985500), 
('Delaware', 434037), 
('District of Columbia', 160090), 
('Florida', 7295121), 
('Georgia', 3738952), 
('Hawaii', 450398), 
('Idaho', 541038), 
('Illinois', 5772947), 
('Indiana' , 2986033), 
('Iowa', 1691090), 
('Kansas', 880308), 
('Kentucky', 1890079), 
('Louisiana', 1917283), 
('Maine', 518779), 
('Maryland', 2590777), 
('Massachusetts', 3144691), 
('Michigan', 5135712), 
('Minnesota', 2459074), 
('Mississippi', 1143527), 
('Missouri', 2578536), 
('Montana' , 351574), 
('Nebraska', 773080), 
('Nevada', 690124), 
('New Hampshire', 618598), 
('New Jersey', 3971896), 
('New Mexico', 702897), 
('New York' , 7950192), 
('North Carolina', 3281831), 
('North Dakota', 340756), 
('Ohio', 5614698), 
('Oklahoma', 1581768), 
('Oregon', 1488595), 
('Pennsylvania', 5682239), 
('Rhode Island', 478624), 
('South Carolina', 2030632), 
('South Dakota', 406531), 
('Tennessee', 2734382), 
('Texas', 8331127), 
('Utah', 1316966), 
('Vermont', 293084), 
('Virginia', 3510417), 
('Virgin Islands', 0),
('Washington', 2599791), 
('West Virginia', 702587), 
('Wisconsin', 2461343), 
('Wyoming', 209777), 
('Puerto Rico', 2421055)] 
registered_vehicles_states, registered_vehicles_count = zip(*registered_vehicles)
registered_vehicles_states = list(registered_vehicles_states)
sorted(registered_vehicles_states) == sorted(state_code_name)
# add column for state name
new_col = [registered_vehicles_count[state_code_number.index(i)] for i in hit_and_run_by_state["state_number"].values]
hit_and_run_by_state = hit_and_run_by_state.assign(registered_autombiles = new_col)
hit_and_run_by_state.head()
# normalize count of hit and run
new_col = hit_and_run_by_state["count"].values/hit_and_run_by_state["registered_autombiles"].values
new_col = new_col/max(new_col)
hit_and_run_by_state = hit_and_run_by_state.assign(count_normalized_autos = new_col)
hit_and_run_by_state.head()
plt.figure(figsize=(20,15))

map = Basemap(projection='cyl', 
            lat_0=46.2374,
            lon_0=2.375,
            resolution='h',
            llcrnrlon=-130, llcrnrlat=24,
            urcrnrlon=-62, urcrnrlat=51)

map.readshapefile('../input/cb_2016_us_state_500k', name='states', drawbounds=False)

map.drawcoastlines()
map.drawcountries(linewidth=2)
map.drawmapboundary()
map.drawstates()

ax = plt.gca()

# color each state based on the count of hit and run
for state_name in hit_and_run_by_state['state_name']:
    for idx, info in enumerate(map.states_info):
        if info['NAME'] == state_name:
            val = hit_and_run_by_state[hit_and_run_by_state['state_name'] == state_name]['count_normalized_autos'].values
            # The color map is bounded between 0 and 1, as count_normalized
            col = cm.gray_r(val[0])[:3] # only rgb and not alpha value
            poly = Polygon(map.states[idx], facecolor=col, edgecolor=None)
            ax.add_patch(poly)

plt.show()