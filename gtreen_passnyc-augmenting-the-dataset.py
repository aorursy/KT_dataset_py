# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# function for printing out top schools for a given parameter

def print_top_N_schools (data, target_key, N, order='Top'):
    if (target_key in data.keys()):
        print ('%s schools for parameter %s:' % (order, target_key))
        d = np.array(data[target_key])
        for i in range(N):
            if (order=='Top'):
                index = np.argmax(d)
                value = np.max(d)
                d[index] = 0.0
            else:
                index = np.argmin(d)
                value = np.min(d)
                d[index] = 1000000.0
            print ('%d. %s, value = %f' % ((i+1), data['School Name'][index], value))




schools_data    = pd.read_csv ('../input/nyc-2016-school-explorer-refined/nyc_school_explorer_refined.csv')

# Any results you write to the current directory are saved as output.
zip_code_density_data = pd.read_csv ('../input/us-population-density-by-zip-code-2010/Zipcode-ZCTA-Population-Density-And-Area-Unsorted.csv')


def get_density (zip_code):
    for i in range(len(zip_code_density_data)):
        if (zip_code_density_data['Zip/ZCTA'][i]==zip_code):
            return zip_code_density_data['Density Per Sq Mile'][i]

densities = []
for i in range(len(schools_data)):
    zipcode = schools_data['Zip'][i]
    densities.append (get_density(zipcode))

# create a new schools data parameter: Zip Density
    
schools_data['Zip Density'] = densities

print_top_N_schools (schools_data, 'Zip Density', 10, 'Top')
print ('')
print_top_N_schools (schools_data, 'Zip Density', 10, 'Bottom')




school_district_breakdowns = pd.read_csv ('../input/nyc-school-district-breakdowns/school-district-breakdowns.csv')


# get the district breakdown data

district_asian_pct = []
district_black_pct = []
district_hispanic_pct = []
district_white_pct = []
district_public_assistance_pct = []
district_us_citizen_pct = []
district_permanent_res_alien_pct = []
for i in range(len(schools_data)):
    district  = schools_data['District'][i]
    district_asian_pct.append(school_district_breakdowns['PERCENT ASIAN NON HISPANIC'][district-1])
    district_black_pct.append(school_district_breakdowns['PERCENT BLACK NON HISPANIC'][district-1])
    district_hispanic_pct.append(school_district_breakdowns['PERCENT HISPANIC LATINO'][district-1])
    district_white_pct.append(school_district_breakdowns['PERCENT WHITE NON HISPANIC'][district-1])
    district_public_assistance_pct.append(school_district_breakdowns['PERCENT RECEIVES PUBLIC ASSISTANCE'][district-1])
    district_us_citizen_pct.append(school_district_breakdowns['PERCENT US CITIZEN'][district-1])
    district_permanent_res_alien_pct.append(school_district_breakdowns['PERCENT PERMANENT RESIDENT ALIEN'][district-1])

# add to the schools dataset

schools_data['District Asian %'] = district_asian_pct
schools_data['District Black %'] = district_black_pct
schools_data['District Hispanic %'] = district_hispanic_pct
schools_data['District White %'] = district_white_pct
schools_data['District Public Assistance %'] = district_public_assistance_pct
schools_data['District U.S. Citizen %'] = district_us_citizen_pct
schools_data['District Permanent Resident Alien %'] = district_permanent_res_alien_pct






#collisions_data = pd.read_csv ('../input/nypd-motor-vehicle-collisions/nypd-motor-vehicle-collisions.csv')

#def get_distance_in_km (ll1, ll2):
#   # approximate radius of earth in km
#    R = 6373.0
#    lat1 = radians(ll1[0])
#    lon1 = radians(ll1[1])
#    lat2 = radians(ll2[0])
#    lon2 = radians(ll2[1])
#    dlon = lon2 - lon1
#    dlat = lat2 - lat1
#    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#    c = 2 * atan2(sqrt(a), sqrt(1 - a))
#    distance = R * c
#    return distance

## find the number of collisions reported within the vicinity of a school

#collision_dist_thresh = 0.5
#collisions_count = []
#for i in range(len(schools_data)):
#    nearby_collisions = 0
#    print (time.ctime())
#    print ('School: %s:' % schools_data['School Name'][i])
#    for j in range(len(collisions_data)):
#        dist = get_distance_in_km ((schools_data['Latitude'][i], schools_data['Longitude'][i]),
#                                   (collisions_data['LATITUDE'][j], collisions_data['LONGITUDE'][j]))
#        if (dist < collision_dist_thresh):
#            nearby_collisions+=1
#    print ('Found %d nearby collisions.' % nearby_collisions)
#    collisions_count.append(nearby_collisions)
#    np.save ('collisions_count', collisions_count)
    
#school_df = pd.DataFrame(schools_data['School Name'])
#school_df['Nearby Auto Collisions 2012-18'] = collisions_count
#school_df.to_csv('nyc_middle_schools_collisions.csv', index=False)

schools_collisions_data = pd.read_csv ('../input/nyc-middle-schools-and-nearby-auto-collisions/nyc_middle_schools_collisions_2012_2018.csv')


schools_data['Nearby Auto Collisions'] = schools_collisions_data['Nearby Auto Collisions 2012-18']


# plot the collisions data on a map for visualization

import folium
from folium import plugins
from io import StringIO
import folium 


collisions = schools_data['Nearby Auto Collisions']
map_data = collisions

interval = (max(map_data)-min(map_data))/255.0
red_val = ((map_data-min(map_data))/interval).astype('int')

m = folium.Map([schools_data['Latitude'][0], schools_data['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')

#for lat, long, col in zip(schools_data['Latitude'], schools_data['Longitude'], cols):
for lat, long, red in zip(schools_data['Latitude'], schools_data['Longitude'], red_val):
    #rown = list(rown)
    #folium.CircleMarker([lat, long], color='#0000ff', fill=True, radius=2).add_to(m)
    colourString = '#%0.2x00%0.2x' % (red, (255-red))
    folium.CircleMarker([lat, long], color=colourString, fill=True, radius=2).add_to(m)


m


grade6_applications_offers = pd.read_csv ('../input/nyc-middle-schools-grade-6-applicationsoffers/nyc_middle_school_grade6_offers.csv')

# set the school grade 6 acceptance rate

acceptance_rates=[]
not_found_cnt = 0
not_found_charter_cnt = 0

for i in range(len(schools_data)):
    acc_rate = 0.0
    for j in range(len(grade6_applications_offers)):
        if (schools_data['Location Code'][i]==grade6_applications_offers['DBN'][j]):
            acc_rate = float(grade6_applications_offers['Offers'][j]) / (
                       float(grade6_applications_offers['Applications'][j]))
            break
    if (acc_rate==0.0):
        not_found_cnt += 1
        if ('CHARTER' in schools_data['School Name'][i]):
            not_found_charter_cnt += 1
        print ('Warning: could not find applications and offers data for school %s.' % schools_data['School Name'][i])
    acceptance_rates.append(acc_rate)

print ('Could not find application/offer data for %d schools (%d charter schools).' % (not_found_cnt, not_found_charter_cnt))





# set the charter school acceptance rate to 1.0

for i in range(len(schools_data)):
    if (acceptance_rates[i]==0.0):
        if ('CHARTER' in schools_data['School Name'][i]):
            acceptance_rates[i] = 1.0

# remove the schools for which we have no acceptance rate data


acceptance_rates = np.array(acceptance_rates)
valid_acceptance_rates = acceptance_rates[acceptance_rates>0.0]
schools_data = schools_data[acceptance_rates>0.0]

# re-construct the DataFrame
schools_data = pd.DataFrame(data=np.array(schools_data),columns=schools_data.keys())
schools_data['Grade 6 Acceptance Rate'] = valid_acceptance_rates


print_top_N_schools (schools_data, 'Grade 6 Acceptance Rate', 10, 'Lowest')

from scipy.stats.stats import pearsonr


admin_index = 10
X=np.array(schools_data)[:,admin_index:]
X_keys=schools_data.keys()[admin_index:]



def print_top_N_correlations (keys, data, target_key, N, order='Highest'):
    coefficients = []
    target_index = -1
    for i in range(len(keys)):
        if (keys[i] == target_key):
            target_index=i
            break
    if (target_index<0):
        print ('Could not find key %s in key list.' % target_key)
        return
    for i in range(len(keys)):
        coeff = pearsonr (data[:,i], data[:,target_index])[0]
        coefficients .append (coeff)
    print ('%s Pearson\'s correlation with %s:' % (order, target_key))
    c = np.array(coefficients)
    for i in range(N):
        if (order=='Highest'):
            index = np.argmax(c)
        else:
            index = np.argmin(c)
        print ('%d. Key: %s, correlation = %f' % ((i+1), keys[index], c[index]))
        c[index] = 0.0




print_top_N_correlations (X_keys, X, 'Zip Density', 10, 'Highest')



print_top_N_correlations (X_keys, X, 'Grade 6 Acceptance Rate', 10, 'Lowest')
schools_data.to_csv('nyc_augmented_school_explorer.csv', index=False)
