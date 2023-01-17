# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sf_file = pd.read_csv("../input/Restaurant_Scores_-_LIVES_Standard.csv")
print (sf_file.describe())
sf_rev = sf_file.drop(['business_address', 'business_city', 'business_state',

	'business_postal_code', 'business_location', 'business_phone_number'], axis=1)

sf_rev['inspection_date'] = pd.to_datetime(sf_rev['inspection_date'], 

                                           format='%m/%d/%Y %H:%M:%S %p')

sf_rev['year'], sf_rev['month'] = sf_rev['inspection_date'].apply(lambda x: x.year), sf_rev['inspection_date'].apply(lambda x:x.month)
print (sf_rev.describe())

print ('--------------------------------')

print (sf_rev.head())
sf_rev['risk_category'] = sf_rev['risk_category'].fillna(value="No Risk")

sf_rev_dum = pd.get_dummies(sf_rev['risk_category'], prefix='Risk')

sf_rev_grouped = pd.concat([sf_rev, sf_rev_dum], axis = 1, join='inner')



sf_rev_grouped = sf_rev_grouped.loc[:,['business_id','inspection_id','year', 'month',

	'Risk_High Risk','Risk_Moderate Risk','Risk_Low Risk','Risk_No Risk']]
sf_rev_grouped = sf_rev_grouped.groupby(['business_id', 'inspection_id', 'year', 'month'], 

                                        as_index=False).sum()

col_names = ['business_id','inspection_id','year', 'month','Tot_High_Risk',

             'Tot_Moderate_Risk','Tot_Low_Risk', 'Tot_No_Risk']

sf_rev_grouped.columns = col_names



sf_rev_grouped['Max_Risk'] = 'Low'

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_Moderate_Risk'] != 0, 

                                      'Moderate', sf_rev_grouped['Max_Risk'])

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_High_Risk'] != 0, 

                                      'High', sf_rev_grouped['Max_Risk'])

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_No_Risk'] == 1, 

                                      'No', sf_rev_grouped['Max_Risk'])
sf_rev_grouped2 = sf_rev_grouped.loc[:,['business_id','Max_Risk']]

sf_rev_dum_2 = pd.get_dummies(sf_rev_grouped2['Max_Risk'], prefix='Risk')

sf_rev_grouped2 = pd.concat([sf_rev_grouped2, sf_rev_dum_2], axis = 1, join='inner')



sf_rev_grouped2 = sf_rev_grouped2.loc[:,['business_id','Risk_High','Risk_Moderate',

                                         'Risk_Low','Risk_No']]

sf_rev_grouped2 = sf_rev_grouped2.groupby(['business_id'], as_index=False).sum()

col_names = ['business_id','Tot_High_Risk','Tot_Moderate_Risk','Tot_Low_Risk', 'Tot_No_Risk']

sf_rev_grouped2.columns = col_names



sf_rev_grouped2['Max_Risk'] = 'No'

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_Low_Risk'] != 0, 

                                       'Low', sf_rev_grouped2['Max_Risk'])

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_Moderate_Risk'] != 0, 

                                       'Moderate', sf_rev_grouped2['Max_Risk'])

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_High_Risk'] != 0, 

                                       'High', sf_rev_grouped2['Max_Risk'])
len_bus = len(sf_rev['business_id'].unique())

len_ins = len(sf_rev_grouped['inspection_id'].unique())

len_nolat = len(sf_rev[sf_rev.business_latitude.isnull()]['business_id'].unique())

len_lat = len(sf_rev[sf_rev.business_latitude.isnull() == False]['business_id'].unique())

len_ins2 = len(sf_rev_grouped['inspection_id'].unique())

len_hi = len(sf_rev_grouped2[sf_rev_grouped2.Max_Risk == 'High'])

len_mod = len(sf_rev_grouped2[sf_rev_grouped2.Max_Risk == 'Moderate'])

len_low = len(sf_rev_grouped2[sf_rev_grouped2.Max_Risk == 'Low'])

len_no = len(sf_rev_grouped2[sf_rev_grouped2.Max_Risk == 'No'])



print ("Numer of inspected business", len_bus)

print ("Number of inspected business w/ location data", len_lat)

print ("Number of inspected business w/o location data", len_nolat)

print ("Total business w/ and w/o location", len_lat + len_nolat)

print ("Total number of business w/ worst High risk", len_hi)

print ("Total number of business w/ worst Moderate risk", len_mod)

print ("Total number of business w/ worst Low risk", len_low)

print ("Total number of business w/ No historic risk", len_no)

print ("Total number of business by risk", 

       len_hi + len_mod + len_low + len_no)

print ("Total number of business from DF", len(sf_rev_grouped2))
print ("Out of", len_bus, "business inspected between 2014 and 2016, only", len_no,

      "did not report observations")
sf_rev_grouped = pd.concat([sf_rev, sf_rev_dum], axis = 1, join='inner')



sf_rev_grouped = sf_rev_grouped.loc[:,['business_id','inspection_id','business_latitude',

                                       'business_longitude','year', 'month',

	'Risk_High Risk','Risk_Moderate Risk','Risk_Low Risk','Risk_No Risk']]

sf_rev_grouped = sf_rev_grouped.groupby(['business_id', 'inspection_id','business_latitude',

                                         'business_longitude', 'year', 'month'], as_index=False).sum()

col_names = ['business_id','inspection_id','business_latitude','business_longitude', 

             'year', 'month','Tot_High_Risk','Tot_Moderate_Risk','Tot_Low_Risk', 'Tot_No_Risk']

sf_rev_grouped.columns = col_names



sf_rev_grouped['Max_Risk'] = 'Low'

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_Moderate_Risk'] != 0, 

                                      'Moderate', sf_rev_grouped['Max_Risk'])

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_High_Risk'] != 0, 

                                      'High', sf_rev_grouped['Max_Risk'])

sf_rev_grouped['Max_Risk'] = np.where(sf_rev_grouped['Tot_No_Risk'] == 1, 

                                      'No', sf_rev_grouped['Max_Risk'])

sf_rev_grouped2 = sf_rev_grouped.loc[:,['business_id','business_latitude','business_longitude',

                                        'Max_Risk']]

sf_rev_dum = pd.get_dummies(sf_rev_grouped2['Max_Risk'], prefix='Risk')

sf_rev_grouped2 = pd.concat([sf_rev_grouped2, sf_rev_dum], axis = 1, join='inner')



sf_rev_grouped2 = sf_rev_grouped2.loc[:,['business_id','business_latitude','business_longitude',

                                         'Risk_High','Risk_Moderate','Risk_Low','Risk_No']]

sf_rev_grouped2 = sf_rev_grouped2.groupby(['business_id','business_latitude',

                                           'business_longitude'], as_index=False).sum()

col_names = ['business_id','business_latitude','business_longitude',

	'Tot_High_Risk','Tot_Moderate_Risk','Tot_Low_Risk', 'Tot_No_Risk']

sf_rev_grouped2.columns = col_names



sf_rev_grouped2['Max_Risk'] = 'No'

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_Low_Risk'] != 0, 

                                       'Low', sf_rev_grouped2['Max_Risk'])

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_Moderate_Risk'] != 0, 

                                       'Moderate', sf_rev_grouped2['Max_Risk'])

sf_rev_grouped2['Max_Risk'] = np.where(sf_rev_grouped2['Tot_High_Risk'] != 0, 

                                       'High', sf_rev_grouped2['Max_Risk'])

import matplotlib.pyplot as plt



colors = {'High' : 'green', 'Moderate': 'blue', 'Low': 'orange', 'No': 'red'}

sf_rev_grouped2['Colors'] ='None'



for i in range(0,len(sf_rev_grouped2)):

	sf_rev_grouped2.loc[i, 'Colors'] = colors[sf_rev_grouped2.loc[i, 'Max_Risk']]



fig, ax = plt.subplots()

ax.set_title('Worst rating in SF restaurants inspections, by restaurant')

plt.axis([-122.55, -122.37, 37.7, 37.82])



plt.scatter(sf_rev_grouped2['business_longitude'], sf_rev_grouped2['business_latitude'],

            marker='.', c=sf_rev_grouped2['Colors'])

plt.show()
fig, ax = plt.subplots()

ax.set_title('Worst rating in SF restaurants inspections, by restaurant')

plt.axis([-122.44, -122.39, 37.77, 37.8])



plt.scatter(sf_rev_grouped2['business_longitude'], sf_rev_grouped2['business_latitude'],

            marker='.', c=sf_rev_grouped2['Colors'])

plt.show()
fig, ax = plt.subplots()

ax.set_title('Worst rating in SF restaurants inspections, by restaurant')

plt.axis([-122.41, -122.395, 37.785, 37.8])



plt.scatter(sf_rev_grouped2['business_longitude'], sf_rev_grouped2['business_latitude'],

            marker='.', c=sf_rev_grouped2['Colors'])

plt.show()
import random

import math

from collections import Counter



def distance2(v, w):

	vector_subtract = [v_i - w_i for v_i, w_i in zip(v, w)]

	vector_2 = sum(v_i * v_i for v_i in vector_subtract)

	return math.sqrt(vector_2)



def majority_vote(labels):

	vote_counts = Counter(labels)

	winner, winner_count = vote_counts.most_common(1)[0]

	num_winners = len([count

						for count in vote_counts.values()

						if count == winner_count])

	if num_winners == 1:

		return winner

	else:

		return majority_vote(labels[:-1])



def knn_classify(k, labeled_points, new_point):

	by_distance = sorted(labeled_points, key=lambda pointx: distance2(pointx[0], new_point))

	k_nearest_labels = [label for _, label in by_distance[:k]]

	return majority_vote(k_nearest_labels)



test_sample = random.sample(range(0, len(sf_rev_grouped2)), 100)

sf_count = sf_rev_grouped2.loc[test_sample,['business_longitude', 'business_latitude', 

                                            'Max_Risk']]

sf_count = sf_count.reset_index()



for k in [1, 3, 5, 7]:

	num_correct = 0

	for i in range(0, len(sf_count)):

		location = (sf_count.loc[i,'business_longitude'], sf_count.loc[i, 'business_latitude'])

		risk = sf_count.loc[i, 'Max_Risk']

		other_locations = []

		for i in range(0, len(sf_count)):

			if (sf_count.loc[i,'business_longitude'], sf_count.loc[i, 'business_latitude']) != location:

				other_locations.append([(sf_count.loc[i,'business_longitude'], sf_count.loc[i, 'business_latitude']),

				 sf_count.loc[i, 'Max_Risk']])

			else:

				pass

		predicted_risk = knn_classify(k, other_locations, location)

		if predicted_risk == risk:

			num_correct += 1



	print (k, "neighbor[s]:", num_correct, "correct out of", len(sf_count))