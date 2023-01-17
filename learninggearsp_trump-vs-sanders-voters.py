"""
In this script, I look for county_facts features that distinguish voters of two candidates, by state. For example,
what are the states and features where the voters of each camp show opposite behavior. Does Trump really appeal
to the poor or wealthy or any race, more than Clinton ? ..

For that, I run a  Mann Whitney Test, that asses the H0 hypothesis: Do the distributions have the same median ?
(non parametric version of ANOVA for two variables..), Then, I plot fitted gaussians to the distributions of
the found features.

I would appreciate your feedback on my code and my approach!
"""



import pandas as pd
import numpy as np

from scipy import stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 6, 6

candidate1 = 'Bernie Sanders'
candidate2 = 'Donald Trump'

###########
# load data

df_county_facts = pd.read_csv('../input/county_facts.csv')

dict_county_facts = pd.read_csv('../input/county_facts_dictionary.csv')
dict_county_facts = {k: v for k, v in zip(dict_county_facts['column_name'].values, dict_county_facts['description'].values)}

df_prim_res = pd.read_csv('../input/primary_results.csv')

cols_to_drop = ['state', 'state_abbreviation_x', 'county', 'fips', 'party', 'candidate',
				'votes', 'fraction_votes', 'area_name', 'state_abbreviation']

cols = set(df_county_facts).difference(cols_to_drop)


# function to extract column values for a candidate 
# in the counties where they had high voting (over 1000)

def get_col(df_prim_res, df_county_facts, column, candidate):

	df_candidate = df_prim_res[df_prim_res['candidate'] == candidate]

	df_all = df_candidate.merge(df_county_facts, on=['fips'])

	df_all = df_all[df_all['votes'] > 100]

	out = df_all.sort(column)[column][int(0.05 * df_all.shape[0]) : int(0.95 * df_all.shape[0])]

	return out

#########	
# U TESTS

# iterate over columns and states, and store test's results in dict when H0 is rejected

res = {}
for column in cols:

	for sa in df_county_facts['state_abbreviation'].unique():

		df_state = df_county_facts[df_county_facts['state_abbreviation'] == sa]

		col_cand1 = get_col(df_prim_res, df_state, column, candidate1)
		col_cand2 = get_col(df_prim_res, df_state, column, candidate2)

        # considering only big enough arrays with close lengths
		is_ok = (len(col_cand1) > 50) and (len(col_cand2) > 50) and \
				(abs(len(col_cand1) - len(col_cand2)) < min(len(col_cand1), len(col_cand2)))

		if is_ok:

			try:
				mw_stat, p_val = stats.mannwhitneyu(col_cand1, col_cand2, use_continuity=False)

				if p_val < 0.05: # reject H0
					res[(sa, column)] = {'col_cand1': col_cand1.values,
										'col_cand2': col_cand2.values,
										'median_diff': abs(np.median(col_cand1) - np.median(col_cand2)) \
														/ (np.max(col_cand1) - np.median(col_cand1))}

			except:
				continue


#######
# PLOTS

sorted_res = sorted(res.items(), key=lambda k: k[1]['median_diff'], reverse=True)


for ii, (key, val) in enumerate(sorted_res[:7]):

	sa = key[0]
	column = key[1]

	col_cand1 = val['col_cand1']
	col_cand2 = val['col_cand2']
	med_diff = val['median_diff']

	print("{} - {} - {} - {}".format(sa, dict_county_facts[column], p_val, med_diff))

	fig = plt.figure()

	n, bins, patches = plt.hist(col_cand1, color='k', label=candidate1, bins=30, normed=1, histtype='step')
	plt.hold(False)
	
    # fit a normal distirbution to the histogram
	
	(mu, sigma) = stats.norm.fit(col_cand1, )
	y = mlab.normpdf( bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth=2, label=candidate1)
	plt.hold(True)

	(mu, sigma) = stats.norm.fit(col_cand2)
	y = mlab.normpdf( bins, mu, sigma)
	l = plt.plot(bins, y, 'k--', linewidth=2, label=candidate2)
	plt.hold(True)

	plt.legend()

	plt.xlabel(dict_county_facts[column])
	
	plt.title(sa + '_' + dict_county_facts[column])
	plt.grid()

	fig.savefig("{}_{}.png".format(ii, sa))



