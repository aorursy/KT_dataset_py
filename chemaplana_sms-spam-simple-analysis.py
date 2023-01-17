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
import pandas as pd

import numpy as np



sms_file_r = pd.read_csv("../input/spam.csv", encoding='latin-1')

sms_train = pd.DataFrame(sms_file_r.loc[0:5000,:])

print (sms_train.dtypes)
print (sms_train.head())
print (sms_train.describe())
print (sms_train['v1'].unique())
print (sms_train['Unnamed: 2'].unique())
print (sms_train['Unnamed: 3'].unique())
print (sms_train['Unnamed: 4'].unique())
sms_train['num_c'] = 0



for i in np.arange(0,len(sms_train)):

	sms_train.loc[i,'num_c'] = len(sms_train.loc[i,'v2'])



sms_ham = sms_train[sms_train.v1 == 'ham']

sms_spam = sms_train[sms_train.v1 == 'spam']



sms_ham_count = pd.DataFrame(pd.value_counts(sms_ham['num_c'],sort=True).sort_index())

sms_spam_count = pd.DataFrame(pd.value_counts(sms_spam['num_c'],sort=True).sort_index())



print (sms_ham_count.describe())

print (sms_spam_count.describe())
import matplotlib.pyplot as plt



ax = plt.axes()

ax.set_title('SMS Ham by length of message')

xline = np.linspace(0, len(sms_ham_count) - 1, len(sms_ham_count))

width = 0.50



bar_1 = ax.bar(xline, sms_ham_count['num_c'], width=width, color='r')

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()
ax = plt.axes()

ax.set_title('SMS Spam by length of message')

xline = np.linspace(0, len(sms_spam_count) - 1, len(sms_spam_count))

width = 0.50



bar_2 = ax.bar(xline, sms_spam_count['num_c'], width=width, color='b')

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()
from scipy.stats import gaussian_kde



def sms_predict(low,high,low_test,high_test):

	kde = gaussian_kde

	ax = plt.axes()

	sms_train = pd.DataFrame(sms_file_r.loc[low:high,:])



	sms_train['num_c'] = 0



	for i in np.arange(0,len(sms_train)):

		sms_train.loc[i,'num_c'] = len(sms_train.loc[i,'v2'])



	density_h = kde(sms_train[sms_train.v1 == 'ham'].loc[:,'num_c'])

	density_s = kde(sms_train[sms_train.v1 == 'spam'].loc[:,'num_c'])



	density_h.covariance_factor = lambda : 0.5

	density_h._compute_covariance()

	density_s.covariance_factor = lambda : 0.5

	density_s._compute_covariance()



	xs = np.linspace(0, 200, 201)

	cntr_h_l = 0

	cntr_h_h = 0

	cdf_h = 0



	for i in xs:

		cntr_h_l = i

		cdf_h += density_h(cntr_h_l)

		if cdf_h >= 0.25:

			break



	print (cntr_h_l, cdf_h)



	for i in xs:

		cntr_h_h = i

		cdf_h += density_h(cntr_h_h)

		if cdf_h >= 0.75:

			break



	print (cntr_h_h, cdf_h)



	cntr_s_l = 0

	cntr_s_h = 0

	cdf_s = 0



	for i in xs:

		cntr_s_l = i

		cdf_s += density_s(cntr_s_l)

		if cdf_s >= 0.25:

			break



	print (cntr_s_l, cdf_s)



	for i in xs:

		cntr_s_h = i

		cdf_s += density_s(cntr_s_h)

		if cdf_s >= 0.75:

			break



	print (cntr_s_h, cdf_s)



	den_h, = plt.plot(xs,density_h(xs), color='red')

	den_s, = plt.plot(xs,density_s(xs), color='blue')

	plt.axvline(cntr_h_l, color = 'k')

	plt.axvline(cntr_h_h, color = 'k')

	plt.axvline(cntr_s_l, color = 'k')

	plt.axvline(cntr_s_h, color = 'k')

	plt.xlabel('message length')

	plt.ylabel('density')

	plt.legend([den_h, den_s], ['Density ham', 'Density spam'], loc='upper left')

	plt.show()



	sms_test = pd.DataFrame(sms_file_r.loc[low_test:high_test,:])

	sms_test.reset_index(drop=True, inplace=True)



	sms_test['num_c'] = 0



	for i in np.arange(0,len(sms_test)):

		sms_test.loc[i,'num_c'] = len(sms_test.loc[i,'v2'])



	sms_test['pred'] = ''

	sms_test['eval'] = 0



	for i in np.arange(0,len(sms_test)):

		if sms_test.loc[i,'num_c'] < cntr_h_h:

			sms_test.set_value(i,'pred','ham')

		elif sms_test.loc[i,'num_c'] > cntr_s_l:

			sms_test.set_value(i,'pred','spam')

		else:

			sms_test.set_value(i,'pred','unk')

		if sms_test.loc[i,'v1'] == sms_test.loc[i,'pred']:

			sms_test.set_value(i, 'eval', 1)

		else:

			sms_test.set_value(i, 'eval', 0)



	print ("{:.1f}".format(sms_test['eval'].sum()*100/len(sms_test)),'%')

	return sms_predict
sms_predict(0,999,1000,1999)