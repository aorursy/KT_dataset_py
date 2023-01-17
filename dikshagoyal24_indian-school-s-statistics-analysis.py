# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt



girls_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')

boys_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')

water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')

electricity = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')

comps = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')

enrollment_ratio = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

dropout_ratio = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')

print(girls_toilet.head())
girls_toilet.info()
data = girls_toilet.loc[girls_toilet['State_UT']=='All India']

plt.bar(data.year,data['All Schools'],color = 'm')

plt.legend(['All Schools'])

plt.title('AllIndia_girls_toilet_perc')
X = np.arange(3)

fig = plt.subplots(figsize =(15,5)) 

plt.bar(X + 0.00, data['Primary_Only'], color = 'b', width = 0.05, label='Primary_Only')

plt.bar(X + 0.05, data['U_Primary_Only'], color = 'g', width = 0.05,label = 'U_Primary_Only')

plt.bar(X + 0.10, data['Primary_with_U_Primary_Sec_HrSec'], color = 'r', width = 0.05,label='Primary_with_U_Primary_Sec_HrSec')

plt.bar(X + 0.15, data['Primary_with_U_Primary'], color = 'm', width = 0.05,label='Primary_with_U_Primary')

plt.bar(X + 0.20, data['U_Primary_With_Sec_HrSec'], color = 'c', width = 0.05,label='U_Primary_With_Sec_HrSec')

plt.bar(X + 0.25, data['Primary_with_U_Primary_Sec'], color = 'y', width = 0.05,label='Primary_with_U_Primary_Sec')

plt.bar(X + 0.30, data['U_Primary_With_Sec'], color = 'k', width = 0.05,label='U_Primary_With_Sec')

plt.bar(X + 0.35, data['Sec_Only'], color = 'grey', width = 0.05,label='Sec_Only')

plt.bar(X + 0.40, data['Sec_with_HrSec.'], color = 'orange', width = 0.05,label='Sec_with_HrSec.')

plt.bar(X + 0.45, data['HrSec_Only'], color = 'turquoise', width = 0.05,label='HrSec_Only')





plt.legend(['Primary_Only','U_Primary_Only','Primary_with_U_Primary_Sec_HrSec','Primary_with_U_Primary','U_Primary_With_Sec_HrSec',

           'Primary_with_U_Primary_Sec','U_Primary_With_Sec','Sec_Only','Sec_with_HrSec.','HrSec_Only'], loc='center left',bbox_to_anchor=(1, 0.5))

plt.xlabel('YEAR', fontweight ='bold') 

plt.ylabel('PERCENTAGE', fontweight ='bold') 

plt.xticks([r + 0.20 for r in range(0,3)], 

           ['2013-14', '2014-15', '2015-16']) 

plt.title('AllIndia_Girls_toilet_perc')

#plt.show()
boys_data = boys_toilet.loc[boys_toilet['State_UT']=='All India']

plt.bar(boys_data.year,boys_data['All Schools'], color='c')

plt.legend(['All Schools'])

plt.title('AllIndia_boys_toilet_perc')
X = np.arange(3)

fig = plt.subplots(figsize =(15,5)) 

plt.bar(X + 0.00, boys_data['Primary_Only'], color = 'b', width = 0.05, label='Primary_Only')

plt.bar(X + 0.05, boys_data['U_Primary_Only'], color = 'g', width = 0.05,label = 'U_Primary_Only')

plt.bar(X + 0.10, boys_data['Primary_with_U_Primary_Sec_HrSec'], color = 'r', width = 0.05,label='Primary_with_U_Primary_Sec_HrSec')

plt.bar(X + 0.15, boys_data['Primary_with_U_Primary'], color = 'm', width = 0.05,label='Primary_with_U_Primary')

plt.bar(X + 0.20, boys_data['U_Primary_With_Sec_HrSec'], color = 'c', width = 0.05,label='U_Primary_With_Sec_HrSec')

plt.bar(X + 0.25, boys_data['Primary_with_U_Primary_Sec'], color = 'y', width = 0.05,label='Primary_with_U_Primary_Sec')

plt.bar(X + 0.30, boys_data['U_Primary_With_Sec'], color = 'k', width = 0.05,label='U_Primary_With_Sec')

plt.bar(X + 0.35, boys_data['Sec_Only'], color = 'grey', width = 0.05,label='Sec_Only')

plt.bar(X + 0.40, boys_data['Sec_with_HrSec.'], color = 'orange', width = 0.05,label='Sec_with_HrSec.')

plt.bar(X + 0.45, boys_data['HrSec_Only'], color = 'turquoise', width = 0.05,label='HrSec_Only')





plt.legend(['Primary_Only','U_Primary_Only','Primary_with_U_Primary_Sec_HrSec','Primary_with_U_Primary','U_Primary_With_Sec_HrSec',

           'Primary_with_U_Primary_Sec','U_Primary_With_Sec','Sec_Only','Sec_with_HrSec.','HrSec_Only'], loc='center left',bbox_to_anchor=(1, 0.5))

plt.xlabel('YEAR', fontweight ='bold') 

plt.ylabel('PERCENTAGE', fontweight ='bold') 

plt.xticks([r + 0.20 for r in range(0,3)], 

           ['2013-14', '2014-15', '2015-16']) 

plt.title('AllIndia_boys_toilet_perc')

#plt.show()
a = girls_toilet.loc[girls_toilet['year']=='2014-15'].groupby('State_UT').sum().reset_index()

fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(a['State_UT'],a['Primary_Only'])

plt.bar(a['State_UT'],a['U_Primary_Only'])

plt.bar(a['State_UT'],a['Primary_with_U_Primary_Sec_HrSec'])

plt.bar(a['State_UT'],a['Primary_with_U_Primary'])

plt.bar(a['State_UT'],a['U_Primary_With_Sec_HrSec'])

plt.bar(a['State_UT'],a['Primary_with_U_Primary_Sec'])

plt.bar(a['State_UT'],a['U_Primary_With_Sec'])

plt.bar(a['State_UT'],a['Sec_Only'])

plt.bar(a['State_UT'],a['Sec_with_HrSec.'])

plt.bar(a['State_UT'],a['HrSec_Only'])

plt.legend(['Primary_Only','U_Primary_Only','Primary_with_U_Primary_Sec_HrSec','Primary_with_U_Primary','U_Primary_With_Sec_HrSec',

           'Primary_with_U_Primary_Sec','U_Primary_With_Sec','Sec_Only','Sec_with_HrSec.','HrSec_Only'],loc='center left',bbox_to_anchor=(1, 0.5))
b = boys_toilet.loc[boys_toilet['year']=='2014-15'].groupby('State_UT').sum().reset_index()

fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(b['State_UT'],b['Primary_Only'])

plt.bar(b['State_UT'],b['U_Primary_Only'])

plt.bar(b['State_UT'],b['Primary_with_U_Primary_Sec_HrSec'])

plt.bar(b['State_UT'],b['Primary_with_U_Primary'])

plt.bar(b['State_UT'],b['U_Primary_With_Sec_HrSec'])

plt.bar(b['State_UT'],b['Primary_with_U_Primary_Sec'])

plt.bar(b['State_UT'],b['U_Primary_With_Sec'])

plt.bar(b['State_UT'],b['Sec_Only'])

plt.bar(b['State_UT'],b['Sec_with_HrSec.'])

plt.bar(b['State_UT'],b['HrSec_Only'])

plt.legend(['Primary_Only','U_Primary_Only','Primary_with_U_Primary_Sec_HrSec','Primary_with_U_Primary','U_Primary_With_Sec_HrSec',

           'Primary_with_U_Primary_Sec','U_Primary_With_Sec','Sec_Only','Sec_with_HrSec.','HrSec_Only'],loc='center left',bbox_to_anchor=(1, 0.5))
dropout_ratio.head()

dropout_ratio1 = dropout_ratio[['State_UT','year','Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total']]
print(dropout_ratio1.head())
c = dropout_ratio1.loc[dropout_ratio1['year'] =='2012-13']

d = dropout_ratio1.loc[dropout_ratio1['year'] == '2013-14']

e = dropout_ratio1.loc[dropout_ratio1['year'] == '2014-15']
c = c.replace(to_replace='NR',value='NaN',regex = True)

c['Primary_Total'] = c['Primary_Total'].values.astype(np.float32)

c['Upper Primary_Total'] = c['Upper Primary_Total'].values.astype(np.float32)

c['Secondary _Total'] = c['Secondary _Total'].values.astype(np.float32)

c['HrSecondary_Total'] = c['HrSecondary_Total'].values.astype(np.float32)



d = d.replace(to_replace='NR',value='NaN',regex = True)

d['Primary_Total'] = d['Primary_Total'].values.astype(np.float32)

d['Upper Primary_Total'] = d['Upper Primary_Total'].values.astype(np.float32)

d['Secondary _Total'] = d['Secondary _Total'].values.astype(np.float32)

d['HrSecondary_Total'] = d['HrSecondary_Total'].values.astype(np.float32)



e = e.replace(to_replace='NR',value='NaN',regex = True)

e['Primary_Total'] = e['Primary_Total'].values.astype(np.float32)

e['Upper Primary_Total'] = e['Upper Primary_Total'].values.astype(np.float32)

e['Secondary _Total'] = e['Secondary _Total'].values.astype(np.float32)

e['HrSecondary_Total'] = e['HrSecondary_Total'].values.astype(np.float32)
c.info()
#x = np.arange()

fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(c['State_UT'],c['Primary_Total'])

plt.bar(c['State_UT'],c['Upper Primary_Total'])

plt.bar(c['State_UT'],c['Secondary _Total'])

plt.bar(c['State_UT'],c['HrSecondary_Total'])



plt.title('2012-13')

plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])



fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(d['State_UT'],d['Primary_Total'])

plt.bar(d['State_UT'],d['Upper Primary_Total'])

plt.bar(d['State_UT'],d['Secondary _Total'])

plt.bar(d['State_UT'],d['HrSecondary_Total'])



plt.title('2013-14')

plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])



fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(e['State_UT'],e['Primary_Total'])

plt.bar(e['State_UT'],e['Upper Primary_Total'])

plt.bar(e['State_UT'],e['Secondary _Total'])

plt.bar(e['State_UT'],e['HrSecondary_Total'])



plt.title('2014-15')

plt.legend(['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'])
enrollment_ratio.info()
enrollment_ratio = enrollment_ratio.replace(to_replace='NR',value='NaN',regex = True)

enrollment_ratio = enrollment_ratio.replace(to_replace='@',value='NaN',regex = True)
enrollment_ratio['Higher_Secondary_Boys'] = enrollment_ratio['Primary_Total'].values.astype(np.float64)

enrollment_ratio['Higher_Secondary_Girls'] = enrollment_ratio['Higher_Secondary_Girls'].values.astype(np.float64)

enrollment_ratio['Higher_Secondary_Total'] = enrollment_ratio['Higher_Secondary_Total'].values.astype(np.float64)

enrollment_ratio.info()
enrollment_ratio1 = enrollment_ratio.loc[enrollment_ratio['State_UT'] == 'All India']

print(enrollment_ratio1)
X = np.arange(3)

fig = plt.subplots(figsize =(15,5)) 

plt.bar(X + 0.00, enrollment_ratio1['Primary_Boys'], color = 'b', width = 0.05, label='Primary_Boys')

plt.bar(X + 0.05, enrollment_ratio1['Primary_Girls'], color = 'g', width = 0.05,label = 'U_Primary_Girls')

plt.bar(X + 0.15, enrollment_ratio1['Upper_Primary_Boys'], color = 'r', width = 0.05,label='Upper_Primary_Boys')

plt.bar(X + 0.20, enrollment_ratio1['Upper_Primary_Girls'], color = 'm', width = 0.05,label='Upper_Primary_Girls')

plt.bar(X + 0.30, enrollment_ratio1['Secondary_Boys'], color = 'c', width = 0.05,label='Secondary_Boys')

plt.bar(X + 0.35, enrollment_ratio1['Secondary_Girls'], color = 'y', width = 0.05,label='Secondary_Girls')

plt.bar(X + 0.45, enrollment_ratio1['Higher_Secondary_Boys'], color = 'k', width = 0.05,label='Higher_Secondary_Boys')

plt.bar(X + 0.50, enrollment_ratio1['Higher_Secondary_Girls'], color = 'grey', width = 0.05,label='Higher_Secondary_Girls')





plt.legend(['Primary_Boys','Primary_Girls','Upper_Primary_Boys','Upper_Primary_Girls','Secondary_Boys','Secondary_Girls',

            'Higher_Secondary_Boys','Higher_Secondary_Girls'], loc='center left',bbox_to_anchor=(1, 0.5))

plt.xlabel('YEAR', fontweight ='bold') 

plt.ylabel('PERCENTAGE', fontweight ='bold') 

plt.xticks([r + 0.25 for r in range(0,3)], 

           ['2013-14', '2014-15', '2015-16']) 

plt.title('All India Boys & Girls Enrollment Ratio')

#plt.show()
X = np.arange(3)

fig = plt.subplots(figsize =(15,5)) 

plt.bar(X + 0.00, enrollment_ratio1['Primary_Total'], color = 'b', width = 0.10, label='Primary_Total')

plt.bar(X + 0.10, enrollment_ratio1['Upper_Primary_Total'], color = 'g', width = 0.10,label = 'Upper_Primary_Total')

plt.bar(X + 0.20, enrollment_ratio1['Secondary_Total'], color = 'r', width = 0.10,label='Secondary_Total')

plt.bar(X + 0.30, enrollment_ratio1['Higher_Secondary_Total'], color = 'm', width = 0.10,label='Higher_Secondary_Total')





plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'], loc='center left',bbox_to_anchor=(1, 0.5))

plt.xlabel('YEAR', fontweight ='bold') 

plt.ylabel('PERCENTAGE', fontweight ='bold') 

plt.xticks([r + 0.20 for r in range(0,3)], 

           ['2013-14', '2014-15', '2015-16']) 

plt.title('All India Boys & Girls Enrollment Ratio')

#plt.show()
f = enrollment_ratio.loc[enrollment_ratio['Year'] == '2013-14']

g = enrollment_ratio.loc[enrollment_ratio['Year'] == '2014-15']

h = enrollment_ratio.loc[enrollment_ratio['Year'] == '2015-16']

#x = np.arange()

fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(f['State_UT'],f['Primary_Total'])

plt.bar(f['State_UT'],f['Upper_Primary_Total'])

plt.bar(f['State_UT'],f['Secondary_Total'])

plt.bar(f['State_UT'],f['Higher_Secondary_Total'])



plt.title('2013-14')

plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])



fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(g['State_UT'],g['Primary_Total'])

plt.bar(g['State_UT'],g['Upper_Primary_Total'])

plt.bar(g['State_UT'],g['Secondary_Total'])

plt.bar(g['State_UT'],g['Higher_Secondary_Total'])



plt.title('2014-15')

plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])



fig = plt.subplots(figsize =(15,5))

plt.xticks(rotation='vertical')

plt.bar(h['State_UT'],h['Primary_Total'])

plt.bar(h['State_UT'],h['Upper_Primary_Total'])

plt.bar(h['State_UT'],h['Secondary_Total'])

plt.bar(h['State_UT'],h['Higher_Secondary_Total'])



plt.title('2015-16')

plt.legend(['Primary_Total','Upper_Primary_Total','Secondary_Total','Higher_Secondary_Total'])


