import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/all.csv')

#data.head()

#data.shape

#data.columns

#np.unique(data.State)

#data.isnull().sum() -> todo: deal with these

#data.groupby('State').mean()
#how many numeric columns are there? 47?

num_cols = data._get_numeric_data().columns

#print(num_cols.shape)



#what are the categorical columns??

cat_cols = list(set(data.columns)-set(num_cols))

#data[num_cols].isnull().sum()
# let's plot the average data statewise

x = range(len(np.unique(data['State'])))

'''a = data.groupby('State').mean()

a = a.drop('Unnamed: 0',axis=1)

num_cols = num_cols.drop('Unnamed: 0')

for cols in num_cols:

    plt.figure()

    plt.bar(x,a[cols],alpha=0.5)

    plt.xticks(x,a.index,rotation = 90)

    plt.ylabel('mean value per state')

    plt.title('%s'%cols)

    plt.show()'''

#uncomment for the quick peek plots
cols_of_interest = ['State','Persons','Males', 'Females','Persons..literate',

       'Males..Literate', 'Females..Literate','Below.Primary', 'Primary', 'Middle',

       'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above', 'X0...4.years',

       'X5...14.years', 'X15...59.years', 'X60.years.and.above..Incl..A.N.S..',

       'Total.workers','Main.workers', 'Marginal.workers', 'Non.workers',

       'Drinking.water.facilities', 'Safe.Drinking.water',

       'Electricity..Power.Supply.', 'Electricity..domestic.',

       'Electricity..Agriculture.', 'Primary.school', 'Middle.schools',

       'Secondary.Sr.Secondary.schools', 'College', 'Medical.facility',

       'Primary.Health.Centre', 'Primary.Health.Sub.Centre',

       'Post..telegraph.and.telephone.facility', 'Bus.services',

       'Paved.approach.road', 'Mud.approach.road', 'Permanent.House',

       'Semi.permanent.House', 'Temporary.House']



state_wise_data = data[cols_of_interest].groupby('State').sum()

#state_wise_data
state_wise_data['Size'] = [8249,275045,83743,78438,94163,135192,114,111,491,1483,

                           3702,196244,55673,44212,222236,79716,191791,38852,30,

                           308252,307713,22327,22429,21081,16579,155707,490,50362,

                           342239,7096,130060,10486,240928,53483,88752]
'''plt.scatter(state_wise_data['Size'],state_wise_data['Drinking.water.facilities'],alpha=0.5)

plt.xlabel('Size')

plt.ylabel('Drinking water facilities')

plt.xticks(rotation=90)'''
'''plt.scatter(state_wise_data['Persons'],state_wise_data['Drinking.water.facilities'],alpha=0.5)

plt.xlabel('persons')

plt.ylabel('Drinking water facilities')

plt.xticks(rotation=90)'''
'''

#IS there relation between population and area of states?

plt.scatter(state_wise_data['Persons'],state_wise_data['Size'],alpha=0.5)

plt.xlabel('Persons')

plt.ylabel('Area')

plt.xticks(rotation=90)'''
import seaborn as sns

corrmat = state_wise_data.corr()

sns.heatmap(corrmat, square=True)
state_wise_data.columns
lit_ratio = np.divide(state_wise_data['Persons..literate'],state_wise_data['Persons'])

emp_ratio = np.divide(state_wise_data['Total.workers'],state_wise_data['Persons'])

plt.scatter(lit_ratio,emp_ratio)

plt.xlabel('literacy ratio')

plt.ylabel('employment ratio')

for i,j,k in zip(range(35),lit_ratio,emp_ratio):

    plt.annotate(i,xy = (j,k))

for i,j in zip(range(35),state_wise_data.index):

    print(i,j)

    