from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
alcdata = pd.read_csv("alcoholism/student-mat.csv")
#fifadata = pd.read_csv("fifa18/data.csv")
#accidata1 = pd.read_csv("accidents/accidents_2005_to_2007.csv")
#accidata2 = pd.read_csv("accidents/accidents_2009_to_2011.csv")
#accidata3 = pd.read_csv("accidents/accidents_2012_to_2014.csv")
std_grades
alcdata.nunique(axis=0)  # To check count of unique values in each column. If unique_val = 2, use label enc, else one-hot.

#Extract the columns for label encoding
label_feature = alcdata[['school','sex','address','famsize','Pstatus','schoolsup','famsup','paid',
                      'activities','nursery','higher','internet','romantic']]
le = LabelEncoder()

#do label encoding
for i in label_feature:
    alcdata[i] = le.fit_transform(alcdata[i])
    
#Extract columns for one hot encoding
one_hot_feature = alcdata[['Mjob','Fjob','reason','guardian']]
temp1 = alcdata.copy()

one_done = pd.get_dummies(one_hot_feature)
del temp1['Mjob']
del temp1['Fjob']
del temp1['reason']
del temp1['guardian']
pd.concat([temp1,one_done],axis=1)
temp_data = alcdata[['famrel','Pstatus','avg_grade']]
temp_data.head()

sns.set_style("whitegrid")
sns.pairplot(temp_data,hue='Pstatus')

sns.set_style("whitegrid")
sns.pairplot(temp_data,hue='famrel')

#Read the plot and see the affect of these features.
# Let's see from histograms which features are skewed.
temp_alc.hist(bins=50, figsize=(15,15))
plt.show()

#We will also use skew function to see which features are skewed
temp_alc.skew() 

#To reduce skewness, i used log10()
skew_free = temp_alc.copy()
skew_free['traveltime'] = np.log10(temp_alc['traveltime'] + 1)
skew_free['absences'] = np.log10(temp_alc['absences'] + 1)
skew_free['Dalc'] = np.log10(temp_alc['Dalc'] + 1)
skew_free['failures'] = np.log10(temp_alc['failures'] + 1)
skew_free.skew()

#This is just for verification after removing skewness.
skew_free.hist(bins=50, figsize=(15,15))
plt.show()
fifadata.head()
fifadata.columns
fifadata.isnull().sum()
# Unnamed 0 is useless to use so we drop it.
del fifadata['Unnamed: 0']

temp_fifa = fifadata.copy() #Let's create a copy just to be on the safe side

#Imputing missing integer values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')         # We are using the 'mean' strategy to fill in the missing values.
imputer.fit(temp_fifa.iloc[:, -6:-1])                                   #compute the missing values
temp_fifa.iloc[:, -6:-1] = imputer.transform(temp_fifa.iloc[:, -6:-1])   # fill in the missing values

temp_fifa['Release Clause'].replace('', np.nan, inplace=True)  #Replace empty values with nan so that we can use dropna()
temp_fifa.dropna(subset=['Release Clause'], inplace=True)

#Converting string to appropriate float value
temp_fifa['Wage'] = temp_fifa['Wage'].map(lambda x: x.lstrip('€').replace('M','000000').replace('K','000').replace('.',''))
temp_fifa['Value'] = temp_fifa['Value'].map(lambda x: x.lstrip('€').replace('M','000000').replace('K','000').replace('.',''))
temp_fifa['Release Clause'] = temp_fifa['Release Clause'].map(lambda x: x.lstrip('€').replace('M','000000').replace('K','000').replace('.',''))

temp_fifa['Wage'] = temp_fifa['Wage'].astype(float)
temp_fifa['Value'] = temp_fifa['Value'].astype(float)
temp_fifa['Release Clause'] = temp_fifa['Release Clause'].astype(float)

club_group = temp_fifa.groupby('Club')  #to get economy of clubs , group by
club_wage = club_group['Wage'].agg(np.sum)
club_value = club_group['Value'].agg(np.sum)
club_RC = club_group['Release Clause'].agg(np.sum)
economy = club_RC - (club_wage + club_value)
economy.sort_values(ascending=False)[:10]  #To get the top 10 clubs with most profit.








