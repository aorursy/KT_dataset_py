import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Fill in the line below: Specify the path of the CSV file to r

my_filepath = '../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath,index_col='sl_no')



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the d

my_data.head()

import pandas as pd

from sklearn.impute import SimpleImputer

my_data = my_data.fillna(0)

my_data['status'] = my_data['status'].apply(lambda x:1 if x == 'Placed' else 0)

my_data['ssc_b'] = my_data['ssc_b'].apply(lambda x:1 if x == 'Central' else 0)



my_data['hsc_b'] = my_data['hsc_b'].apply(lambda x:1 if x == 'Central'  else 0)

my_data['science'] = my_data['hsc_s'].apply(lambda x:1 if x == 'Science'  else 0)

my_data['Arts'] = my_data['hsc_s'].apply(lambda x:1 if x == 'Arts' else 0)

my_data.drop(['hsc_s'],axis=1)

my_data['workex'] = my_data['workex'].apply(lambda x:1 if x == 'No' else 0)

my_data['specialisation'] = my_data['specialisation'].apply(lambda x:1 if x == 'Mkt&HR' else 0)

my_data['degree_t'] = my_data['degree_t'].apply(lambda x:1 if x == 'Sci&Tech' else 0)





my_data.head()
sns.swarmplot(y= 'ssc_p',x='ssc_b',data = my_data)
sns.swarmplot(y= 'hsc_p',x='hsc_b',data = my_data)
plt.subplot()

sns.swarmplot(x= 'science',y='salary',data = my_data)

sns.swarmplot(y= 'salary',x='Arts',data = my_data)
sns.lmplot(x = 'mba_p',y = 'hsc_p',hue = 'science',data= my_data)
sns.lmplot(x = 'mba_p',y = 'hsc_p',hue = 'Arts',data= my_data)
plt.figure(figsize = (100,20))

sns.barplot(x= my_data.index,y=my_data['ssc_p'])

plt.xlabel('index')

plt.figure(figsize = (100,20))

sns.barplot(x= my_data.index,y=my_data['hsc_p'])

plt.xlabel('index')

plt.figure(figsize = (100,20))

sns.barplot(x= my_data.index,y=my_data['mba_p'])

plt.xlabel('index')

sns.lmplot(x= 'ssc_p',y = 'hsc_p',hue= 'status',data= my_data)
sns.lmplot(x= 'ssc_p',y = 'mba_p',hue= 'status',data= my_data)
sns.lmplot(x= 'mba_p',y = 'hsc_p',hue= 'status',data= my_data)
sns.swarmplot(x = 'specialisation'	,y = 'salary' ,data = my_data)
sns.swarmplot(y = 'ssc_p'	,x = 'status' ,data = my_data)
sns.swarmplot(y = 'hsc_p'	,x = 'status' ,data = my_data)
sns.swarmplot(y = 'mba_p'	,x = 'status' ,data = my_data)