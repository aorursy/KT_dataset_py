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



glassdata = pd.read_csv('../input/glass.csv') 



glassdata.describe()

# Notice that in the description, it has 1-7 types, but the dataset does not have Type 4
# Let's count the number of glasses in each group



pd.value_counts(glassdata['Type'].values.ravel()) 





# It is clear that type 6 has very few data. 

# That's why it is hard to predict glasses in that type.
# According to http://www.mathwords.com/o/outlier.htm, we can calulate the outliers. 

# The formula tells us that any number that outside the range [Q1 - 1.5*IQR, Q2 + 1.5*IQR] will be outliers



# For examples, for Na, 

# IQR = Q3 - Q1 = 13.825000 - 12.907500 = 0.9175

# Q1 - 1.5*IQR = 12.907500 - 1.5*0.9175 = 11.53125 

# Q3 + 1.5*IQR = 13.825000 + 1.5*0.9175 = 15.20125



# Therefore any number that's outside [11.53125, 15.20125] will be an outlier. 

# Since the maximum number for Na is 17.380000, there exists at least one outlier. 



# Before we remove the outliers, let's first check the box plot for each feature. 



import seaborn as sns

import matplotlib.pyplot as plt

feature_names = glassdata.columns

for i in range(len(feature_names)-1):

    figure = plt.figure()

    ax = sns.boxplot(x='Type', y=feature_names[i], data=glassdata)
# The diamond shaped dots outside the boxplot indicates the outliers 

# There are some extreme cases in K, Ba and Fe. 

# That why it is necessary to rule out the outliers 



df = glassdata.copy(deep=True) # Make a copy of original data, just in case



# Create new dataframe for each type



types = df['Type'].unique()

d = {type: df[df['Type'] == type] for type in types}



d[1]
# Set the quantile



low = .25

high = .75



bounds = {}

for type in types:

    filt_df = d[type].loc[:, d[type].columns != 'Type'] # Remove 'Type' Column

    quant_df = filt_df.quantile([low, high])

    IQR = quant_df.iloc[1,:]-  quant_df.iloc[0,:]

    quant_df.iloc[0,:] = quant_df.iloc[0,:] - 1.5*IQR

    quant_df.iloc[1,:] = quant_df.iloc[1,:] + 1.5*IQR

    bounds[type] = quant_df

    

bounds[1]
# Define our new dataset by removing the outliers 



filt_df = d[1].loc[:, d[1].columns != 'Type'] # Remove 'Type' Column

filt_df = filt_df.apply(lambda x: x[(x>bounds[1].loc[low,x.name]) & (x < bounds[1].loc[high,x.name])], axis=0)

filt_df = pd.concat([filt_df,d[1].loc[:,'Type']], axis=1)



filt_df
# Let's remove the outliers from the dataset 

df_new = {}



for type in types:

    filt_df = d[type].loc[:, d[type].columns != 'Type'] # Remove 'Type' Column

    filt_df = filt_df.apply(lambda x: x[(x>bounds[type].loc[low,x.name]) & (x < bounds[type].loc[high,x.name])], axis=0)

    df_new[type] = pd.concat([filt_df,d[type].loc[:,'Type']], axis=1)





glassdata_new = result = pd.concat(df_new)

glassdata_new
# Now we have our glass data that has all outliers removed

# Check out the boxplot again



for i in range(len(feature_names)-1):

    figure = plt.figure()

    ax = sns.boxplot(x='Type', y=feature_names[i], data=glassdata_new)
# Now let's look at the average of the glass_newdata



types = np.unique(train['Type'])



for i in range(len(types)):

    fig = plt.figure()

    average = glassdata_new[[glassdata_new.columns[i], "Type"]].groupby(['Type'],as_index=False).mean()

    sns.barplot(x = 'Type', y = glassdata_new.columns[i], data= average)
# There is some great information. 

# 1. Type 7 vanished in Mg 

# 2. Type 6 vanished in K



# This two important messages can help us classify Type 6 and 7! 

# We do not need to worry about the small data for these two types. 



# Check the original data 



glassdata[glassdata['Type'] == 6]



# Like what we expected all K are 0. That the character for 6. 
glassdata[glassdata['Type'] == 7]



# There are few has nonzero Mg in Type 7, but most of them has 0. 
# Now let's plot type 6 and 7 using Mg and K





x_6 = glassdata[glassdata['Type']==6]['K']

y_6 = glassdata[glassdata['Type']==6]['Mg']

plt.scatter(x_6,y_6, color = 'red', label = "Type 6")



x_7 = glassdata[glassdata['Type']==7]['K']

y_7 = glassdata[glassdata['Type']==7]['Mg']

plt.scatter(x_7,y_7, color = 'blue', label = "Type 7")



plt.legend()

plt.show()
# From the picture above, it is easy to see, except for one outlier in Type 6 (right top corner)

# The rest are fairly easy to distinguish between 6 and 7. 



# Let's just use Mg and K for SVM



# Now let's separate the data in to training and test data



alpha = 0.7 # training data ratio



# Splitting glassdata to training and test data

train = pd.DataFrame()

test = pd.DataFrame()

for i in range(len(types)):

    tempt = glassdata_new[glassdata_new.Type == types[i]]

    train = train.append(tempt[0:int(alpha*len(tempt))])

    test = test.append(tempt[int(alpha*len(tempt)): len(tempt)])

    # test.append(tempt[int(alpha*len(tempt)): len(tempt)])



# Check whether the dimension match

print (train.shape, test.shape, glassdata.shape)
fig = plt.figure()

ax = fig.add_subplot(111)



for type in types:

    x = glassdata_new[glassdata_new['Type'] == type]['Mg']

    y = glassdata_new[glassdata_new['Type'] == type]['K']

    ax.scatter(x,y,label = "Type" + str(type))



plt.legend()

plt.show()





# Calculate the distance between test points and the centers 


