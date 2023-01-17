# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Test.csv', low_memory=False)
data.head()
# data = data.astype(object).where(pd.notnull(data),"None")

# data.head()
# data = data.astype(int)

data = data.astype({"SalesID": int, "MachineID": int, "datasource":int,"YearMade":int})

# data.info()
data.head()
# check if a column has any nan values

data.fiSecondaryDesc.isna().any()
# replacing with mean values

# data.fillna(0, inplace=True)

data['MachineHoursCurrentMeter'] = data['MachineHoursCurrentMeter'].fillna((data['MachineHoursCurrentMeter'].mean()))

# data.MachineHoursCurrentMeter.isna().any()

data['UsageBand'] = data['UsageBand'].fillna('NA')

data.UsageBand.unique()
data['saledate'] = data['saledate'].astype('datetime64[ns]')
# # sns.heatmap(data, cmap='RdYlGn_r', linewidths=0.5, annot=True)

# cols_lis = data.columns

# # cols_lis = data.columns

# # cols_lis

# for col in cols_lis:

#     data[col].plot.hist()

#     plt.title(col)

#     plt.show()

    

# def plot_corr(df,size=10):

#     '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



#     Input:

#         df: pandas DataFrame

#         size: vertical and horizontal size of the plot'''



#     corr = df.corr()

#     fig, ax = plt.subplots(figsize=(size, size))

#     ax.matshow(corr)

#     plt.xticks(range(len(corr.columns)), corr.columns);

#     plt.yticks(range(len(corr.columns)), corr.columns);

    



# data.corr(method='pearson')

# plot_corr(data)

# rs = np.random.RandomState(0)

# df = pd.DataFrame(rs.rand(10, 10))

# corr = data.corr()

# corr.style.background_gradient(cmap='coolwarm')
data.fiSecondaryDesc.fillna('NA', inplace=True)

data.fiModelDescriptor.fillna('NA',inplace=True)

data.ProductSize.fillna('NA',inplace=True)

data.Drive_System.fillna('NA',inplace=True)

data.Enclosure.fillna('NA',inplace=True)

data.Forks.fillna('None or Unspecified',inplace=True)

data.Pad_Type.fillna('None or Unspecified',inplace=True)

data.Ride_Control.fillna('None or Unspecified',inplace=True)

data.Stick.fillna('NA',inplace=True)

data.Transmission.fillna('None or Unspecified',inplace=True)

data.Turbocharged.fillna('None or Unspecified',inplace=True)

data.Blade_Extension.fillna('None or Unspecified',inplace=True)

data.Blade_Width.fillna('None or Unspecified',inplace=True)

data.Enclosure_Type.fillna('None or Unspecified',inplace=True)

data.Engine_Horsepower.fillna('Not Applicable',inplace=True)

data.Hydraulics.fillna('NA',inplace=True)

data.Pushblock.fillna('None or Unspecified',inplace=True)

data.Ripper.fillna('None or Unspecified',inplace=True)

data.Scarifier.fillna('None or Unspecified',inplace=True)

data.Tip_Control.fillna('None or Unspecified',inplace=True)

data.Tire_Size.fillna('None or Unspecified',inplace=True)

data.Coupler.fillna('None or Unspecified',inplace=True)

data.Grouser_Tracks.fillna('None or Unspecified', inplace=True)

data.Coupler_System.fillna('None or Unspecified', inplace=True)

data.Hydraulics_Flow.fillna('None or Unspecified', inplace=True)

data.Track_Type.fillna(data.Track_Type.mode()[0],inplace=True)

data.Undercarriage_Pad_Width.fillna('None or Unspecified', inplace=True)

data.Stick_Length.fillna('None or Unspecified', inplace=True)

data.Thumb.fillna('None or Unspecified', inplace=True)

data.Pattern_Changer.fillna('None or Unspecified', inplace=True)

data.Grouser_Type.fillna('Not Specified', inplace=True)

data.Backhoe_Mounting.fillna('Not Specified', inplace=True)

data.Blade_Type.fillna('Not Specified', inplace=True)

data.Travel_Controls.fillna('Not Specified', inplace=True)

data.Differential_Type.fillna('Not Specified', inplace=True)

data.Steering_Controls.fillna('Not Specified', inplace=True)

data.fiModelSeries.fillna('NA', inplace=True)
cleansed = data.select_dtypes(include=['object']).copy()
# get unique values of a column in a panda

cleansed.UsageBand.value_counts()
# plotting and checking a column

UsageBand = cleansed['UsageBand'].value_counts()

sns.set(style="darkgrid")

sns.barplot(UsageBand.index, UsageBand.values, alpha=0.9)

plt.title('Frequency Distribution of Usageband')

plt.ylabel('Number of Occurrences', fontsize=12)
# plotting a pie chart similarly

labels = cleansed['UsageBand'].astype('category').cat.categories.tolist()

counts = cleansed['UsageBand'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot

ax1.axis('equal')

plt.show()
# general label encoding

cleansed.UsageBand = cleansed.UsageBand.astype('category')

cleansed['UsageBand'] = cleansed['UsageBand'].cat.codes

cleansed.UsageBand.unique()
# using scikit label encoding

lb_make = LabelEncoder()

cleansed['Tire_Size_id'] = lb_make.fit_transform(cleansed['Tire_Size'])

cleansed.Tire_Size_id.unique()
# one hot encoding

cleansed = pd.get_dummies(cleansed, columns=['Blade_Type'], prefix = ['Blade_Type'])

cleansed.head()