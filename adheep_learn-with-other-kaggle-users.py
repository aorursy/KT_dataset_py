# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_columns',100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ls
train = pd.read_csv('../input/learn-together/train.csv')
train.head()
train.shape
train.drop(['Id'], axis = 1).describe()
binary_rows = train.iloc[:,11:55]

binary_rows['Id'] = train['Id']

binary_rows['Cover_Type'] = train['Cover_Type']
train.drop(binary_rows, axis = 1).skew()
train.drop(binary_rows, axis = 1).head(2)
corr = train.drop(binary_rows, axis = 1).corr()
mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (10,10))

sns.heatmap(corr, annot= True, linewidths= 3, mask = mask, fmt='.1f')
robustscaler = preprocessing.RobustScaler()

stdscaler = preprocessing.StandardScaler()

minmaxscaler = preprocessing.MinMaxScaler()

maxabsscaler = preprocessing.MaxAbsScaler()

normalizer = preprocessing.Normalizer()

quantiletransformation_normal = preprocessing.QuantileTransformer(output_distribution= 'normal')

quantiletransformation_uniform = preprocessing.QuantileTransformer(output_distribution= 'uniform')

yeo_johnson = preprocessing.PowerTransformer(method= 'yeo-johnson')

box_cox = preprocessing.PowerTransformer(method= 'box-cox')







stdscaled_train = stdscaler.fit_transform(train.drop(binary_rows, axis = 1))

stdscaled_train_df = pd.DataFrame(stdscaled_train,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



minmaxscaler_train = minmaxscaler.fit_transform(train.drop(binary_rows, axis = 1))

minmaxscaler_train_df = pd.DataFrame(minmaxscaler_train,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



maxabsscaler_train = maxabsscaler.fit_transform(train.drop(binary_rows, axis = 1))

maxabsscaler_train_df = pd.DataFrame(maxabsscaler_train,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



rscaled_train = robustscaler.fit_transform(train.drop(binary_rows, axis = 1))

rscaled_train_df = pd.DataFrame(rscaled_train,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



norm_train = normalizer.fit_transform(train.drop(binary_rows, axis = 1))

norm_train_df = pd.DataFrame(norm_train ,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



quantiletransformation_normal = quantiletransformation_normal.fit_transform(train.drop(binary_rows, axis = 1))

quantiletransformation_normal_df = pd.DataFrame(quantiletransformation_normal ,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



quantiletransformation_uniform = quantiletransformation_uniform.fit_transform(train.drop(binary_rows, axis = 1))

quantiletransformation_uniform_df = pd.DataFrame(quantiletransformation_uniform ,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



yeo_johnson = yeo_johnson.fit_transform(train.drop(binary_rows, axis = 1))

yeo_johnson_df = pd.DataFrame(yeo_johnson ,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])



box_cox = box_cox.fit_transform(train[train.drop(binary_rows, axis = 1).apply(lambda x: x>0)].drop(binary_rows, axis = 1))

box_cox_df = pd.DataFrame(box_cox ,columns=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'])







ax1 = train.drop(binary_rows, axis = 1).plot.kde()

ax1.set_title('Before Scaling')

ax6 = norm_train_df.plot.kde(legend = None)

ax6.set_title('Normalizer Scaling')

ax2= minmaxscaler_train_df.plot.kde(legend = None)

ax2.set_title('Minmax Scaling')

ax3 = maxabsscaler_train_df.plot.kde(legend = None)

ax3.set_title('Maxabs Scaling')

ax4 = stdscaled_train_df.plot.kde(legend = None)

ax4.set_title('Standard Scaling')

ax5 = rscaled_train_df.plot.kde(legend = None)

ax5.set_title('Robust Scaling')

ax7 = quantiletransformation_normal_df.plot.kde()

ax7.set_title('Quantile Normal Transformation')

ax8 = quantiletransformation_uniform_df.plot.kde()

ax8.set_title('Quantile Uniform Transformation')

ax9 = yeo_johnson_df.plot.kde()

ax9.set_title('Yeo Johnson Power Transformation')

ax10 = box_cox_df.plot.kde()

ax10.set_title('Box Cox Power Transformation')
print('Normalizer skewness')

print(norm_train_df.skew())

print('\nMinmax Scaler Skewness')

print(minmaxscaler_train_df.skew())

print('\nMaxabs Scaler Skewness')

print(maxabsscaler_train_df.skew())

print('\nStandard Scaler Skewness')

print(stdscaled_train_df.skew())

print('\nRobust Scaling Skewness')

print(rscaled_train_df.skew())

print('\nQuantile Normal Transformation Skewness')

print(quantiletransformation_normal_df.skew())

print('\nQuantile Uniform Transformation Skewness')

print(quantiletransformation_uniform_df.skew())

print('\nYeo Johnson Skewness')

print(yeo_johnson_df.skew())

print('\nBox-cox Skewness')

print(box_cox_df.skew())
quantiletransformation_uniform_df.head(2)
sns.pairplot(quantiletransformation_uniform_df, diag_kind='kade', kind = 'reg')
sns.jointplot(x = train['Hillshade_3pm'], y= train['Hillshade_Noon'], kind = 'hex', color = 'Pink')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Hillshade_3pm'], y= quantiletransformation_uniform_df['Hillshade_Noon'], kind = 'hex', color = 'Pink')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = train['Hillshade_3pm'], y= train['Hillshade_9am'], kind= 'hex')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Hillshade_3pm'], y= quantiletransformation_uniform_df['Hillshade_9am'], kind= 'hex')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = train['Aspect'], y= train['Hillshade_3pm'], kind= 'hex', color='Black')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Aspect'], y= quantiletransformation_uniform_df['Hillshade_3pm'], kind= 'hex', color='Black')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = train['Aspect'], y= train['Hillshade_9am'], kind= 'hex', color='Red')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Aspect'], y= quantiletransformation_uniform_df['Hillshade_9am'], kind= 'hex', color='Red')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = train['Horizontal_Distance_To_Hydrology'], y= train['Vertical_Distance_To_Hydrology'], kind= 'kde', color = 'Yellow')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Horizontal_Distance_To_Hydrology'], y= quantiletransformation_uniform_df['Vertical_Distance_To_Hydrology'], kind= 'kde', color = 'Yellow')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = train['Horizontal_Distance_To_Roadways'], y= train['Elevation'], kind= 'kde', color = 'Orange')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)

sns.jointplot(x = quantiletransformation_uniform_df['Horizontal_Distance_To_Roadways'], y= quantiletransformation_uniform_df['Elevation'], kind= 'kde', color = 'Orange')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = quantiletransformation_uniform_df['Slope'], y= quantiletransformation_uniform_df['Hillshade_Noon'], kind= 'hex', color='Purple')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
sns.jointplot(x = quantiletransformation_uniform_df['Horizontal_Distance_To_Roadways'], y= quantiletransformation_uniform_df['Horizontal_Distance_To_Fire_Points'], kind= 'kde', color='Green')

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(10,6)
for i in train[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]:

    train[i].replace(1,i,inplace = True)
train['Wilderness_Area'] = train[['Wilderness_Area1', 

                                  'Wilderness_Area2', 

                                  'Wilderness_Area3', 

                                  'Wilderness_Area4']].apply(lambda x: x['Wilderness_Area1'] 

                                                                    or x['Wilderness_Area2'] 

                                                                    or x['Wilderness_Area3'] 

                                                                    or x['Wilderness_Area4'], axis = 1)
for i in train.drop(binary_rows, axis = 1).columns:

    print(i,train[i].nunique())
plt.figure(figsize = (25,10))

ax = sns.countplot(train['Wilderness_Area'], palette= 'Reds_r', order= train['Wilderness_Area'].value_counts().index)

ax.set_xticklabels(train['Wilderness_Area'].value_counts().index, fontsize = 15)

ax.set_yticklabels(range(0,7000,1000), fontsize = 15)

ax.set_xlabel('Wilderness Area', fontdict = {'size':20})

ax.set_ylabel('Count', fontdict = {'size':20})

ax.set_title('Number of Wilderness Area', fontdict = {'size':25, 'weight':'bold'})

ax.grid(True, c= 'black')

pd.crosstab(train['Cover_Type'], train['Wilderness_Area'])
ax = pd.crosstab(train['Cover_Type'], train['Wilderness_Area']).plot.barh(align = 'center', width = 0.8)

ax.set_xlabel('Number of Listings')

ax.set_title("Number of Listings with it's Type in each Borough", fontdict = {'size':15, 'weight':'bold'})

fig = plt.gcf()

plt.tight_layout()

fig.set_size_inches(18,6)

plt.grid(True, color = 'black')
train.head(2)