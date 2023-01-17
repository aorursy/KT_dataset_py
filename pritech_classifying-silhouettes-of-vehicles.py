# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix

from scipy.stats import zscore

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from sklearn.metrics import accuracy_score



from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from sklearn.metrics import roc_auc_score

from itertools import cycle



import time
df = pd.read_csv('/kaggle/input/vehicle-silhouettes/vehicle.csv')

df.head()
df.rename(columns={'pr.axis_aspect_ratio': 'pr_axis_aspect_ratio', 

                   'max.length_aspect_ratio': 'max_length_aspect_ratio', 

                   'pr.axis_rectangularity': 'pr_axis_rectangularity', 

                   'max.length_rectangularity': 'max_length_rectangularity', 

                   'scaled_variance.1': 'scaled_variance_1', 

                   'scaled_radius_of_gyration.1': 'scaled_radius_of_gyration_1',

                   'skewness_about.1': 'skewness_about_1', 'skewness_about.2': 'skewness_about_2'}, inplace=True)
df.head(1)
rows_count, columns_count = df.shape

print('Total Number of rows :', rows_count)

print('Total Number of columns :', columns_count)
df.info()
df = df.replace({'car': 0, 'bus': 1, 'van':2})

df.head()
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
df.isna().any()
df.apply(lambda x: sum(x.isnull()))
df_transpose = df.describe().T

df_transpose
df_transpose[['min', '25%', '50%', '75%', 'max']]
vehicle_df = df.copy()
vehicle_df.fillna(vehicle_df.median(), axis=0, inplace=True)
vehicle_df.apply(lambda x: sum(x.isnull()))
sns.heatmap(vehicle_df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.figure(figsize=(12,6))

sns.boxplot(data=vehicle_df, orient='h', palette='Set2', dodge=False)
sns.pairplot(vehicle_df, hue='class', diag_kind='kde')
vehicles_counts = pd.DataFrame(vehicle_df['class'].value_counts()).reset_index()

vehicles_counts.columns = ['Labels', 'class']

vehicles_counts['Labels'] = ['Car', 'Bus', 'Van']
vehicles_counts
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(15,7))

sns.countplot(vehicle_df['class'], ax = ax1)

ax1.set_xlabel('Vehicle Type', fontsize=10)

ax1.set_ylabel('Count', fontsize=10)

ax1.set_title('Vehicle Type Distribution')

ax1.set_xticklabels(labels=["Car", 'Bus', 'Van'])



explode = (0, 0.1,0)

ax2.pie(vehicles_counts["class"], explode=explode, labels=["Car", 'Bus', 'Van'], autopct='%1.2f%%',

        shadow=True, startangle=70)

ax2.axis('equal')

plt.title("Vehicles Types Percentage")

plt.legend(["Car", 'Bus', 'Van'], loc=3)

plt.show()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['compactness'],ax=ax1)

ax1.tick_params(labelsize=15)

ax1.set_xlabel('Compactness', fontsize=15)

ax1.set_title("Distribution Plot")





sns.boxplot(vehicle_df['compactness'],ax=ax2)

ax2.set_title("Box Plot")

ax2.set_xlabel('Compactness', fontsize=15)





bins = range(20, 200, 20)

ax3 = sns.distplot(vehicle_df.compactness[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.compactness[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.compactness[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Compactness', fontsize=15)

plt.title("Compactness vs Class")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['circularity'],ax=ax1)

ax1.set_xlabel('Circularity', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['circularity'],ax=ax2)

ax2.set_xlabel('Circularity', fontsize=15)

ax2.set_title("Box Plot")



bins = range(10, 100, 10)

ax3 = sns.distplot(vehicle_df.circularity[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.circularity[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.circularity[vehicle_df['class']==2],ax=ax3, color='yellow', kde=False, bins=bins, label="van")

ax3.set_xlabel('Circularity', fontsize=15)

plt.title("Circularity vs Class")

plt.legend()
ax = vehicle_df[vehicle_df['class']==0].plot.scatter(x='compactness', y='circularity', 

                                                    color='red', label='car')

vehicle_df[vehicle_df['class']==1].plot.scatter(x='compactness', y='circularity', 

                                                color='green', label='bus', ax=ax)

vehicle_df[vehicle_df['class']==2].plot.scatter(x='compactness', y='circularity', 

                                                color='blue', label='van', ax=ax)

ax.set_title("scatter")

plt.title("Distance Circularity vs compactness with Class")

fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['distance_circularity'],ax=ax1)

ax1.set_xlabel('Distance Circularity', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['distance_circularity'],ax=ax2)

ax2.set_xlabel('Distance Circularity', fontsize=15)

ax2.set_title("Box Plot")



bins = range(20, 150, 10)

ax3 = sns.distplot(vehicle_df.distance_circularity[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.distance_circularity[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.distance_circularity[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Distance Circularity', fontsize=15)

plt.title("Distance Circularity vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['radius_ratio'],ax=ax1)

ax1.set_xlabel('Radius Ratio', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['radius_ratio'],ax=ax2)

ax2.set_xlabel('Radius Ratio', fontsize=15)

ax2.set_title("Box Plot")





bins = range(10, 400, 20)

ax3 = sns.distplot(vehicle_df.radius_ratio[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.radius_ratio[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.radius_ratio[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Radius Ratio', fontsize=15)

plt.title("Radius Ratio vs Class Distribution")

plt.legend()
outlier_columns = []



Q1 =  vehicle_df['radius_ratio'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['radius_ratio'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_radius_ratio = Q1 - 1.5 * IQR   # lower bound 

UTV_radius_ratio = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('radius_ratio <',LTV_radius_ratio ,'and >',UTV_radius_ratio, ' are outliers')

print('Numerber of outliers in radius_ratio column below the lower whisker =', vehicle_df[vehicle_df['radius_ratio'] < (Q1-(1.5*IQR))]['radius_ratio'].count())

print('Numerber of outliers in radius_ratio column above the upper whisker =', vehicle_df[vehicle_df['radius_ratio'] > (Q3+(1.5*IQR))]['radius_ratio'].count())



# storing column name and upper-lower bound value where outliers are presense 

outlier_columns.append('radius_ratio')

upperLowerBound_Disct = {'radius_ratio':UTV_radius_ratio}
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['pr_axis_aspect_ratio'],ax=ax1)

ax1.set_xlabel('Axis Aspect Ratio ', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['pr_axis_aspect_ratio'],ax=ax2)

ax2.set_xlabel('Axis Aspect Ratio ', fontsize=15)

ax2.set_title("Box Plot")





bins = range(20, 200, 10)

ax3 = sns.distplot(vehicle_df.pr_axis_aspect_ratio[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.pr_axis_aspect_ratio[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.pr_axis_aspect_ratio[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Axis Aspect Ratio', fontsize=15)

plt.title("Axis Aspect Ratio vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['pr_axis_aspect_ratio'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['pr_axis_aspect_ratio'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_pr_axis_aspect_ratio = Q1 - 1.5 * IQR   # lower bound 

UTV_pr_axis_aspect_ratio = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('pr_axis_aspect_ratio <',LTV_pr_axis_aspect_ratio ,'and >',UTV_pr_axis_aspect_ratio, ' are outliers')

print('Numerber of outliers in axis_aspect_ratio column below the lower whisker =', vehicle_df[vehicle_df['pr_axis_aspect_ratio'] < (Q1-(1.5*IQR))]['pr_axis_aspect_ratio'].count())

print('Numerber of outliers in axis_aspect_ratio column above the upper whisker =', vehicle_df[vehicle_df['pr_axis_aspect_ratio'] > (Q3+(1.5*IQR))]['pr_axis_aspect_ratio'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('pr_axis_aspect_ratio')

upperLowerBound_Disct['pr_axis_aspect_ratio'] = UTV_pr_axis_aspect_ratio
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['max_length_aspect_ratio'],ax=ax1)

ax1.set_xlabel('Max Length Aspect Ratio', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['max_length_aspect_ratio'],ax=ax2)

ax2.set_xlabel('Max Length Aspect Ratio', fontsize=15)

ax2.set_title("Box Plot")



bins = range(10, 100, 10)

ax3 = sns.distplot(vehicle_df.max_length_aspect_ratio[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.max_length_aspect_ratio[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.max_length_aspect_ratio[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Max Length Aspect Ratio', fontsize=15)

plt.title("Max Length Aspect Ratio vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['max_length_aspect_ratio'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['max_length_aspect_ratio'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_length_aspect_ratio = Q1 - 1.5 * IQR   # lower bound 

UTV_length_aspect_ratio = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('length_aspect_ratio <',LTV_length_aspect_ratio ,'and >',UTV_length_aspect_ratio, ' are outliers')

print('Numerber of outliers in length_aspect_ratio column below the lower whisker =', 

      vehicle_df[vehicle_df['max_length_aspect_ratio'] < (Q1-(1.5*IQR))]['max_length_aspect_ratio'].count())

print('Numerber of outliers in length_aspect_ratio column above the upper whisker =', 

      vehicle_df[vehicle_df['max_length_aspect_ratio'] > (Q3+(1.5*IQR))]['max_length_aspect_ratio'].count())

outlier_columns.append('max_length_aspect_ratio')



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append(LTV_length_aspect_ratio)

upperLowerBound_Disct['length_aspect_ratio_LTV'] = LTV_length_aspect_ratio

upperLowerBound_Disct['length_aspect_ratio_UTV'] = UTV_length_aspect_ratio
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['scatter_ratio'],ax=ax1)

ax1.set_xlabel('Scatter Ratio', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['scatter_ratio'],ax=ax2)

ax2.set_xlabel('Scatter Ratio', fontsize=15)

ax2.set_title("Box Plot")



bins = range(10, 300, 10)

ax3 = sns.distplot(vehicle_df.scatter_ratio[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.scatter_ratio[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.scatter_ratio[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Scatter Ratio', fontsize=15)

plt.title("Scatter Ratio vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['elongatedness'],ax=ax1)

ax1.set_xlabel('Elongatedness', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['elongatedness'],ax=ax2)

ax2.set_xlabel('Elongatedness', fontsize=15)

ax2.set_title("Box Plot")



bins = range(10, 100, 10)

ax3 = sns.distplot(vehicle_df.elongatedness[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.elongatedness[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.elongatedness[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Elongatedness', fontsize=15)

plt.title("Elongatedness vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['pr_axis_rectangularity'],ax=ax1)

ax1.set_xlabel('Axis Rectangularity', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['pr_axis_rectangularity'],ax=ax2)

ax2.set_xlabel('Axis Rectangularity', fontsize=15)

ax2.set_title("Box Plot")



bins = range(10, 50, 10)

ax3 = sns.distplot(vehicle_df.pr_axis_rectangularity[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.pr_axis_rectangularity[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.pr_axis_rectangularity[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Axis Rectangularity', fontsize=15)

plt.title("Axis Rectangularity vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['max_length_rectangularity'],ax=ax1)

ax1.set_xlabel('Max Length Rectangularity', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['max_length_rectangularity'],ax=ax2)

ax2.set_xlabel('Max Length Rectangularity', fontsize=15)

ax2.set_title("Box Plot")



bins = range(100, 300, 10)

ax3 = sns.distplot(vehicle_df.max_length_rectangularity[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.max_length_rectangularity[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.max_length_rectangularity[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Max Length Rectangularity', fontsize=15)

plt.title("Max Length Rectangularity vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['scaled_variance'],ax=ax1)

ax1.set_xlabel('Scaled Variance', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['scaled_variance'],ax=ax2)

ax2.set_xlabel('Scaled Variance', fontsize=15)

ax2.set_title("Box Plot")



bins = range(100, 500, 10)

ax3 = sns.distplot(vehicle_df.scaled_variance[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.scaled_variance[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.scaled_variance[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Scaled Variance', fontsize=15)

plt.title("Scaled Variance vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['scaled_variance'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['scaled_variance'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_scaled_variance = Q1 - 1.5 * IQR   # lower bound 

UTV_scaled_variance = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('scaled_variance <',LTV_scaled_variance ,'and >',UTV_scaled_variance, ' are outliers')

print('Numerber of outliers in scaled_variance column below the lower whisker =', 

      vehicle_df[vehicle_df['scaled_variance'] < (Q1-(1.5*IQR))]['scaled_variance'].count())

print('Numerber of outliers in scaled_variance column above the upper whisker =', 

      vehicle_df[vehicle_df['scaled_variance'] > (Q3+(1.5*IQR))]['scaled_variance'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('scaled_variance')

upperLowerBound_Disct['scaled_variance'] = UTV_scaled_variance
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['scaled_variance_1'],ax=ax1)

ax1.set_xlabel('Scaled Variance_1', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['scaled_variance_1'],ax=ax2)

ax2.set_xlabel('Scaled Variance_1', fontsize=15)

ax2.set_title("Box Plot")



bins = range(100, 1500, 10)

ax3 = sns.distplot(vehicle_df.scaled_variance_1[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.scaled_variance_1[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.scaled_variance_1[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Scaled Variance_1', fontsize=15)

plt.title("Scaled Variance_1 vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['scaled_variance_1'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['scaled_variance_1'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_scaled_variance_1 = Q1 - 1.5 * IQR   # lower bound 

UTV_scaled_variance_1 = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('scaled_variance_1 <',LTV_scaled_variance_1 ,'and >',UTV_scaled_variance_1, ' are outliers')

print('Numerber of outliers in scaled_variance_1 column below the lower whisker =', 

      vehicle_df[vehicle_df['scaled_variance_1'] < (Q1-(1.5*IQR))]['scaled_variance_1'].count())

print('Numerber of outliers in scaled_variance_1 column above the upper whisker =', 

      vehicle_df[vehicle_df['scaled_variance_1'] > (Q3+(1.5*IQR))]['scaled_variance_1'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('scaled_variance_1')

upperLowerBound_Disct['scaled_variance_1'] = UTV_scaled_variance_1
upperLowerBound_Disct
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['scaled_radius_of_gyration'],ax=ax1)

ax1.set_xlabel('Scaled Radius of Gyration', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['scaled_radius_of_gyration'],ax=ax2)

ax2.set_xlabel('Scaled Radius of Gyration', fontsize=15)

ax2.set_title("Box Plot")



bins = range(100, 300, 10)

ax3 = sns.distplot(vehicle_df.scaled_radius_of_gyration[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.scaled_radius_of_gyration[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.scaled_radius_of_gyration[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Scaled Radius of Gyration', fontsize=15)

plt.title("Scaled Radius of Gyration vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['scaled_radius_of_gyration_1'],ax=ax1)

ax1.set_xlabel('Scaled Radius of Gyration_1', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['scaled_radius_of_gyration_1'],ax=ax2)

ax2.set_xlabel('Scaled Radius of Gyration_1', fontsize=15)

ax2.set_title("Box Plot")



bins = range(50, 200, 10)

ax3 = sns.distplot(vehicle_df.scaled_radius_of_gyration_1[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.scaled_radius_of_gyration_1[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.scaled_radius_of_gyration_1[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Scaled Radius of Gyration_1', fontsize=15)

plt.title("Scaled Radius of Gyration_1 vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['scaled_radius_of_gyration_1'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['scaled_radius_of_gyration_1'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_scaled_radius_of_gyration_1 = Q1 - 1.5 * IQR   # lower bound 

UTV_scaled_radius_of_gyration_1 = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('scaled_radius_of_gyration_1 <',LTV_scaled_radius_of_gyration_1 ,'and >',UTV_scaled_radius_of_gyration_1, ' are outliers')

print('Numerber of outliers in scaled_radius_of_gyration_1 column below the lower whisker =', 

      vehicle_df[vehicle_df['scaled_radius_of_gyration_1'] < (Q1-(1.5*IQR))]['scaled_radius_of_gyration_1'].count())

print('Numerber of outliers in scaled_radius_of_gyration_1 column above the upper whisker =', 

      vehicle_df[vehicle_df['scaled_radius_of_gyration_1'] > (Q3+(1.5*IQR))]['scaled_radius_of_gyration_1'].count())





# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('scaled_radius_of_gyration_1')

upperLowerBound_Disct['scaled_radius_of_gyration_1'] = UTV_scaled_radius_of_gyration_1
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['skewness_about'],ax=ax1)

ax1.set_xlabel('Skewness About', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['skewness_about'],ax=ax2)

ax2.set_xlabel('Skewness About', fontsize=15)

ax2.set_title("Box Plot")



bins = range(0, 50, 10)

ax3 = sns.distplot(vehicle_df.skewness_about[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.skewness_about[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.skewness_about[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Skewness About', fontsize=15)

plt.title("Skewness About vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['skewness_about'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['skewness_about'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_skewness_about = Q1 - 1.5 * IQR   # lower bound 

UTV_skewness_about = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('skewness_about <',LTV_skewness_about ,'and >',UTV_skewness_about, ' are outliers')

print('Numerber of outliers in skewness_about column below the lower whisker =', 

      vehicle_df[vehicle_df['skewness_about'] < (Q1-(1.5*IQR))]['skewness_about'].count())

print('Numerber of outliers in skewness_about column above the upper whisker =', 

      vehicle_df[vehicle_df['skewness_about'] > (Q3+(1.5*IQR))]['skewness_about'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('skewness_about')

upperLowerBound_Disct['skewness_about'] = UTV_skewness_about
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['skewness_about_1'],ax=ax1)

ax1.set_xlabel('Skewness About_1', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['skewness_about_1'],ax=ax2)

ax2.set_xlabel('Skewness About_1', fontsize=15)

ax2.set_title("Box Plot")



bins = range(0, 50, 10)

ax3 = sns.distplot(vehicle_df.skewness_about_1[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.skewness_about_1[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.skewness_about_1[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Skewness About_1', fontsize=15)

plt.title("Skewness About_1 vs Class Distribution")

plt.legend()
Q1 =  vehicle_df['skewness_about_1'].quantile(0.25) # 1º Quartile

Q3 =  vehicle_df['skewness_about_1'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_skewness_about_1 = Q1 - 1.5 * IQR   # lower bound 

UTV_skewness_about_1 = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('skewness_about_1 <',LTV_skewness_about_1 ,'and >',UTV_skewness_about_1, ' are outliers')

print('Numerber of outliers in skewness_about_1 column below the lower whisker =', 

      vehicle_df[vehicle_df['skewness_about_1'] < (Q1-(1.5*IQR))]['skewness_about_1'].count())

print('Numerber of outliers in skewness_about_1 column above the upper whisker =', 

      vehicle_df[vehicle_df['skewness_about_1'] > (Q3+(1.5*IQR))]['skewness_about_1'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('skewness_about_1')

upperLowerBound_Disct['skewness_about_1'] = UTV_skewness_about_1
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['skewness_about_2'],ax=ax1)

ax1.set_xlabel('Skewness About_2', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['skewness_about_2'],ax=ax2)

ax2.set_xlabel('Skewness About_2', fontsize=15)

ax2.set_title("Box Plot")



bins = range(100, 300, 10)

ax3 = sns.distplot(vehicle_df.skewness_about_2[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.skewness_about_2[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.skewness_about_2[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Skewness About_2', fontsize=15)

plt.title("Skewness About_2 vs Class Distribution")

plt.legend()
fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(vehicle_df['hollows_ratio'],ax=ax1)

ax1.set_xlabel('Hollows Ratio', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(vehicle_df['hollows_ratio'],ax=ax2)

ax2.set_xlabel('Hollows Ratio', fontsize=15)

ax2.set_title("Box Plot")





#fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))

#ax3.set_xlabel('Distance Circularity', fontsize=15)

bins = range(100, 300, 10)

ax3 = sns.distplot(vehicle_df.hollows_ratio[vehicle_df['class']==0], color='red', kde=False, bins=bins, label='Car')

sns.distplot(vehicle_df.hollows_ratio[vehicle_df['class']==1],ax=ax3, color='blue', kde=False, bins=bins, label="Bus")

sns.distplot(vehicle_df.hollows_ratio[vehicle_df['class']==2],ax=ax3, color='cyan', kde=False, bins=bins, label="van")

ax3.set_xlabel('Hollows Ratio', fontsize=15)

plt.title("Hollows Ratio vs Class Distribution")

plt.legend()
print('These are the columns which have outliers : \n\n',outlier_columns)

print('\n\n',upperLowerBound_Disct)
vehicle_df_new = vehicle_df.copy()
for col_name in vehicle_df_new.columns[:-1]:

    q1 = vehicle_df_new[col_name].quantile(0.25)

    q3 = vehicle_df_new[col_name].quantile(0.75)

    iqr = q3 - q1

    low = q1-1.5*iqr

    high = q3+1.5*iqr

    

    vehicle_df_new.loc[(vehicle_df_new[col_name] < low) | (vehicle_df_new[col_name] > high), col_name] = vehicle_df_new[col_name].median()
plt.figure(figsize=(15,8))

sns.boxplot(data=vehicle_df_new, orient="h", palette="Set2", dodge=False)
vehicle_df_new.shape
vehicle_df_new.corr()
corr_matrix = vehicle_df_new.corr().abs()

high_corr_var=np.where(corr_matrix>0.95)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

high_corr_var
mask = np.zeros_like(vehicle_df_new.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(15,7))

plt.title('Correlation of Attributes', y=1.05, size=19)

sns.heatmap(vehicle_df_new.corr(),vmin=-1, cmap='plasma',annot=True,  mask=mask, fmt='.2f')
# Dropping class variables

X    = vehicle_df_new.drop('class', axis =1)

y    = vehicle_df_new['class']
X.shape
y.shape
# All variables are on same scale, hence we can omit scaling.

# But to standardize the process we will do it here

X_scaled = X.apply(zscore)
plt.figure(figsize=(12,6))

sns.boxplot(data=X_scaled, orient="h", palette="Set2", dodge=False)
covMatrix = np.cov(X_scaled, rowvar=False)

print('Covarinace Matrix Shape:', covMatrix.shape)

print('Covarinace Matrix:\n', covMatrix)
eig_vals, eig_vecs = np.linalg.eig(covMatrix)

tot = sum(eig_vals)

var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)
pca = PCA(n_components=18)

pca.fit(X_scaled)
print(pca.explained_variance_)
print(pca.components_)
print(pca.explained_variance_ratio_)
plt.bar(list(range(1,19)),pca.explained_variance_ratio_,alpha=1, align='center')

plt.ylabel('Variation explained')

plt.xlabel('Eigen Value/Component')

plt.show()

plt.plot(var_exp)
plt.step(list(range(1,19)),np.cumsum(pca.explained_variance_ratio_), where='mid')

plt.ylabel('cummalative of variation explained')

plt.xlabel('Eigen/Components Value')

plt.show()



# Ploting 

plt.bar(range(1, eig_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')

plt.step(range(1, eig_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')

plt.ylabel('Explained Variance Ratio')

plt.xlabel('Principal Components')

plt.legend(loc = 'best')

plt.tight_layout()

plt.show()
pca_eight_components = PCA(n_components=8)

pca_eight_components.fit(X_scaled)
X_scaled_pca_eight_attr = pca_eight_components.transform(X_scaled)

X_scaled_pca_eight_attr.shape
pca_datafram=pd.DataFrame(X_scaled_pca_eight_attr) 
sns.pairplot(pca_datafram, diag_kind = 'kde')
#Raw data spliting

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=7)



#PCA tranfromed data spliting

pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(X_scaled_pca_eight_attr, y, test_size=0.30, random_state=7)

print('-----------------------Origina Data----------------------------- \n')

print('x train data {}'.format(X_train.shape))

print('y train data {}'.format(y_train.shape))

print('x test data  {}'.format(X_test.shape))

print('y test data  {}'.format(y_test.shape))



print('\n\n-----------------------PCA Transformed Data-----------------------------\n')



print('x train data {}'.format(pca_X_train.shape))

print('y train data {}'.format(pca_y_train.shape))

print('x test data  {}'.format(pca_X_test.shape))

print('y test data  {}'.format(pca_y_test.shape))
svm_model = SVC(gamma="scale")

params = {'kernel': ['linear', 'rbf'], 'C':[0.01, 0.1, 0.5, 1]}

gridSearch_model = GridSearchCV(svm_model, param_grid=params, cv=5)

gridSearch_model.fit(X_scaled_pca_eight_attr, y)

print("Best Hyper Surface Parameters found by GridSearch:\n", gridSearch_model.best_params_)
# prepare cross validation and array decleration

seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)

kfold

cv_results = []  # to store cross validation result 

model_names = [] # to store each model name

models_results = [] # to store the model final result with accuracy, precisio, recall, f1-score and cv_results

target_names = ['car', 'bus', 'van']
pca_svm_start_time = time.time()

pca_svm = SVC(C = 1, kernel = 'rbf', gamma= "auto")

pca_svm.fit(pca_X_train, pca_y_train)

pca_svm_computational_time = time.time() - pca_svm_start_time

print('Done in %0.3fs' %(pca_svm_computational_time))



# Predicting for test set

pca_svm_y_predicted            = pca_svm.predict(pca_X_test)

pca_svm_Score                = pca_svm.score(pca_X_test, pca_y_test)

pca_svm_Accuracy        = accuracy_score(pca_y_test, pca_svm_y_predicted)



pca_svm_classification_Report = metrics.classification_report(pca_y_test, pca_svm_y_predicted,  target_names=target_names)

pca_svm_classification_Report_dict = metrics.classification_report(pca_y_test, pca_svm_y_predicted, target_names=target_names, output_dict=True)



pca_svm_precision_weightedAvg   = pca_svm_classification_Report_dict['weighted avg']['precision']

pca_svm_recall_weightedAvg      = pca_svm_classification_Report_dict['weighted avg']['recall']

pca_svm_f1_score_weightedAvg    = pca_svm_classification_Report_dict['weighted avg']['f1-score']



pca_svm_confusion_matrix = metrics.confusion_matrix(pca_y_test, pca_svm_y_predicted)

print('\nPCA SVM: \n', pca_svm_confusion_matrix)



print('\nPCA SVM classification Report : \n', pca_svm_classification_Report)



# Cross Validation 

pca_svm_cross_validation_result = model_selection.cross_val_score(pca_svm, X_scaled_pca_eight_attr, y, cv=kfold, scoring='accuracy')



cv_results.append(pca_svm_cross_validation_result)

model_names.append('PCA-SVM')



pca_svm_model_results = pd.DataFrame([['PCA SVM (RBF)', 'Yes', pca_svm_Accuracy, pca_svm_precision_weightedAvg, pca_svm_recall_weightedAvg,

                                       pca_svm_f1_score_weightedAvg,pca_svm_cross_validation_result.mean(), 

                                       pca_svm_cross_validation_result.std(), round(pca_svm_computational_time,3)]], 

                              columns = ['Model', 'PCA', 'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = pca_svm_model_results

models_results
pca_log_start_time = time.time()

pca_log = LogisticRegression(solver = 'lbfgs', multi_class='auto')

pca_log.fit(pca_X_train, pca_y_train)

pca_log_computational_time = time.time() - pca_log_start_time

print('Done in %0.3fs' %(pca_log_computational_time))



# Predicting for test set

pca_log_y_predicted      = pca_log.predict(pca_X_test)

pca_log_Score            = pca_log.score(pca_X_test, pca_y_test)

pca_log_Accuracy         = accuracy_score(pca_y_test, pca_log_y_predicted)



pca_log_classification_Report = metrics.classification_report(pca_y_test, pca_log_y_predicted, target_names=target_names)

pca_log_classification_Report_dict = metrics.classification_report(pca_y_test, pca_log_y_predicted, target_names=target_names, output_dict=True)



pca_log_precision_weightedAvg   =   pca_log_classification_Report_dict['weighted avg']['precision']

pca_log_recall_weightedAvg      =   pca_log_classification_Report_dict['weighted avg']['recall']

pca_log_f1_score_weightedAvg    =   pca_log_classification_Report_dict['weighted avg']['f1-score']



pca_log_confusion_matrix = metrics.confusion_matrix(pca_y_test, pca_log_y_predicted)



print('\nPCA Logistic Regression: \n', pca_log_confusion_matrix)

print('\nPCA Logistic Regression classification Report : \n', pca_log_classification_Report)



# Cross Validation 

pca_log_cross_validation_result = model_selection.cross_val_score(pca_log, X_scaled_pca_eight_attr, y, cv=kfold, scoring='accuracy')



cv_results.append(pca_log_cross_validation_result)

model_names.append('PCA-LogReg')



pca_log_model_results   = pd.DataFrame([['PCA Logistic Regression', 'Yes', pca_log_Accuracy, pca_log_precision_weightedAvg,

                                         pca_log_recall_weightedAvg, pca_log_f1_score_weightedAvg,

                                       pca_log_cross_validation_result.mean(), 

                                         pca_log_cross_validation_result.std(), round(pca_log_computational_time,3)]], 

                              columns = ['Model', 'PCA', 'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = models_results.append(pca_log_model_results, ignore_index=True)

models_results
pca_gnb_start_time = time.time()

pca_gnb = GaussianNB()

pca_gnb.fit(pca_X_train, pca_y_train)

pca_gnb_computational_time = time.time() - pca_gnb_start_time

print('Done in %0.3fs' %(pca_gnb_computational_time))



# Predicting for test set

pca_gnb_y_predicted          = pca_gnb.predict(pca_X_test)

pca_gnb_Score                = pca_gnb.score(pca_X_test, pca_y_test)

pca_gnb_Accuracy             = accuracy_score(pca_y_test, pca_gnb_y_predicted)



pca_gnb_classification_Report = metrics.classification_report(pca_y_test, pca_gnb_y_predicted,  target_names=target_names)

pca_gnb_classification_Report_dict = metrics.classification_report(pca_y_test, pca_gnb_y_predicted, target_names=target_names, output_dict=True)



pca_gnb_precision_weightedAvg   =   pca_gnb_classification_Report_dict['weighted avg']['precision']

pca_gnb_recall_weightedAvg      =   pca_gnb_classification_Report_dict['weighted avg']['recall']

pca_gnb_f1_score_weightedAvg    =   pca_gnb_classification_Report_dict['weighted avg']['f1-score']



pca_gnb_confusion_matrix = metrics.confusion_matrix(pca_y_test, pca_gnb_y_predicted)



print('\nPCA Naive Bayes: \n', pca_gnb_confusion_matrix)

print('\nPCA Naive Bayes classification Report : \n', pca_gnb_classification_Report)



# Cross Validation 

pca_gnb_cross_validation_result = model_selection.cross_val_score(pca_gnb, X_scaled_pca_eight_attr, y, cv=kfold, scoring='accuracy')



cv_results.append(pca_gnb_cross_validation_result)

model_names.append('PCA-GNB')



pca_gnb_model_results  = pd.DataFrame([['PCA Naive Bayes (Gaussian)', 'Yes', pca_gnb_Accuracy, pca_gnb_precision_weightedAvg,

                                        pca_gnb_recall_weightedAvg,  pca_gnb_f1_score_weightedAvg,

                                        pca_gnb_cross_validation_result.mean(), 

                                        pca_gnb_cross_validation_result.std(), round(pca_gnb_computational_time,3)]], 

                               columns = ['Model', 'PCA', 'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = models_results.append(pca_gnb_model_results, ignore_index=True)

models_results
print('-----------------------Origina Data----------------------------- \n')

print('x train data {}'.format(X_train.shape))

print('y train data {}'.format(y_train.shape))

print('x test data  {}'.format(X_test.shape))

print('y test data  {}'.format(y_test.shape))
svm_model = SVC(gamma="scale")

params = {'kernel': ['linear', 'rbf'], 'C':[0.01, 0.1, 0.5, 1]}

gridSearch_model = GridSearchCV(svm_model, param_grid=params, cv=5)

gridSearch_model.fit(X_train, y_train)

print("Best Hyper Parameters:\n", gridSearch_model.best_params_)
rawData_svm_start_time = time.time()

rawData_svm = SVC(C = 1, kernel = 'rbf', gamma= "auto")

rawData_svm.fit(X_train, y_train)

rawData_svm_computational_time = time.time() - rawData_svm_start_time

print('Done in %0.3fs' %(rawData_svm_computational_time))



# Predicting for test set

rawData_svm_y_predicted     = rawData_svm.predict(X_test)

rawData_svm_Score           = rawData_svm.score(X_test, y_test)

rawData_svm_Accuracy        = accuracy_score(y_test, rawData_svm_y_predicted)



rawData_svm_classification_Report = metrics.classification_report(y_test, rawData_svm_y_predicted,  target_names=target_names)

rawData_svm_classification_Report_dict = metrics.classification_report(y_test, rawData_svm_y_predicted, target_names=target_names, output_dict=True)



rawData_svm_precision_weightedAvg   =   rawData_svm_classification_Report_dict['weighted avg']['precision']

rawData_svm_recall_weightedAvg      =   rawData_svm_classification_Report_dict['weighted avg']['recall']

rawData_svm_f1_score_weightedAvg    =   rawData_svm_classification_Report_dict['weighted avg']['f1-score']



rawData_svm_confusion_matrix = metrics.confusion_matrix(y_test, rawData_svm_y_predicted)



print('\nRawData SVM: \n', rawData_svm_confusion_matrix)

print('\nRawData SVM classification Report : \n', rawData_svm_classification_Report)





# Cross Validation 

rawData_svm_cross_validation_result = model_selection.cross_val_score(rawData_svm, X_scaled, y, cv=kfold, scoring='accuracy')



cv_results.append(rawData_svm_cross_validation_result)

model_names.append('RawData-SVM')



rawData_svm_model_results = pd.DataFrame([['RawData SVM (RBF)', 'No', rawData_svm_Accuracy, rawData_svm_precision_weightedAvg, rawData_svm_recall_weightedAvg,

                                   rawData_svm_f1_score_weightedAvg, rawData_svm_cross_validation_result.mean(), 

                                   rawData_svm_cross_validation_result.std(), round(rawData_svm_computational_time,3)]], 

                              columns = ['Model', 'PCA',  'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = models_results.append(rawData_svm_model_results, ignore_index=True)

models_results
rawData_log_start_time = time.time()

rawData_log = LogisticRegression(solver = 'lbfgs', multi_class='auto')

rawData_log.fit(X_train, y_train)

rawData_log_computational_time = time.time() - rawData_log_start_time

print('Done in %0.3fs' %(rawData_log_computational_time))



# Predicting for test set

rawData_log_y_predicted      = rawData_log.predict(X_test)

rawData_log_Score            = rawData_log.score(X_test, y_test)

rawData_log_Accuracy         = accuracy_score(y_test, rawData_log_y_predicted)



rawData_log_classification_Report = metrics.classification_report(y_test, rawData_log_y_predicted,  target_names=target_names)

rawData_log_classification_Report_dict = metrics.classification_report(y_test, rawData_log_y_predicted, target_names=target_names, output_dict=True)



rawData_log_precision_weightedAvg   =   rawData_log_classification_Report_dict['weighted avg']['precision']

rawData_log_recall_weightedAvg      =   rawData_log_classification_Report_dict['weighted avg']['recall']

rawData_log_f1_score_weightedAvg    =   rawData_log_classification_Report_dict['weighted avg']['f1-score']



rawData_log_confusion_matrix = metrics.confusion_matrix(y_test, rawData_log_y_predicted)



print('\nRawData Logistic Regression: \n', rawData_log_confusion_matrix)

print('\nnRawData Logistic Regression classification Report : \n', rawData_log_classification_Report)



# Cross Validation

rawData_log_cross_validation_result = model_selection.cross_val_score(rawData_log, X_scaled, y, cv=kfold, scoring='accuracy')



cv_results.append(rawData_log_cross_validation_result)

model_names.append('RawData-LogReg')



rawData_log_model_results   = pd.DataFrame([['RawData Logistic Regression', 'No', rawData_log_Accuracy,rawData_log_precision_weightedAvg,

                                     rawData_log_recall_weightedAvg, rawData_log_f1_score_weightedAvg, rawData_log_cross_validation_result.mean(), 

                                     rawData_log_cross_validation_result.std(), round(rawData_log_computational_time, 3)]], 

                              columns = ['Model', 'PCA', 'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = models_results.append(rawData_log_model_results, ignore_index=True)

models_results
rawData_gnb_start_time = time.time()

rawData_gnb = GaussianNB()

rawData_gnb.fit(X_train, y_train)

rawData_gnb_computational_time = time.time() - rawData_gnb_start_time

print('Done in %0.3fs' %(rawData_gnb_computational_time))



# Predicting for test set

rawData_gnb_y_predicted          = rawData_gnb.predict(X_test)

rawData_gnb_Score                = rawData_gnb.score(X_test, y_test)

rawData_gnb_Accuracy             = accuracy_score(y_test, rawData_gnb_y_predicted)



rawData_gnb_classification_Report = metrics.classification_report(y_test, rawData_gnb_y_predicted,  target_names=target_names)

rawData_gnb_classification_Report_dict = metrics.classification_report(y_test, rawData_gnb_y_predicted, target_names=target_names, output_dict=True)



rawData_gnb_precision_weightedAvg   =   rawData_gnb_classification_Report_dict['weighted avg']['precision']

rawData_gnb_recall_weightedAvg      =   rawData_gnb_classification_Report_dict['weighted avg']['recall']

rawData_gnb_f1_score_weightedAvg    =   rawData_gnb_classification_Report_dict['weighted avg']['f1-score']



rawData_gnb_confusion_matrix = metrics.confusion_matrix(y_test, rawData_gnb_y_predicted)



print('\nRawData GNB: \n', rawData_gnb_confusion_matrix)

print('\nRawData GNB classification Report : \n', rawData_gnb_classification_Report)



# Cross Validation 

rawData_gnb_cross_validation_result = model_selection.cross_val_score(rawData_gnb, X_scaled, y, cv=kfold, scoring='accuracy')



cv_results.append(rawData_gnb_cross_validation_result)

model_names.append('RawData-GNB')



rawData_gnb_model_results  = pd.DataFrame([['RawData Naive Bayes (Gaussian)', 'No', rawData_gnb_Accuracy, rawData_gnb_precision_weightedAvg, 

                                    rawData_gnb_recall_weightedAvg, rawData_gnb_f1_score_weightedAvg, rawData_gnb_cross_validation_result.mean(),

                                    rawData_gnb_cross_validation_result.std(), round(rawData_gnb_computational_time,3)]], 

                              columns = ['Model', 'PCA', 'Accuracy', 'Precision', 'Recall',

                                         'F1-Score', 'CV Mean', 'CV Std Deviation', 'Execution Time in Seconds'])



models_results = models_results.append(rawData_gnb_model_results, ignore_index=True)

models_results
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (25,15))





# Confusion metrix using PCA transformed data

cm    = metrics.confusion_matrix(pca_y_test, pca_svm_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[0,0])

axs[0,0].tick_params(labelsize=20)

axs[0,0].set_xlabel('Predicted Labels', fontsize=20);

axs[0,0].set_ylabel('Actual Labels', fontsize=20); 

axs[0,0].set_title('PCA SVM(rbf)', fontsize=20); 



cm    = metrics.confusion_matrix(pca_y_test, pca_log_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[0,1])

axs[0,1].tick_params(labelsize=20)

axs[0,1].set_xlabel('Predicted Labels', fontsize=20);

axs[0,1].set_ylabel('Actual Labels', fontsize=20); 

axs[0,1].set_title('PCA Logistic Regression', fontsize=20); 



cm    = metrics.confusion_matrix(pca_y_test, pca_gnb_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[0,2])

axs[0,2].tick_params(labelsize=20)

axs[0,2].set_xlabel('Predicted Labels', fontsize=20);

axs[0,2].set_ylabel('Actual Labels', fontsize=20); 

axs[0,2].set_title('PCA Naive Bayes', fontsize=20);





cm    = metrics.confusion_matrix(y_test, rawData_svm_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[1,0])

axs[1,0].tick_params(labelsize=20)

axs[1,0].set_xlabel('Predicted Labels', fontsize=20);

axs[1,0].set_ylabel('Actual Labels', fontsize=20); 

axs[1,0].set_title('SVM(rbf)', fontsize=20);





cm    = metrics.confusion_matrix(y_test, rawData_log_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[1,1])

axs[1,1].tick_params(labelsize=20)

axs[1,1].set_xlabel('Predicted Labels', fontsize=20);

axs[1,1].set_ylabel('Actual Labels', fontsize=20); 

axs[1,1].set_title('Logistic Regression', fontsize=20);



cm    = metrics.confusion_matrix(y_test, rawData_gnb_y_predicted, labels=[0, 1, 2])

df_cm = pd.DataFrame(cm, index = [i for i in ["Car","Bus", "Van"]],

                  columns = [i for i in ["Car","Bus", "Van"]])

sns.heatmap(df_cm, annot=True , annot_kws={'size': 15}, fmt='g', ax = axs[1,2])

axs[1,2].tick_params(labelsize=20)

axs[1,2].set_xlabel('Predicted Labels', fontsize=20);

axs[1,2].set_ylabel('Actual Labels', fontsize=20); 

axs[1,2].set_title('Naive Bayes', fontsize=20);



plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.show()
sns.set(rc={'figure.figsize':(16,5)})

sns.boxplot(model_names,cv_results)
y_classes = label_binarize(df['class'], classes=[0, 1, 2])

n_classes = y_classes.shape[1]



# shuffle and split training and test sets

X_train__, X_test__, y_train__, y_test__ = train_test_split(X_scaled_pca_eight_attr, y_classes, test_size=0.30, random_state=7)



# Learn to predict each class against the other

classifier = OneVsRestClassifier(SVC(gamma="scale", kernel='rbf', probability=True,random_state=7))

y_score = classifier.fit(X_train__, y_train__).decision_function(X_test__)



lw = 2 #linewidth

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test__[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test__.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr





roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVM With PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()





#-------------------------------------------# WITHOUT PCA #-----------------------------------------------------#



X_train_, X_test_, y_train_, y_test_ = train_test_split(X_scaled, y_classes, test_size=0.30, random_state=7)

# Learn to predict each class against the other

classifier = OneVsRestClassifier(SVC(gamma="scale", kernel='rbf', probability=True,random_state=7))

y_score = classifier.fit(X_train_, y_train_).decision_function(X_test_)

lw = 2    #linewi

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVM Without PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()



#*************************************Logistic Regression**************************************************************************



#------------------------------------------# With PCA #-------------------------------------------------------------

# Learn to predict each class against the other

classifier = OneVsRestClassifier(LogisticRegression(C=1.0,solver = 'lbfgs', multi_class='auto'))

y_score = classifier.fit(X_train__, y_train__).decision_function(X_test__)



lw = 2 #linewidth

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test__[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test__.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr





roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression With PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()

#-------------------------------------------# WITHOUT PCA #-----------------------------------------------------#

# Learn to predict each class against the other

classifier = OneVsRestClassifier(LogisticRegression(C=1.0,solver = 'lbfgs', multi_class='auto'))

y_score = classifier.fit(X_train_, y_train_).decision_function(X_test_)

lw = 2    #linewi

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression Without PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()



#*************************************Gaussian Naive Bayes**************************************************************************



#--------------------------------------# Gaussian Naive Bayes With PCA #-------------------------------------------------------------

# Learn to predict each class against the other

classifier = OneVsRestClassifier(GaussianNB())

y_score = classifier.fit(X_train__, y_train__).predict(X_test__)



lw = 2 #linewidth

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test__[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test__.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr





roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gaussian Naive Bayes With PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()





#-------------------------------------------# Gaussian Naive WITHOUT PCA #-----------------------------------------------------#



X_train_, X_test_, y_train_, y_test_ = train_test_split(X_scaled, y_classes, test_size=0.30, random_state=7)

# Learn to predict each class against the other

classifier = OneVsRestClassifier(GaussianNB())

y_score = classifier.fit(X_train_, y_train_).predict(X_test_)

lw = 2    #linewi

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_[:, i], y_score[:, i])

    roc_auc[i] = metrics.auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_.ravel(), y_score.ravel())

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gaussian Naive Bayes Without PCA - Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()

models_results[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std Deviation']] = models_results[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std Deviation']].applymap(lambda x: "{0:.2f}".format(x*100))                               

models_results