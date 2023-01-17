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
# import necessary libraries



# File read and EDA(Data Cleansing & Transformations)

import numpy as np  

import pandas as pd 



# EDA Visualization

import matplotlib.pyplot as plt

import seaborn as sns
mc_data = pd.read_json('../input/clothing-fit-dataset-for-size-recommendation/modcloth_final_data.json', lines=True)

mc_data.head() # displays first 5 records in the dataframe
mc_data.columns = ['item_id', 'waist', 'mc_size', 'quality', 'cup_size', 'hips', 'bra_size', 'category', 'bust', 'height', 'user_name', 'length', 'fit', 'user_id', 'shoe_size', 'shoe_width', 'review_summary', 'review_test']
mc_data.info()
missing_data_sum = mc_data.isnull().sum()

missing_data = pd.DataFrame({'total_missing_values': missing_data_sum,'percentage_of_missing_values': (missing_data_sum/mc_data.shape[0])*100})

missing_data
mc_data.dtypes
mc_data.nunique()
def countplot(indipendent_features):

  plt.figure(figsize=(25, 25))

  for loc, feature in enumerate(indipendent_features):

    ax = plt.subplot(3, 4, loc+1)

    ax.set_xlabel('{}'.format(feature), fontsize=10)

    chart = sns.countplot(mc_data[feature])

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

  return None
uniques_data = ['quality', 'cup_size', 'bra_size', 'category', 'length', 'fit',  'shoe_size', 'shoe_width', 'height', 'bust', 'mc_size']

countplot(uniques_data)
# replacing bust unformatted value with mean 38 which is taken from the values 37 & 39 

mc_data.at[mc_data[mc_data.bust == '37-39'].index[0],'bust'] = '38'
def height_in_cms(ht):

  if ht.lower() != 'nan':

    ht = ht.replace('ft','').replace('in', '')

    h_ft = int(ht.split()[0])

    if len(ht.split()) > 1:

      h_inch = int(ht.split()[1])

    else:

      h_inch = 0

    h_inch += h_ft * 12

    h_cm = round(h_inch * 2.54, 1)

    return h_cm



mc_data.height = mc_data.height.astype(str).apply(height_in_cms)

mc_data.head()
mc_data.height.fillna(value=mc_data.height.mean(), inplace=True)

mc_data.height.isnull().sum()
def plot_outlier(feature):

  plt.figure(figsize=(25, 6))

  ax = sns.boxplot(x=feature, linewidth=2.5)

plot_outlier(mc_data.height)
def get_outliers_range(datacolumn):

  sorted(datacolumn)

  Q1,Q3 = np.percentile(datacolumn , [25,75])

  IQR = Q3 - Q1

  lower_range = Q1 - (1.5 * IQR)

  upper_range = Q3 + (1.5 * IQR)

  return lower_range,upper_range


ht_lower_range,ht_upper_range = get_outliers_range(mc_data.height)

ht_lower_range,ht_upper_range
mc_data[(mc_data.height < ht_lower_range) | (mc_data.height > ht_upper_range)]
mc_df = mc_data.drop(mc_data[(mc_data.height < ht_lower_range) | (mc_data.height > ht_upper_range)].index)



mc_df.reset_index(drop=True, inplace=True)

mc_df.shape
plot_outlier(mc_df.height)
def plot_dist(df, indipendent_features):

  plt.figure(figsize=(25, 20))

  for loc, feature in enumerate(indipendent_features):

    ax = plt.subplot(3, 3, loc+1)

    sns.distplot(df[feature]) # you can try histplot as well

  return None
plot_dist(mc_data, ['height', 'waist', 'mc_size', 'quality', 'hips', 'bra_size', 'shoe_size'])
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)



# finding imputation using other features (it will take couple of minutes to complete the execution)

mc_data_knn_ind_features = mc_df[['waist', 'hips', 'bra_size', 'bust', 'height', 'shoe_size']]



df_filled = imputer.fit_transform(mc_data_knn_ind_features)





knn_numeric_imputations = pd.DataFrame(data=df_filled, columns=['waist', 'hips', 'bra_size', 'bust', 'height', 'shoe_size'])





# remove the existing numeric columns (waist, height, hips, bra_size, bust, shoe_size ) from the main dataframe and concatenate  with knn imputed data

#mc_df = mc_data

mc_new_df = mc_df.drop(['waist', 'hips', 'bra_size', 'bust', 'height', 'shoe_size'], axis=1)





# concat the imputations data with mc data frame

mc = pd.concat([mc_new_df, knn_numeric_imputations], axis=1)

mc.isnull().sum()
plot_outlier(mc.shoe_size)
ss_lower_range,ss_upper_range = get_outliers_range(mc.shoe_size)

#print(ss_lower_range,ss_upper_range)



mc.drop(mc[(mc.shoe_size < ss_lower_range) | (mc.shoe_size > ss_upper_range)].index, axis=0, inplace=True) # found 390 observations 

plot_outlier(mc.shoe_size)
def convert_cup_size_to_cms(cup_size_code):

  if cup_size_code == 'aa':

    return 10, 11

  if cup_size_code == 'a':

    return 12, 13

  if cup_size_code == 'b':

    return 14, 15

  if cup_size_code == 'c':

    return 16, 17

  if cup_size_code == 'd':

    return 18, 19

  if cup_size_code == 'dd/e':

    return 20, 21

  if cup_size_code == 'ddd/f':

    return 22, 23

  if cup_size_code == 'dddd/g':

    return 24, 25

  if cup_size_code == 'h':

    return 26, 27

  if cup_size_code == 'i':

    return 28, 29

  if cup_size_code == 'j':

    return 30, 31

  if cup_size_code == 'k':

    return 32, 33 

  else:

    return str('unknown')
mc['cup_size_in_cms'] = mc.cup_size.apply(convert_cup_size_to_cms)

mc.head()
def split_cup_size_data(data, index):

  if data.lower() == 'unknown':

    return 0

  value = data.replace('(','').replace(')','').replace(',','')

  return value.split()[index]



mc['cup_size_start_in_cms'] =  mc.cup_size_in_cms.astype(str).apply(lambda x : split_cup_size_data(x, 0))

mc['cup_size_end_in_cms'] =  mc.cup_size_in_cms.astype(str).apply(lambda x : split_cup_size_data(x, 1))

mc.head()
mc['cup_size_start_in_cms'] = mc.cup_size_start_in_cms.astype('int')

mc['cup_size_end_in_cms'] = mc.cup_size_end_in_cms.astype('int')





# missing values imputation with mean

mc['cup_size_start_in_cms']  = mc.cup_size_start_in_cms.mask(mc.cup_size_start_in_cms==0).fillna(value=mc.cup_size_start_in_cms.mean())

mc['cup_size_end_in_cms']  = mc.cup_size_end_in_cms.mask(mc.cup_size_end_in_cms==0).fillna(value=mc.cup_size_end_in_cms.mean())
mc[mc.cup_size.isnull()]
# drop the columns which are used for reference

mc = mc.drop(['cup_size', 'cup_size_in_cms'], axis = 1)

mc.reset_index(drop=True,  inplace=True)
def countplot_wrt_target(indipendent_features, df):

  plt.figure(figsize=(28, 10))

  for loc, feature in enumerate(indipendent_features):

    ax = plt.subplot(1, 3, loc+1)

    ax.set_xlabel('{}'.format(feature), fontsize=10)

    chart = sns.countplot(x=df[feature], hue=df.fit)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

  return None

countplot_wrt_target(['category', 'length', 'quality'], mc)
# fill NaN with average shoe width category (this is just an assumption)

mc.shoe_width = mc.shoe_width.fillna('average')
# Use above chart to convert shoe width data such as 'wide','average','narrow' to inches

mc['shoe_width_in_inches'] = np.where(((mc.shoe_size >= 5) & (mc.shoe_size < 5.5)) & (mc.shoe_width == 'narrow') , 2.81, 

np.where(((mc.shoe_size >= 5) & (mc.shoe_size < 5.5)) & (mc.shoe_width == 'average') , 3.19, 

np.where(((mc.shoe_size >= 5) & (mc.shoe_size < 5.5)) & (mc.shoe_width == 'wide') , 3.56,

np.where(((mc.shoe_size >= 5.5) & (mc.shoe_size < 6)) & (mc.shoe_width == 'narrow') , 2.87, 

np.where(((mc.shoe_size >= 5.5) & (mc.shoe_size < 6)) & (mc.shoe_width == 'average') , 3.25, 

np.where(((mc.shoe_size >= 5.5) & (mc.shoe_size < 6)) & (mc.shoe_width == 'wide') , 3.62, 

np.where(((mc.shoe_size >= 6) & (mc.shoe_size < 6.5)) & (mc.shoe_width == 'narrow') , 2.94, 

np.where(((mc.shoe_size >= 6) & (mc.shoe_size < 6.5)) & (mc.shoe_width == 'average') , 3.31, 

np.where(((mc.shoe_size >= 6) & (mc.shoe_size < 6.5)) & (mc.shoe_width == 'wide') , 3.69,

np.where(((mc.shoe_size >= 6.5) & (mc.shoe_size < 7)) & (mc.shoe_width == 'narrow') , 3, 

np.where(((mc.shoe_size >= 6.5) & (mc.shoe_size < 7)) & (mc.shoe_width == 'average') , 3.37, 

np.where(((mc.shoe_size >= 6.5) & (mc.shoe_size < 7)) & (mc.shoe_width == 'wide') , 3.75,

np.where(((mc.shoe_size >= 7) & (mc.shoe_size < 7.5)) & (mc.shoe_width == 'narrow') , 3.06, 

np.where(((mc.shoe_size >= 7) & (mc.shoe_size < 7.5)) & (mc.shoe_width == 'average') , 3.44, 

np.where(((mc.shoe_size >= 7) & (mc.shoe_size < 7.5)) & (mc.shoe_width == 'wide') , 3.81, 

np.where(((mc.shoe_size >= 7.5) & (mc.shoe_size < 8)) & (mc.shoe_width == 'narrow') , 3.12, 

np.where(((mc.shoe_size >= 7.5) & (mc.shoe_size < 8)) & (mc.shoe_width == 'average') , 3.5, 

np.where(((mc.shoe_size >= 7.5) & (mc.shoe_size < 8)) & (mc.shoe_width == 'wide') , 3.87, 

np.where(((mc.shoe_size >= 8) & (mc.shoe_size < 8.5)) & (mc.shoe_width == 'narrow') , 3.19, 

np.where(((mc.shoe_size >= 8) & (mc.shoe_size < 8.5)) & (mc.shoe_width == 'average') , 3.56, 

np.where(((mc.shoe_size >= 8) & (mc.shoe_size < 8.5)) & (mc.shoe_width == 'wide') , 3.94, 

np.where(((mc.shoe_size >= 8.5) & (mc.shoe_size < 9)) & (mc.shoe_width == 'narrow') , 3.25, 

np.where(((mc.shoe_size >= 8.5) & (mc.shoe_size < 9)) & (mc.shoe_width == 'average') , 3.62, 

np.where(((mc.shoe_size >= 8.5) & (mc.shoe_size < 9)) & (mc.shoe_width == 'wide') , 4, 

np.where(((mc.shoe_size >= 9) & (mc.shoe_size < 9.5)) & (mc.shoe_width == 'narrow') , 3.37, 

np.where(((mc.shoe_size >= 9) & (mc.shoe_size < 9.5)) & (mc.shoe_width == 'average') , 3.69, 

np.where(((mc.shoe_size >= 9) & (mc.shoe_size < 9.5)) & (mc.shoe_width == 'wide') , 4.06, 

np.where(((mc.shoe_size >= 9.5) & (mc.shoe_size < 10)) & (mc.shoe_width == 'narrow') , 3.37, 

np.where(((mc.shoe_size >= 9.5) & (mc.shoe_size < 10)) & (mc.shoe_width == 'average') , 3.75, 

np.where(((mc.shoe_size >= 9.5) & (mc.shoe_size < 10)) & (mc.shoe_width == 'wide') , 4.12, 

np.where(((mc.shoe_size >= 10) & (mc.shoe_size < 10.5)) & (mc.shoe_width == 'narrow') , 3.44, 

np.where(((mc.shoe_size >= 10) & (mc.shoe_size < 10.5)) & (mc.shoe_width == 'average') , 3.75, 

np.where(((mc.shoe_size >= 10) & (mc.shoe_size < 10.5)) & (mc.shoe_width == 'wide') , 4.19, 

np.where(((mc.shoe_size >= 10.5) & (mc.shoe_size < 11)) & (mc.shoe_width == 'narrow') , 3.5, 

np.where(((mc.shoe_size >= 10.5) & (mc.shoe_size < 11)) & (mc.shoe_width == 'average') , 3.87, 

np.where(((mc.shoe_size >= 10.5) & (mc.shoe_size < 11)) & (mc.shoe_width == 'wide') , 4.19, 

np.where(((mc.shoe_size >= 11) & (mc.shoe_size < 12)) & (mc.shoe_width == 'narrow') , 3.56, 

np.where(((mc.shoe_size >= 11) & (mc.shoe_size < 12)) & (mc.shoe_width == 'average') , 3.94, 

np.where(((mc.shoe_size >= 11) & (mc.shoe_size < 12)) & (mc.shoe_width == 'wide') , 4.19,

np.nan)))))))))))))))))))))))))))))))))))))))
# drop the refrence colum shoe_width

mc.drop(['shoe_width'], axis=1, inplace=True)
# lets replace NaN values with unknown for the feature length

mc.length = mc.length.fillna('unknown')
# apply one hot encoding using dummies



length_dummies  = pd.get_dummies(mc['length'])

length_dummies.columns = ['just_right','slightly_long','very_short','slightly_short','very_long', 'length_unkown']



category_dummies  = pd.get_dummies(mc['category'])

category_dummies.columns = ['new','dresses','wedding','sale','tops', 'bottoms','outerwear']



model_input_df = pd.concat([mc, length_dummies,category_dummies], axis = 1)

model_input_df.drop(['length'], axis=1, inplace=True)

model_input_df.drop(['category'], axis=1, inplace=True)



# target variable 

fit = {'small':0, 'fit':1, 'large':2}

model_input_df['fit'] = model_input_df['fit'].map(fit)

# since there is no value add to the features like item_id , user_id and user_name



model_input_df.drop(['item_id'], axis=1, inplace=True)



model_input_df.drop(['user_id'], axis=1, inplace=True)



model_input_df.drop(['user_name'], axis=1, inplace=True)

model_input_df.head()