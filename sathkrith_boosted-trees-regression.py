# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("/kaggle/input/singapore-airbnb/listings.csv")
df.info()

df.isna().sum()
trun_df = df.drop(["id","name","host_name","last_review","host_id"],axis=1)

trun_df.head()
from scipy import stats

outlier = (np.abs(stats.zscore(trun_df["price"]))<0.7)

outlier_ix = np.where(outlier==False)

clean_df = trun_df.drop(index=outlier_ix[0])
clean_df.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.kdeplot(clean_df["price"])
clean_df["reviews_per_month"] = clean_df["reviews_per_month"].fillna(0)
clean_df[["neighbourhood_group","neighbourhood","room_type"]].nunique()
clean_df["neighbourhood_group"].value_counts()
clean_df["neighbourhood"].value_counts()
clean_df.groupby("neighbourhood_group")["neighbourhood"].value_counts()
fig, ax = plt.subplots(figsize=(25,8))

sns.boxplot(y="price",x="neighbourhood", hue="neighbourhood_group",data=clean_df)
sns.boxplot(y="price",x="neighbourhood_group",showfliers=False, data=clean_df)
clean_df["room_type"].value_counts()
sns.boxplot(y="price",x="room_type",data=clean_df,showfliers=False)
sns.pairplot(clean_df,hue="room_type", 

            kind="scatter",diag_kind="hist");

clean_df = clean_df.drop("availability_365",axis=1)
clean_df.describe()
cols_to_noramilze = ["latitude", "longitude", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]



clean_df[cols_to_noramilze] = (clean_df[cols_to_noramilze]-clean_df[cols_to_noramilze].mean())/clean_df[cols_to_noramilze].std()
clean_df.head()
import tensorflow as tf



CATEGORICAL_COLS = ["room_type","neighbourhood_group","neighbourhood"]

NUMERICAL_COLS = ["latitude", "longitude", "minimum_nights", "number_of_reviews", "reviews_per_month"]



feature_cols = []



for feature_name in CATEGORICAL_COLS:

    vocab = clean_df[feature_name].unique()

    feature_cols.append(tf.feature_column.indicator_column(

      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,

                                                 vocab)))

    

for feature_name in NUMERICAL_COLS:

    feature_cols.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float64))



feature_cols
from sklearn.model_selection import train_test_split

y = clean_df.pop("price")

dftrain, dfeval, y_train, y_eval = train_test_split(clean_df,y,test_size=0.2,random_state=0)

print(dftrain.head())
NUM_EXAMPLES = len(y_train)



def make_input_fn(data_df, label_df, num_epochs=None, shuffle=True, batch_size=32):

  def input_function():

    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if shuffle:

      ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds

  return input_function



train_input_fn = make_input_fn(dftrain, y_train)

eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
dfeval.head()
b_reg = tf.estimator.BoostedTreesRegressor(feature_cols,n_batches_per_layer=1,n_trees=50, max_depth=3,center_bias=True)

b_reg.train(train_input_fn,max_steps=200)

result = b_reg.evaluate(eval_input_fn)

pd.Series(result).to_frame()
pred_dicts = list(b_reg.experimental_predict_with_explanations(eval_input_fn))
df_dfc = pd.DataFrame([pred["dfc"] for pred in pred_dicts])

df_dfc.describe().T
importances = b_reg.experimental_feature_importances(normalize=True)

df_imp = pd.Series(importances)



# Visualize importances.

N = 8

ax = (df_imp.iloc[0:N][::-1]

    .plot(kind='barh',

          title='Gain feature importances',

          figsize=(10, 6)))

ax.grid(False, axis='y')