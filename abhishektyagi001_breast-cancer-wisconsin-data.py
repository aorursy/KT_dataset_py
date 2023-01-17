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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

import time

import itertools

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

%matplotlib inline
h2o.init()
data_df = h2o.import_file("../input/breast-cancer-wisconsin-data/data.csv", destination_frame="data_df")
data_df.describe()
data_df.describe(1)
df_group=data_df.group_by("diagnosis").count()

df_group.get_frame()
features = [f for f in data_df.columns if f not in ['id', 'diagnosis', 'C33']]



i = 0

t0 = data_df[data_df['diagnosis'] == 'M'].as_data_frame()

t1 = data_df[data_df['diagnosis'] == 'B'].as_data_frame()



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6,5,figsize=(16,24))



for feature in features:

    i += 1

    plt.subplot(6,5,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Malignant")

    sns.kdeplot(t1[feature], bw=0.5,label="Benign")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();

    
plt.figure(figsize=(16,16))

corr = data_df[features].cor().as_data_frame()

corr.index = features

sns.heatmap(corr, annot = True, cmap='YlGnBu', linecolor="white", vmin=-1, vmax=1, cbar_kws={"orientation": "horizontal"})

plt.title("Correlation Heatmap for the features (excluding id, C33 & diagnosis)", fontsize=14)

plt.show()
train_df, valid_df, test_df = data_df.split_frame(ratios=[0.6,0.2], seed=2018)

target = "diagnosis"

train_df[target] = train_df[target].asfactor()

valid_df[target] = valid_df[target].asfactor()

test_df[target] = test_df[target].asfactor()

print("Number of rows in train, valid and test set : ", train_df.shape[0], valid_df.shape[0], test_df.shape[0])
# define the predictor list - it will be the same as the features analyzed previously

predictors = features

# initialize the H2O GBM 

gbm = H2OGradientBoostingEstimator()

# train with the initialized model

gbm.train(x=predictors, y=target, training_frame=train_df)
gbm.summary()
print(gbm.model_performance(valid_df))
gbm.varimp_plot()
pred_val = list(gbm.predict(test_df[predictors])[0])

true_val = list(test_df[target])

prediction_acc = np.mean(pred_val == true_val)

print("Prediction accuracy: ", prediction_acc)