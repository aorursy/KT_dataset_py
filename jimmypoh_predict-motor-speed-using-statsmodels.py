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
import seaborn as sns

import matplotlib.pyplot as plt



motor = pd.read_csv("/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv")
motor.describe()
# drop column "profile_id"

motor.drop("profile_id", axis=1, inplace=True)
motor.info()
motor.isnull().sum()

# no missing values
###################################### Explonatory Data Analysis (EDA)#####################



plt.figure(figsize=(16,8))

sns.heatmap(motor.corr(), annot=True)

# it seems that "motor_speed" has moderate strong correlation with "u_q" and "i_d"
plt.figure(figsize=(16,8))

motor["motor_speed"].plot.hist(bins=30)

# it seems that there is high frequency at around -1.25
plt.figure(figsize=(16,8))

sns.scatterplot(x="u_q", y="motor_speed", data=motor)

# it seems that increase in voltage q-component will increase motor speed as well.
plt.figure(figsize=(16,8))

sns.scatterplot(x="i_d", y="motor_speed", data=motor)

# it seems that decrease in current d-component will increase motor speed.
######################################### Feature Engineer ##################################

# check for outliers



plt.figure(figsize=(16,8))

sns.boxplot(motor["ambient"])

# there are outliers at both ends

# there are extreme outliers at lower end
plt.figure(figsize=(16,8))

sns.boxplot(motor["coolant"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["u_d"])

# there are outliers at upper end
plt.figure(figsize=(16,8))

sns.boxplot(motor["u_q"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["motor_speed"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["torque"])

# there are outliers at both ends
plt.figure(figsize=(16,8))

sns.boxplot(motor["i_d"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["i_q"])

# there are outliers at both ends
plt.figure(figsize=(16,8))

sns.boxplot(motor["pm"])

# no outliers at upper end
plt.figure(figsize=(16,8))

sns.boxplot(motor["stator_yoke"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["stator_tooth"])

# no outliers
plt.figure(figsize=(16,8))

sns.boxplot(motor["stator_winding"])

# no outliers present
# will try to keep those outliers within interquantile proximity rule

motor["ambient"] = np.where(motor["ambient"] < -2.528475, -2.528475, motor["ambient"])

motor["ambient"] = np.where(motor["ambient"] > 2.615765, 2.615765, motor["ambient"])



motor["u_d"] = np.where(motor["u_d"] > 2.135766, 2.135766, motor["u_d"])



motor["torque"] = np.where(motor["torque"] < -1.488049, -1.488049, motor["torque"])

motor["torque"] = np.where(motor["torque"] > 1.768303, 1.768303, motor["torque"])



motor["i_q"] = np.where(motor["i_q"] < -1.3920625, -1.3920625, motor["i_q"])

motor["i_q"] = np.where(motor["i_q"] > 1.6340535, 1.6340535, motor["i_q"])



motor["pm"] = np.where(motor["pm"] > 1.8154845, 1.8154845, motor["pm"])
# check for constant variable 



from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold = 0)



sel.fit(motor)

[feature for feature in motor.columns if feature not in motor.columns[sel.get_support()]]

# return empty list indicates there is no feature with constant value
# check for quasi_constant variable



sel = VarianceThreshold(threshold = 0.01)



sel.fit(motor)

[feature for feature in motor.columns if feature not in motor.columns[sel.get_support()]]

# return empty list indicates there is no quasi_constant variable
# check for duplicated features



duplicated_feat = []

for i in range(0, len(motor.columns)-1):

    col_1 = motor.columns[i]

    col_2 = motor.columns[i+1]

    if motor[col_1].equals(motor[col_2]):

        duplicated_feat.append(motor[col_2])



duplicated_feat

# empty list indicates there is no duplicated feature
# check for multicollinearity



corrmat = motor.corr()

corrmat = corrmat.abs().unstack()

corrmat.sort_values(ascending=False, inplace=True)

corrmat = corrmat[corrmat < 1]

corrmat = corrmat[corrmat >= 0.8]

corrmat = pd.DataFrame(corrmat).reset_index()

corrmat.columns = ["feature1", "feature2", "corr"]



grouped_feature_ls = []

correlated_groups = []



for feature in corrmat["feature1"].unique():

    if feature not in grouped_feature_ls:

        correlated_block = corrmat[corrmat["feature1"] == feature]

        grouped_feature_ls = grouped_feature_ls + list(correlated_block["feature2"].unique()) + [feature]

        correlated_groups.append(correlated_block)
for group in correlated_groups:

    print(group)

# it seems that there are multicollinearity issue.
corrmat_motor = motor.corr()

corrmat_motor.index = motor.columns

corrmat_motor = corrmat_motor["motor_speed"]

corrmat_motor = corrmat_motor.abs().sort_values(ascending=False)
corrmat_motor
# among "torque", "i_q" and "u_d", it seems that "u_d" has higher correlation with motor_speed.

# I will drop both "torque" and "i_q".



# among "stator_tooth", "stator_winding" and "stator_yoke", it seems that "stator_winding" has higher correlation with motor_speed.

# I will drop both "stator_tooth" and "stator_yoke".
motor = motor.drop(["torque","i_q","stator_tooth","stator_yoke"], axis=1)
# split the data into predictors and output

X = motor.drop("motor_speed", axis=1)

Y = motor["motor_speed"]
# split the data into train and test

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
################################## Model Building ########################################

from statsmodels.regression.linear_model import OLS



model = OLS(Y_train, X_train).fit()

model.summary()

# adjusted r_squared is 0.9 which is quite good

# all features have p-values less than 0.05
# evaluate the model

from sklearn.metrics import mean_squared_error



pred = model.predict(X_test)

mean_squared_error(Y_test, pred)
# As a conclusion, the data sets contain significant variables to predict motor_speed.