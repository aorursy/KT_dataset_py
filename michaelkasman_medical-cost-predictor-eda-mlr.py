import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# import dataset using Pandas

data = pd.read_csv('../input/insurance/insurance.csv')



# check if any columns have NaN values

data.isnull().sum()
# output first five rows of the dataset using the ".head()" function

data.head()
data = pd.get_dummies(data)
data.head()
# calculate variable correlations in regards to 'charges'

corr = data.corr()['charges'].sort_values()
# display correlation values

corr
corr = data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(250,10,as_cmap=True),

            square=True,annot=True,ax=ax)
tmp_data = pd.read_csv('../input/insurance/insurance.csv')



# Descriptive statistics smoker

statistics_smoker = tmp_data[tmp_data['smoker'] == 'yes'].describe()

statistics_smoker.rename(columns=lambda x: x + '_smoker', inplace=True)



# Descriptive statistics non-smoker

statistics_non_smoker = tmp_data[tmp_data['smoker'] == 'no'].describe()

statistics_non_smoker.rename(columns=lambda x: x + '_non_smoker', inplace=True)



# Dataframe that contains statistics for both male and female

statistics = pd.concat([statistics_smoker, statistics_non_smoker], axis=1)

statistics
from statistics import mode 



plt.style.use('ggplot')



# histogram of ages

data.age.plot(kind='hist', color='orange', edgecolor='black', figsize=(10,7))

plt.title('Distribution of Age', size=24)

plt.xlabel('Age', size=18)

plt.ylabel('Frequency', size=18)



# find most frequent age

mode_age = mode(data.age)

print('Mode of Age:', mode_age)
# histogram of BMI

data.bmi.plot(kind='hist', color='orange', edgecolor='black', figsize=(10,7))

plt.title('Distribution of BMI', size=24)

plt.xlabel('Body Mass Index (BMI)', size=18)

plt.ylabel('Frequency', size=18)



# find average BMI

avg_BMI = data.bmi.mean()

print('Average BMI:', avg_BMI)
# countplot to compare the number of children

plt.figure(figsize=(12,4))

sns.countplot(x='children', data=data, color='orange', edgecolor='black') 

plt.title('Distribution of Children', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Number of Children',size=18)

plt.show()
# countplot to compare the number of people from different regions

plt.figure(figsize=(12,4))

sns.countplot(x='region', data=tmp_data, color='orange', edgecolor='black') 

plt.title('Distribution of People Across Regions', size='24')

plt.ylabel('Frequency',size=18)

plt.xlabel('Region',size=18)

plt.show()
# histogram of region

data[data['region_northeast'] == 1].charges.plot(kind='hist', color='blue', edgecolor='black', alpha=0.5, figsize=(10, 7))

data[data['region_northwest'] == 1].charges.plot(kind='hist', color='magenta', edgecolor='black', alpha=0.5, figsize=(10, 7))

data[data['region_southeast'] == 1].charges.plot(kind='hist', color='green', edgecolor='black', alpha=0.5, figsize=(10, 7))

data[data['region_southwest'] == 1].charges.plot(kind='hist', color='red', edgecolor='black', alpha=0.5, figsize=(10, 7))

plt.legend(labels=['Northeast','Northwest','Southeast','Southwest'])

plt.title('Distribution of Charges Between Regions', size=24)

plt.xlabel('Medical Charges', size=18)

plt.ylabel('Frequency', size=18)
data[data['smoker_yes'] == 1].charges.plot(kind='hist', color='blue', edgecolor='black', alpha=0.5, figsize=(10, 7))

data[data['smoker_no'] == 1].charges.plot(kind='hist', color='magenta', edgecolor='black', alpha=0.5, figsize=(10, 7))

plt.legend(labels=['Smoker', 'Non-Smoker'])

plt.title('Distribution of Charges on Smokers & Non-Smokers', size=24)

plt.xlabel('Medical Charges', size=18)

plt.ylabel('Frequency', size=18)
data[data['sex_male'] == 1].charges.plot(kind='hist', color='blue', edgecolor='black', alpha=0.5, figsize=(10, 7))

data[data['sex_female'] == 1].charges.plot(kind='hist', color='magenta', edgecolor='black', alpha=0.5, figsize=(10, 7))

plt.legend(labels=['Male', 'Female'])

plt.title('Distribution of Charges on Males & Females', size=24)

plt.xlabel('Medical Charges', size=18)

plt.ylabel('Frequency', size=18)
# scatter plot of Age, Smokers, and Medical Charges

ax1 = data[data['smoker_yes'] == 1].plot(kind='scatter', x='age', y='charges', color='blue', alpha=0.5, figsize=(10, 7))

data[data['smoker_no'] == 1].plot(kind='scatter', x='age', y='charges', color='magenta', alpha=0.5, figsize=(10 ,7), ax=ax1)



# legend, title, and labels

plt.legend(labels=['Smoker', 'Non-Smoker'])

plt.title('Relationship Between Age, Smoking, and Medical Charges', size=24)

plt.xlabel('Age', size=18)

plt.ylabel('Medical Charges', size=18);
# scatter plot of BMI, Smokers, and Medical Charges

ax1 = data[data['smoker_yes'] == 1].plot(kind='scatter', x='bmi', y='charges', color='blue', alpha=0.5, figsize=(10, 7))

data[data['smoker_no'] == 1].plot(kind='scatter', x='bmi', y='charges', color='magenta', alpha=0.5, figsize=(10 ,7), ax=ax1)

plt.legend(labels=['Smoker', 'Non-Smoker'])

plt.title('Relationship Between BMI, Smoking, and Medical Charges', size=24)

plt.xlabel('BMI', size=18)

plt.ylabel('Medical Charges', size=18)
# plot for underweight

plt.figure(figsize=(12,5))

plt.title("Medical Charges of BMI < 18.5 (Underweight)")

ax = sns.distplot(data[(data.bmi <= 18.5)]['charges'], color = 'm')



# calculate average medical charge for someone underweight

underweight_charge = data[(data.bmi <= 18.5)]['charges'].mean()

print('Average Medical Charge (Underweight BMI):', underweight_charge)
# plot for normal weight

plt.figure(figsize=(12,5))

plt.title("Medical Charges of BMI Between 18.5 - 25 (Normal)")

ax = sns.distplot(data[(data.bmi.between(18.5,25))]['charges'], color = 'g')



# calculate average medical charge for someone normal

normal_charge = data[data.bmi.between(18.5,25)]['charges'].mean()

print('Average Medical Charge (Normal BMI):', normal_charge)
# plot for overweight

plt.figure(figsize=(12,5))

plt.title("Medical Charges of BMI Between 25 - 30 (Overweight)")

ax = sns.distplot(data[(data.bmi.between(25,30))]['charges'], color = 'y')



# calculate average medical charge for someone overweight

overweight_charge = data[data.bmi.between(25,30)]['charges'].mean()

print('Average Medical Charge (Overweight BMI):', overweight_charge)
# plot for obese

plt.figure(figsize=(12,5))

plt.title("Medical Charges of BMI >= 30 (Obese)")

ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'r')



# calculate average medical charge for someone obese

obese_charge = data[data.bmi >= 30]['charges'].mean()

print('Average Medical Charge (Obese BMI):', obese_charge)
# calculate average medical charge for someone with zero children

zero_child = data[(data.children == 0)]['charges'].mean()

print('Average Medical Charge (Zero Children):', zero_child)



# calculate average medical charge for someone with one child

one_child = data[(data.children == 1)]['charges'].mean()

print('Average Medical Charge (One Child):', one_child)



# calculate average medical charge for someone with two children

two_child = data[(data.children == 2)]['charges'].mean()

print('Average Medical Charge (Two Children):', two_child)



# calculate average medical charge for someone with three children

three_child = data[(data.children == 3)]['charges'].mean()

print('Average Medical Charge (Three Children):', three_child)



# calculate average medical charge for someone with four children

four_child = data[(data.children == 4)]['charges'].mean()

print('Average Medical Charge (Four Children):', four_child)



# calculate average medical charge for someone with five children

five_child = data[(data.children == 5)]['charges'].mean()

print('Average Medical Charge (Five Children):', five_child)
g= sns.catplot(x="children", y='charges', hue=None, data=tmp_data,

                height= 6, kind="point", aspect=1.0, legend_out=True, width=0.4, linewidth=3,  linestyles = '--', capsize=.1, dodge= 0.15,

                sharey=True, 

                palette = sns.color_palette("deep", n_colors = 1))



g.despine(left=True)

g.set_titles("Relationship Between Children and Medical Charges", weight='bold')

g.set_axis_labels("Number of Children", "Medical Charges")
g= sns.catplot(x="children", y='charges', hue='smoker', data=tmp_data,

                height= 6, kind="point", aspect=1.0, legend_out=True, width=0.4, linewidth=3,  linestyles = '--', capsize=.1, dodge= 0.15,

                sharey=True, 

                palette = sns.color_palette("deep", n_colors = 2))



g.despine(left=True)

g.set_titles("Relationship Between Children, Smoker, and Medical Charges", weight='bold')

g.set_axis_labels("Number of Children", "Medical Charges")

g._legend.set_title("Smoker")
# import dataset using Pandas

data = pd.read_csv('../input/insurance/insurance.csv')



# drop regions column

data = data.drop(['region'], axis=1)
data.head()
# set independent and dependent variables

x = data.iloc[:,:-1].values # age, sex, BMI, children, smoker, region

y = data.iloc[:,-1].values # charges



print('Independent Variables\n',x)

print('\nDependent Variables\n',y)
# get column index of categorical variables (sex, smoker, region)

print('Sex Column Index:', data.columns.get_loc('sex'))

print('Smoker Column Index:', data.columns.get_loc('smoker'))
# import module for one-hot encoding scheme

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



# sex and smoker column index is 1 and 4

dummy_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,4])], remainder='passthrough')

x = np.array(dummy_transformer.fit_transform(x))
# import module to split data into training and test set

from sklearn.model_selection import train_test_split 



# 80% training & 20% testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# import the LinearRegression() class

from sklearn.linear_model import LinearRegression



# create a regressor model

regressor = LinearRegression()



# fit the training data, feature scaling is not needed for regression models

regressor.fit(x_train, y_train)
# the vector of the predicted medical charges in the training set

y_train_pred = regressor.predict(x_train)



# the vector of the predicted medical charges in the test set

y_test_pred = regressor.predict(x_test)
# compare y_test_pred (prediction) to the y_test (actual)

i = 0

while i < len(y_test_pred):

    diff = abs(round(y_test_pred[i], 2) - y_test[i])

    print("Predicted: " + str(round(y_test_pred[i], 2)) + " vs Actual: " + str(round(y_test[i], 2)) +

          " ---> Difference: " + str(round(diff, 2)))

    i += 1
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# calculate MSE values on the training and test set

MSE_train = mean_squared_error(y_train, y_train_pred)

MSE_test = mean_squared_error(y_test, y_test_pred)



# calculate R2 values on the training and test set

R2_train = r2_score(y_train, y_train_pred)

R2_test = r2_score(y_test, y_test_pred)



print('MSE (Training):', MSE_train)

print('MSE (Test):', MSE_test)



print('\nR2 (Training):', R2_train)

print('R2 (Test):', R2_test)
"""

regressor.predict([[sex_female, sex_male, smoker_no, smoker_yes, age, BMI, children]])



Only for categorical variables:

1 - Yes/True

0 - No/False

"""

# enter categorical and numerical inputs

print(regressor.predict([[1,0,0,1,19,27.90,0]]))