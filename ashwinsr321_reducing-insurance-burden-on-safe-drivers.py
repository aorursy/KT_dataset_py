import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib as mpl

import seaborn as sns

from collections import Counter

import missingno as msno

from sklearn.ensemble import RandomForestClassifier

from sklearn import datasets

from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
print("Total number of input features : ", (train.shape[1] - 2)) # id and target are not features, hence -2

print("Number of users in train : ", train.shape[0])

print("Number of users in test : ", test.shape[0])
train.info(memory_usage='deep', verbose=False)

test.info(memory_usage='deep', verbose=False)
print("Total NaN in train data : ", train.isnull().sum().sum())

print("Total NaN in test data : ", test.isnull().sum().sum())
train_missing_count = (train == -1).sum()

plt.rcParams['figure.figsize'] = (15,8)

train_missing_count.plot.bar()

plt.show()
test_missing_count = (test == -1).sum()

test_missing_count.plot.bar()

plt.show()
required_columns = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat', 'ps_car_11', 'ps_car_14']

train_temp = train.copy()

train_temp = pd.DataFrame(train_temp, columns=required_columns)

train_temp = train_temp.replace(-1, np.NaN)

msno.matrix(df=train_temp, figsize=(20, 15))

msno.heatmap(train_temp,figsize=(20,15))

del train_temp
train_statistics = train.iloc[:,2:].describe()

train_statistics
print(train['target'].sum(), "people claimed insurance and",train.shape[0] - train['target'].sum(), "did not claim imsurance")

print(((train['target'].sum()*100.0)/train.shape[0]), "% of the people claimed insurance and",(((train.shape[0] - train['target'].sum())*100.0)/train.shape[0]), "did not claim imsurance")





objects = ('Claimed', 'Not Claimed')

y_pos = np.arange(len(objects))

performance = [train['target'].sum(), train.shape[0] - train['target'].sum()]

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('claim #')

plt.title('Claimed vs Not Claimed')

plt.show()



objects = ('Claimed', 'Not Claimed')

colors = ['red', 'yellowgreen']

sizes = [(train['target'].sum()*100.0)/train.shape[0], ((train.shape[0] - train['target'].sum())*100.0)/train.shape[0]]

explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=objects, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()

unique_counter = Counter()

for col in train.columns:

    unique_counter[col] = len(np.sort(train[col].unique()))

binary_columns = [ col for col , val in unique_counter.items() if(val==2)]

binary_column_sum = []

for col in binary_columns:

    binary_column_sum.append(train[col].sum())

#List of binary columns

binary_columns
# data to plot

n_groups = len(binary_columns)

one_cols = binary_column_sum

zero_cols = train.shape[0] - np.asarray(binary_column_sum)

 

# create plot

plt.rcParams['figure.figsize'] = (15,8)

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.3

opacity = 0.8

 

rects1 = plt.bar(index, one_cols, bar_width,

                 alpha=opacity,

                 color='g',

                 label='1')

 

rects2 = plt.bar(index + bar_width, zero_cols, bar_width,

                 alpha=opacity,

                 color='b',

                 label='0')



plt.ylabel('#', fontsize=14)

plt.title('Binary Features', fontsize=20)

plt.xticks(index + bar_width/2, binary_columns, rotation='vertical', fontsize=12)

plt.legend()

 

plt.tight_layout()

plt.show()
# Univariate Histograms

columns_multi = [x for x in list(train.columns) if x not in binary_columns]

columns_multi.remove('id')

columns_multi

plt.rcParams['figure.figsize'] = (15,40)

names = columns_multi

train.hist(layout = (10,4), column = columns_multi)

plt.show()
names = columns_multi

train.plot(kind='density', subplots=True, layout=(15,4), sharex=False)

plt.show()
# Correction Matrix Plot

names = train.columns

correlations = train.corr()

# plot correlation matrix

plt.rcParams['figure.figsize'] = (15,12)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,59,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names, rotation=90)

ax.set_yticklabels(names)

plt.show()
# Create a random forest classifier

clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the classifier

clf.fit(train.iloc[:,2:], train.iloc[:,1])



# Plot the gini importance of each feature

feature_importances = sorted(zip(clf.feature_importances_, list(train.columns)[2:]), reverse=True)

objects = (list(zip(*feature_importances))[1])

y_pos = np.arange(len(objects))

performance = np.array(list(zip(*feature_importances))[0])

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation='vertical')

plt.ylabel('Importance')

plt.title('Feature Importances using Random forest')

plt.show()
# Feature Importance

# fit an Extra Trees model to the data

model = ExtraTreesClassifier()

model.fit(train.iloc[:,2:], train.iloc[:,1])





# Plot the gini importance of each feature

feature_importances = sorted(zip(model.feature_importances_, list(train.columns)[2:]), reverse=True)

objects = (list(zip(*feature_importances))[1])

y_pos = np.arange(len(objects))

performance = np.array(list(zip(*feature_importances))[0])

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation='vertical')

plt.ylabel('Importance')

plt.title('Feature Importances using Extra Trees Classifier')

plt.show()
X = train.iloc[:,2:]

y = train.iloc[:,1]

# fit model to training data

model = XGBClassifier()

model.fit(X, y)

# plot

feature_importances = sorted(zip(model.feature_importances_, list(train.columns)[2:]), reverse=True)

objects = (list(zip(*feature_importances))[1])

y_pos = np.arange(len(objects))

performance = np.array(list(zip(*feature_importances))[0])

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation='vertical')

plt.ylabel('Importance')

plt.title('Feature Importances using XGBoost')

plt.show()