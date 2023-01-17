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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model as lm

import kagglegym
# Create environment

env = kagglegym.make()



# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train
train_1=observation.train
train.head()
train.shape
def findMatchedColumnsUsingPrefix(prefix, train):

    columns = train.columns[train.columns.str.startswith(prefix)]

    return list(columns.values)
derived_columns = findMatchedColumnsUsingPrefix("derived", train)

fundamental_columns = findMatchedColumnsUsingPrefix("fundamental", train)

technical_columns = findMatchedColumnsUsingPrefix("technical", train)



print("There are {} derived columns".format(len(derived_columns)))

print("There are {} fundamental columns".format(len(fundamental_columns)))

print("There are {} technical columns".format(len(technical_columns)))
#To get statistics on all variables

train.describe().transpose()
#to identify Null Values

train.isnull().sum().sort_index(axis=0, ascending=True, inplace=False)
#To replace null values with mean values

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)

train.head()
#To get statistics on the variable timestamp

train.timestamp.describe()
# Total unique timestamps in the dataset

len(train["timestamp"].unique())
print(train["timestamp"].unique())
# Total unique ids in the dataset

len(train["id"].unique())
#To get statistics on the target variable y 

train.y.describe()




from pandas import Series

from matplotlib import pyplot

series = Series(train["y"])

print(series.head())

series.plot()

pyplot.show()





train["y"].hist(bins = 30, color = "orange")

plt.xlabel("Target Variable")

plt.ylabel("Frequency")
so=train.corr().unstack().order(kind="quicksort")

#print(so[-len(so):-len(so)+20])

print(so["y"].sort_values(ascending=False))
plt.plot(train.technical_20)
plt.plot(train.technical_30)
plt.plot(train.fundamental_11)
train_clean =train_1.dropna(axis=0)
len(train_clean)
alpha = plt.figure()

plt.scatter(train_clean["technical_20"], train_clean["y"], alpha=.1, s=400)

plt.xlabel("technical_20") 

plt.ylabel("Target variable")

plt.show()
alpha = plt.figure()

plt.scatter(train_clean["technical_30"], train_clean["y"], alpha=.1, s=400)

plt.xlabel("technical_30") 

plt.ylabel("Target variable")

plt.show()
alpha = plt.figure()

plt.scatter(train_clean["fundamental_11"], train_clean["y"], alpha=.1, s=400)

plt.xlabel("technical_30") 

plt.ylabel("Target variable")

plt.show()
# Let's take a 1000 sample of the data to explore 

# We will use raw data which has the missing data removed from it

train_sample = train_1.sample(n=1000)
# Plot the most correlated variables 

alpha = plt.figure()

plt.scatter(train_sample["timestamp"], train_sample["technical_20"], alpha=.5)

plt.xlabel("timestamp") 

plt.ylabel("technical_20")

plt.show()
# Plot the most correlated variables 

alpha = plt.figure()

plt.scatter(train_sample["timestamp"], train_sample["technical_30"], alpha=.5)

plt.xlabel("timestamp") 

plt.ylabel("technical_30")

plt.show()
# Plot the most correlated variables 

alpha = plt.figure()

plt.scatter(train_sample["timestamp"], train_sample["fundamental_11"], alpha=.5)

plt.xlabel("timestamp") 

plt.ylabel("fundamental_11")

plt.show()
# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train.columns if col not in ['y', 'id', 'timestamp']]



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(train[col].values, train.y.values)[0,1])

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(values), color='y')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient")

#autolabel(rects)

plt.show()
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']



temp_df = train[cols_to_use]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
#To set low and min to take care of outliers

low_y_cut = -0.086093

high_y_cut = 0.093497
y_values_within = ((train['y'] > low_y_cut) & (train['y'] <high_y_cut))
train_cut = train.loc[y_values_within,:]
cols='technical_20'

x_train = train_cut[cols]

y = train_cut["y"]
models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

    models_dict[col] = model
col = 'technical_30'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train



col = 'technical_20'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train



col = 'technical_20'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info
# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train



cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']



mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)



model = lm.LinearRegression()

model.fit(np.array(train[cols_to_use]), train.y.values)



while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use])

    observation.target.y = model.predict(test_x)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        

    observation, reward, done, info = env.step(target)

    if done:

        break

info