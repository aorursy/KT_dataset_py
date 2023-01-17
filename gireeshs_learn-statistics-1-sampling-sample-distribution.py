import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data.head()
np.random.seed(11)

sample_7_and_above = data[data['quality'].isin([7, 8])].sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

sample_7_and_above
x_bar = np.mean(sample_7_and_above["alcohol"])

print("The mean of the sample is: ", str(x_bar))

s = np.std(sample_7_and_above["alcohol"])

print("The standard deviation of the sample is: ", str(s))
sns.distplot(sample_7_and_above["alcohol"], hist=False)

title = "X_bar_1 = " + str(x_bar) + ", s1 = "+ str(s)

plt.title(title)
seed = np.arange(0, 9)



x_bar = []

std_dev = []



for s in seed:

    np.random.seed(s)

    sample_7_and_above = data[data['quality'].isin([7, 8])].sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))

    std_dev.append(np.std(sample_7_and_above["alcohol"]))

    

samples = pd.DataFrame(columns = ["Sample Means (X_bar)", "Sample Standard Deviation (s)"], data= list(zip(x_bar, std_dev)))

samples
sns.distplot(samples["Sample Means (X_bar)"])

plt.title("Distribution of the sample means")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))



ax = fig.add_subplot(3, 2, 1)

sns.distplot(data["alcohol"])

plt.title("Original distribution of Alcohol level in the entire dataset")



seed = np.arange(0, 2)

x_bar = []

for s in seed:

    np.random.seed(s)

    sample_7_and_above = data.sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))



ax = fig.add_subplot(3, 2, 2)

sns.distplot(x_bar)

plt.title("Sample means of 2 samples of 10 each")



seed = np.arange(0, 7)

x_bar = []

for s in seed:

    np.random.seed(s)

    sample_7_and_above = data.sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))

    

ax = fig.add_subplot(3, 2, 3)

sns.distplot(x_bar)

plt.title("Sample means of 7 samples of 10 each")



seed = np.arange(0, 20)

x_bar = []

for s in seed:

    np.random.seed(s)

    sample_7_and_above = data.sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))



ax = fig.add_subplot(3, 2, 4)

sns.distplot(x_bar)

plt.title("Sample means of 20 samples of 10 each")



seed = np.arange(0, 100)

x_bar = []

for s in seed:

    np.random.seed(s)

    sample_7_and_above = data.sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))



ax = fig.add_subplot(3, 2, 5)

sns.distplot(x_bar)

plt.title("100 samples of 10 each")



seed = np.arange(0, 500)

x_bar = []

for s in seed:

    np.random.seed(s)

    sample_7_and_above = data.sample(10)[['alcohol']].reset_index().drop(columns = ['index'])

    x_bar.append(np.mean(sample_7_and_above["alcohol"]))

    

ax = fig.add_subplot(3, 2, 6)

sns.distplot(x_bar)

plt.title("500 samples of 10 each")



fig.tight_layout()

plt.show()
np.random.seed(11)

sample_7_and_above = data[data['quality'].isin([7, 8])].sample(13)[['alcohol']].reset_index().drop(columns = ['index'])

sample_7_and_above