import numpy as np

import matplotlib.pyplot as plt
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

df.shape
height = df[:100, 3] # рост первых 100 человек



fig, ax = plt.subplots(figsize=(20, 7))

plt.plot(height);
fig, ax = plt.subplots(figsize=(20, 7))

plt.scatter(df[:100, 0], height, marker='*', s=100); # обязательно два аргумента x, y!
id = df[:100, 0]



fig, ax = plt.subplots(figsize=(20, 7))

plt.bar(id, height=height);
fig, ax = plt.subplots(figsize=(7, 20))

plt.barh(id, width=height);
fig, ax = plt.subplots(1, 2, figsize=(20, 7))



ax[0].hist(height)

ax[1].hist(height, histtype='step');
plt.hist(height, bins=5);
fig, ax = plt.subplots(figsize=(20, 7))

plt.boxplot(height, vert=False);
# Сравним мужчин и женщин по росту

women_height =  df[df[:, 2]==1, 3]

men_height = df[df[:, 2]==2, 3]



fig, ax = plt.subplots(figsize=(20, 7))

plt.boxplot([women_height, men_height], vert=False, labels=['women', 'men']);
fig, ax = plt.subplots(figsize=(10, 20))

plt.violinplot([women_height, men_height]);
height = df[:100, 3]

age = df[:100, 1] / 365.25 # изначально возраст был в днях



fig, ax = plt.subplots(figsize=(7, 7))

plt.scatter(age, height)

plt.xlabel('age, days')

plt.ylabel('height, cm')

plt.title('Age vs. Height');
fig, ax = plt.subplots()



plt.bar([1, 2], height=[women_height.mean(), men_height.mean()], color=['red', 'blue'])

ax.set_xticks([1, 2])

ax.set_xticklabels(['women', 'men']);