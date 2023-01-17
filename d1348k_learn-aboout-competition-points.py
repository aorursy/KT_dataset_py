import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
x = np.arange(1, 100)

y = np.power(x, -0.75)

plt.plot(y)

plt.ylim(-0.05,1.05)

plt.title('Effect of Rank')

plt.xlabel('Rnak in private LB')

plt.ylabel('Coefficient');
print('1st :{:.4} '.format(np.power(1, -0.75)))

print('2nd :{:.4} '.format(np.power(2, -0.75)))

print('3rd :{:.4} '.format(np.power(3, -0.75)))

print('10th :{:.4} '.format(np.power(10, -0.75)))

print('50th :{:.4} '.format(np.power(50, -0.75)))

print('100th :{:.4} '.format(np.power(100, -0.75)))
x = np.arange(1, 10000)

y = np.log10(1+np.log10(x))

plt.plot(y)

plt.ylim(-0.05,1.05)

plt.title('Effect of Participants')

plt.xlabel('Number of teams')

plt.ylabel('Coefficient');
print('1000 teams :{:.4} '.format(np.log10(1+np.log10(1000))))

print('10000 teams :{:.4} '.format(np.log10(1+np.log10(10000))))

print('Ratio :{:.4} '.format(np.log10(1+np.log10(10000))/np.log10(1+np.log10(1000))))
x = np.arange(1, 12)

y = 1/np.sqrt(x)

plt.plot(y)

plt.ylim(-0.05,1.05)

plt.title('Effect of teammates')

plt.xlabel('Number of teammates')

plt.ylabel('Coefficient');
print('1 member  :{:.4} '.format(1/np.sqrt(1)))

print('2 members :{:.4} '.format(1/np.sqrt(2)))

print('3 members :{:.4} '.format(1/np.sqrt(3)))

print('4 members :{:.4} '.format(1/np.sqrt(4)))

print('5 members :{:.4} '.format(1/np.sqrt(5)))

print('8 members :{:.4} '.format(1/np.sqrt(8)))
x = np.arange(0, 1000)

y = np.exp(-x/500)

plt.plot(y)

plt.ylim(-0.05,1.05)

plt.title('Effect of Days')

plt.xlabel('Lapsed days')

plt.ylabel('Coefficient');
y_list = []

x1_array = np.arange(1, 31)

x2_array = np.arange(1, 9)

for x1 in x1_array:

    for x2 in x2_array:

        y = np.power(x1, -0.75) * (1/np.sqrt(x2)) * 100

        y_list.append(y)

y_array = np.array(y_list).reshape(x1_array.shape[0], x2_array.shape[0])

y_df = pd.DataFrame(y_array)

y_df.index, y_df.columns = x1_array, x2_array

plt.subplots(figsize=(9,7))

sns.heatmap(y_df, cmap='Blues', annot=True,annot_kws={"size": 10}, fmt=".1f")

plt.title('Coefficient relationship between rank and number of teammates(%)')

plt.xlabel('Number of teammates')

plt.ylabel('Rank');
def calculate_points(teammates, rank, teams, days):

    points = 100000 * 1/np.sqrt(teammates) * np.power(rank, -0.75) * np.log10(1+np.log10(teams)) * np.exp(days/500)

    return points
# My points in Freesound Audio Tagging 2019

teammates = 3

rank = 7

teams = 880

days = 0

points_1 = calculate_points(teammates, rank, teams, days)

points_1
# Solo Gold in 1000teams competition

teammates = 1

rank = 1

teams = 1000

days = 0

points_2 = calculate_points(teammates, rank, teams, days)

points_2