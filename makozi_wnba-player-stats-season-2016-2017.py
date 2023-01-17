import pandas as pd

wnba=pd.read_csv('../input/WNBA Stats.csv')

wnba.head()
wnba.tail()
wnba.shape
parameter= wnba['Games Played'].max()
parameter
sample= wnba['Games Played'].sample(30,random_state=1)

statistic= sample.max()
sampling_error= parameter - statistic
sampling_error


import matplotlib.pyplot as plt



sample_means = []

population_mean = wnba['PTS'].mean()



for i in range(100):

    sample = wnba['PTS'].sample(10, random_state=i)

    sample_means.append(sample.mean())



    plt.ylabel('Sample Mean')

    plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(population_mean)
population_mean


import matplotlib.pyplot as plt



sample_means = []

population_mean = wnba['PTS'].mean()



for i in range(100):

    sample = wnba['PTS'].sample(20, random_state=i)

    sample_means.append(sample.mean())



    plt.ylabel('Sample Mean')

    plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(population_mean)


import matplotlib.pyplot as plt



sample_means = []

population_mean = wnba['PTS'].mean()



for i in range(100):

    sample = wnba['PTS'].sample(60, random_state=i)

    sample_means.append(sample.mean())



    plt.ylabel('Sample Mean')

    plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(population_mean)


import matplotlib.pyplot as plt



sample_means = []

population_mean = wnba['PTS'].mean()



for i in range(100):

    sample = wnba['PTS'].sample(80, random_state=i)

    sample_means.append(sample.mean())



    plt.ylabel('Sample Mean')

    plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(population_mean)


import matplotlib.pyplot as plt



sample_means = []

population_mean = wnba['PTS'].mean()



for i in range(100):

    sample = wnba['PTS'].sample(100, random_state=i)

    sample_means.append(sample.mean())



    plt.ylabel('Sample Mean')

    plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(population_mean)
print(wnba['Pos'].unique())
wnba['Pts_per_game'] = wnba['PTS'] / wnba['Games Played']



# Stratifying the data in five strata

stratum_G = wnba[wnba.Pos == 'G']

stratum_F = wnba[wnba.Pos == 'F']

stratum_C = wnba[wnba.Pos == 'C']

stratum_GF = wnba[wnba.Pos == 'G/F']

stratum_FC = wnba[wnba.Pos == 'F/C']



points_per_position = {}

for stratum, position in [(stratum_G, 'G'), (stratum_F, 'F'), (stratum_C, 'C'),

                (stratum_GF, 'G/F'), (stratum_FC, 'F/C')]:

    

    sample = stratum['Pts_per_game'].sample(10, random_state = 0) # simple random sapling on each stratum

    points_per_position[position] = sample.mean()

    

position_most_points = max(points_per_position, key = points_per_position.get)
print(wnba['Games Played'].min())
print(wnba['Games Played'].max())
print(wnba['Games Played'].value_counts(bins = 3, normalize = True) * 100)
under_12 = wnba[wnba['Games Played'] <= 12]

btw_13_22 = wnba[(wnba['Games Played'] > 12) & (wnba['Games Played'] <= 22)]

over_23 = wnba[wnba['Games Played'] > 22]



proportional_sampling_means = []



for i in range(100):

    sample_under_12 = under_12['PTS'].sample(1, random_state = i)

    sample_btw_13_22 = btw_13_22['PTS'].sample(2, random_state = i)

    sample_over_23 = over_23['PTS'].sample(7, random_state = i)

    

    final_sample = pd.concat([sample_under_12, sample_btw_13_22, sample_over_23])

    proportional_sampling_means.append(final_sample.mean())

    

plt.scatter(range(1,101), proportional_sampling_means)

plt.axhline(wnba['PTS'].mean())
print(wnba['Team'].unique())
print(pd.Series(wnba['Team'].unique()).sample(4, random_state=0))
clusters = pd.Series(wnba['Team'].unique()).sample(4, random_state = 0)



sample = pd.DataFrame()



for cluster in clusters:

    data_collected = wnba[wnba['Team'] == cluster]

    sample = sample.append(data_collected)



sampling_error_height = wnba['Height'].mean() - sample['Height'].mean()

sampling_error_age = wnba['Age'].mean() - sample['Age'].mean()

sampling_error_BMI = wnba['BMI'].mean() - sample['BMI'].mean()

sampling_error_points = wnba['PTS'].mean() - sample['PTS'].mean()
sampling_error_height
sampling_error_age
sampling_error_BMI
sampling_error_points