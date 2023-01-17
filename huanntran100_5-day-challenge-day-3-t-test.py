from scipy.stats import ttest_ind

import pandas as pd

import matplotlib.pyplot as plt
# Load dataset

cereal_data = pd.read_csv('../input/cereal.csv')

# Describe data

cereal_data.describe()
# See sample of data

cereal_data.sample(5)
# Compare sodium between K and G

# Get sodium data for cereal manufacturer K 

mfr_K_sodium_data = cereal_data.loc[cereal_data['mfr'] == 'K']['sodium']

mfr_K_sodium_data.sample(5)
# Get sodium data for cereal manufacturer C

mfr_G_sodium_data = cereal_data.loc[cereal_data['mfr'] == 'G']['sodium']

mfr_G_sodium_data.sample(5)
# Calculate standard deviation sodium data for cereal manufacturer K

mfr_K_sodium_data.std()
# Calculate standard deviation sodium data for cereal manufacturer G

mfr_G_sodium_data.std()
# T-Test between K and G and use unequal variance

ttest_ind(mfr_K_sodium_data, mfr_G_sodium_data, equal_var=False)
mfr_K_sodium_data.hist()

plt.title('Sodium Levels in Cereal Manufacturer K')

plt.xlabel('Amount of Sodium')

plt.ylabel('Frequency')
mfr_G_sodium_data.hist()

plt.title('Sodium Levels in Cereal Manufacturer G')

plt.xlabel('Amount of Sodium')

plt.ylabel('Frequency')