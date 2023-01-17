import pandas as pd

import matplotlib.pyplot as plt
# Load dataset

suicide_data = pd.read_csv('../input/master.csv')

# Describe dataset

suicide_data.describe()
# Get suicide rate column

suicide_rate_column = suicide_data['suicides/100k pop']

suicide_rate_column.sample(5)

# Plot histogram

suicide_rate_column.hist()

plt.title('Suicides per 100,000 people')

plt.ylabel('Frequency')