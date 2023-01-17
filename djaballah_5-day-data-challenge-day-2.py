import pandas as pd

import matplotlib.pyplot as plt
cereal_df = pd.read_csv('../input/cereal.csv')

cereal_df.head()
cereal_df.describe()
protein_quantity = cereal_df['protein']

plt.hist(protein_quantity)

plt.title('Histogram of the protein quantity')