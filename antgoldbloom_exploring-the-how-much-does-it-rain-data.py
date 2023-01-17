%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# reading the file as dataframe and getting summary #
train_file = "../input/train.csv"
train_df = pd.read_csv(train_file, usecols = ['Id','Expected'])

# grouping the rows based on id and get the rainfall #
train_df_grouped = train_df.groupby(['Id'])
exp_rainfall = np.sort(np.array(train_df_grouped['Expected'].aggregate('mean')))

# plotting a scatter plot #
plt.figure()
plt.hist(exp_rainfall)
plt.title("Scatterplot for Rainfall distribution in train sample")
plt.xlabel("Rainfall in mm")
plt.savefig("ExpectedRainfall.png")
plt.show()
# plotting a scatter plot #
plt.figure()
plt.hist(exp_rainfall[1:1000000])
plt.title("Scatterplot for Rainfall distribution in train sample")
plt.xlabel("Rainfall in mm")
plt.savefig("ExpectedRainfall.png")
plt.show()
