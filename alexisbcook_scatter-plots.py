

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

insurance_filepath = "../input/insurance.csv"



# Read the file into a variable insurance_data

insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head()
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
sns.swarmplot(x=insurance_data['smoker'],

              y=insurance_data['charges'])