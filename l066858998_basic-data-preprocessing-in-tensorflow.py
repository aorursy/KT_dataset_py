import pandas as pd

import tensorflow as tf
with open("data.csv", 'w') as f:

    

    f.write("NumRooms,Type,Price\n") # column names in csv

    f.write("5,A,25000\n")

    f.write("NA,B,30000\n")

    f.write("10,C,10000\n")

    f.write("NA,NA,50000\n")
df = pd.read_csv("data.csv")

df
# access element in dataframe through intger-location based index

inputs = df.iloc[:, 0:2]

outputs = df.iloc[:, 2]
inputs
# two ways to handle missing value: imputation or deletion

# we fill missing value with imputation

inputs.mean()
inputs = inputs.fillna(inputs.mean())

inputs
# NaN is considered sa discrete or catogorical value

# hence, we can do "one-hot encoding" on this column

inputs = pd.get_dummies(data=inputs, dummy_na=True)

inputs
# convert dataframe to numpy array

inputs = inputs.values

outputs = outputs.values

inputs, outputs
input_tensor = tf.constant(inputs)

output_tensor = tf.constant(outputs)
input_tensor
output_tensor