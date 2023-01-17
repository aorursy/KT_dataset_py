# Import the pandas package
import pandas as pd
# Read the CSV File stored in your file destination
# Remember to download the csv file and replace the file location in order to read in your csv
df = pd.read_csv('../input/train.csv')
# Run and test the functions here:
df.head()


# Run and test the functions here:
df.isnull().sum()


# Run and test the functions here:
df['Age']


# Run and test the functions here:
df[df['Age'] < 2]

