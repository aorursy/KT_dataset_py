import pandas as pd
bankData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter03/bank-full.csv', sep=";")
bankData.head()
# Printing the shape of the dataframe
bankData.shape
# Summarizing the statistics of the numerical raw data
bankData.describe().T