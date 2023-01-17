import pandas as pd

test = "../input/311-service-requests.csv"
test_data = pd.read_csv(test)
print(test_data)
test_data.shape

test_data.head(15)
test_data.head(5)
test_data.columns

reviews.iloc[0:5, 1]
