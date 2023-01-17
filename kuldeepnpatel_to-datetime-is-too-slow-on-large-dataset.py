# data processing, CSV file I/O
import pandas as pd
#Load the data
data=pd.read_csv("../input/train.csv",nrows = 3000000)
print(data.shape)
data.head(3)
# Type of pickup_datetime is an object
data['pickup_datetime'].dtypes 
#convert data type of pickup_datetime object to datetime
print("Execution Time without setting the parameter infer_datetime_format as True")
%timeit pd.to_datetime(data['pickup_datetime'])
#Load the data again and use parameter infer_datetime_format=True
data=pd.read_csv("../input/train.csv",nrows = 3000000)
print(data.shape)
data.head(3)
# Type of pickup_datetime is an object
data['pickup_datetime'].dtypes 
#convert data type of pickup_datetime, object to datetime
# Use Parameter infer_datetime_format and set it as True
print("Execution Time with the help of parameter infer_datetime_format=True")
%timeit pd.to_datetime(data['pickup_datetime'],infer_datetime_format=True)
