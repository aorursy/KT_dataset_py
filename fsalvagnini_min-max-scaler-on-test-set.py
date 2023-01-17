import numpy as np
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
my_array = np.array([1, 10, 7, 15, 0, 5, 9])
my_array = my_array.reshape((-1, 1))
scaled_array = min_max_scaler.fit_transform(my_array)
scaled_array
test_array = np.array([0, 1, 10, 15, 40, 30])
test_array = test_array.reshape((-1, 1))
min_max_scaler.transform(test_array)
