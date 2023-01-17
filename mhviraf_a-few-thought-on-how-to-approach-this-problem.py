import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv', nrows= 6_000_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train_data.info()
plt.figure()
plt.plot(train_data['acoustic_data']/train_data.acoustic_data.max())
plt.plot(train_data['time_to_failure']/train_data.time_to_failure.max())
print(train_data.time_to_failure.min(), train_data.time_to_failure.max())
print(train_data.acoustic_data.min(), train_data.acoustic_data.max())
minimum_time_to_failure_index = (train_data.time_to_failure <= 0.0007) # 5656573 
print(sum(minimum_time_to_failure_index))
print(train_data.loc[5656573:5656573+1, :])

