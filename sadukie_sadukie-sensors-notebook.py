import pandas as pd
sensors = pd.read_csv('/kaggle/input/sadukie-sensors/sensors.csv')
sensors.shape
sensors.sensorname.unique()
sensors.date.min()
sensors.date.max()
sensors.head()
sensors.tail()
