import pandas as pd

import matplotlib.pyplot as plt
data = (pd.read_csv("/kaggle/input/geospatial-sao-paulo-crime-database/dataset-limpo.csv", parse_dates=['time'])

        )



data = data[data.time > '2015-01-01']
plt.figure(figsize=(12, 4))

resampled_data = data.resample("3m", on='time')

bike_count = resampled_data.Celular.count()

bike_frac = bike_count / resampled_data['id'].count() * 100

bike_frac.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving phones (%)")

plt.title("Stolen phones per trimester")

plt.show()
plt.figure(figsize=(12, 4))

resampled_data = data.resample("3m", on='time')

bike_count = resampled_data.Bicicleta.count()

bike_frac = bike_count / resampled_data['id'].count() * 100

bike_frac.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving bicycles (%)")

plt.title("Stolen bikes per trimester")

plt.show()
plt.figure(figsize=(12, 4))

resampled_data = data.resample("3m", on='time')

bike_count = resampled_data.Documentos.count()

bike_frac = bike_count / resampled_data['id'].count() * 100

bike_frac.plot()

plt.xlabel("Time")

plt.ylabel("Percentages of incidents involving personal documents (%)")

plt.title("Stolen documents per trimester")

plt.show()