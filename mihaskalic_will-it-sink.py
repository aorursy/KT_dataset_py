import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_json("../input/train.json")
df.describe()
df.head()
band1_sample = df.loc[0, "band_1"]

df.loc[0, "band_1"]

print("Iceberg:", df.loc[0, 'is_iceberg'])

print("Shape:", len(band1_sample))

band1_sample = np.array(band1_sample).reshape(75, 75)

plt.imshow(band1_sample)

plt.title("Band 1")

plt.show()

band2_sample = df.loc[0, "band_2"]

band2_sample = np.array(band2_sample).reshape(75, 75)

plt.title("Band 2")

plt.imshow(band2_sample)

plt.show()
band1_sample = df.loc[2, "band_1"]

df.loc[2, "band_1"]

print("Iceberg:", df.loc[2, 'is_iceberg'])

print("Shape:", len(band1_sample))

band1_sample = np.array(band1_sample).reshape(75, 75)

plt.imshow(band1_sample)

plt.title("Band 1")

plt.show()

band2_sample = df.loc[2, "band_2"]

band2_sample = np.array(band2_sample).reshape(75, 75)

plt.title("Band 2")

plt.imshow(band2_sample)

plt.show()