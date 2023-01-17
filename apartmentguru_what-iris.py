
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def norm(data, col):
    if isinstance(col, str):
        M, m = data[col].max(), data[col].min()
        data[col] = (data[col]-m)/(M-m)
    else:
        for c in col:
            data = norm(data, c)
    return data
data = pd.read_csv('../input/Iris.csv')
data['PetalArea'] = (data.PetalLengthCm * data.PetalWidthCm)
data['SepalVol'] = (data.SepalLengthCm * (np.pi * (data.SepalWidthCm/2) **2))

data = norm(data, ['SepalWidthCm', 'PetalLengthCm', 'SepalLengthCm', 'PetalWidthCm'])
data = data.drop(['Id'], axis=1)
plt.figure(figsize=(10 , 10))
pd.tools.plotting.radviz(data, 'Species')

