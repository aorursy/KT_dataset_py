import h5py
import matplotlib.pyplot as plt
from pathlib import Path

dataset = h5py.File('../input/dataset.hdf5', 'r')
test_64 = dataset['test_64']

plt.imshow(test_64[1], cmap='viridis')
plt.colorbar()
plt.show()