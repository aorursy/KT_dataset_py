!pip install pymap3d==2.1.0
!pip install -U l5kit
import zarr
import numpy as np

z = zarr.open("./dataset.zarr", mode="w", shape=(500,), dtype=np.float32, chunks=(100,))

# We can write to it by assigning to it. This gets persisted on disk.
z[0:150] = np.arange(150)
print(z.info)

# Reading from a zarr array is as easy as slicing from it like you would any numpy array. 
# The return value is an ordinary numpy array. Zarr takes care of determining which chunks to read from.
print(z[:10])
print(z[::20]) # Read every 20th value
from l5kit.data import ChunkedDataset

dt = ChunkedDataset("../input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr").open()
centroids= []
for idx in range(10_000):
    centroid = dt.agents[idx]["centroid"]
    centroids.append(centroid)
print(dt)
centroids[0:10]