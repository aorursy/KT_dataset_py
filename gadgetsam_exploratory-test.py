import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])
set160 = [smj for smj in smjpegs if "set160" in smj]
print(set160)
first = plt.imread("../input/train_sm/set160_1.jpeg")
dims = np.shape(first)
print(dims)
plt.imshow(first)