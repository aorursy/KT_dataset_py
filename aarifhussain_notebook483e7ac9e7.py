import os
import cv2
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from os import listdir
listdir("../input/")
path_ct=("../input/covid19ct-scan-images/Train.csv")

print(len(path_ct))