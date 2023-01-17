import tensorflow as tf
import keras
import numpy as np 
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
print(os.getcwd())
print(os.listdir())
print(os.listdir("../input"))
print(os.listdir("../input/imaterialist-fashion-2020-fgvc7"))
import pandas as pd
sample_submi = pd.read_csv("../input/imaterialist-fashion-2020-fgvc7/sample_submission.csv")
sample_submi.head()
train_pd = pd.read_csv("../input/imaterialist-fashion-2020-fgvc7/train.csv")
train_pd.head()
data = open("../input/imaterialist-fashion-2020-fgvc7/label_descriptions.json")
import json
f = json.load(data)

