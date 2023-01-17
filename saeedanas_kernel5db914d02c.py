# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np 

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

from sklearn.metrics import f1_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

list_attr_celeba = pd.read_csv("../input/celeba-dataset/list_attr_celeba.csv")

list_bbox_celeba = pd.read_csv("../input/celeba-dataset/list_bbox_celeba.csv")

list_eval_partition = pd.read_csv("../input/celeba-dataset/list_eval_partition.csv")

list_landmarks_align_celeba = pd.read_csv("../input/celeba-dataset/list_landmarks_align_celeba.csv")