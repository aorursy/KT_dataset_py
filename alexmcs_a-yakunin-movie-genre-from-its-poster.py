from sklearn import cross_validation, metrics, preprocessing 

import xgboost as xgb

import numpy as np

import pandas as pd

import cv2

import os

import matplotlib.pyplot as plt

%matplotlib inline
row_data = pd.read_csv("../input/MovieGenre.csv",encoding="ISO-8859-1", header = 0, sep = ',')
row_data.head()
row_data.info()