# coding: utf-8

__author__ = 'ZFTurbo & anokas :)'



import datetime

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.cross_validation import KFold

from sklearn.metrics import roc_auc_score

from scipy.io import loadmat

from operator import itemgetter

import random

import os

import time

import glob



random.seed(2016)

np.random.seed(2016)






