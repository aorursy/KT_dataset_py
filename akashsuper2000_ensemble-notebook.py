!pip install -q efficientnet



import os

import re

import numpy as np

import pandas as pd

import random

import math

import time

import matplotlib.pyplot as plt



from sklearn import metrics

from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

import dill

from tensorflow.keras import backend as K



import warnings

warnings.filterwarnings('ignore')
sub0 = pd.read_csv('/kaggle/input/melanoma-classification-weights/submission_fold0.csv')

sub0.sort_values(['image_name'], inplace=True)

sub0
sub1 = pd.read_csv('/kaggle/input/melanoma-classification-weights/submission_fold1.csv')

sub1.sort_values(['image_name'], inplace=True)

sub1
sub2 = pd.read_csv('/kaggle/input/melanoma-classification-weights/submission_fold2.csv')

sub2.sort_values(['image_name'], inplace=True)

sub2
sub3 = pd.read_csv('/kaggle/input/melanoma-classification-weights/submission_fold3.csv')

sub3.sort_values(['image_name'], inplace=True)

sub3
sub4 = pd.read_csv('/kaggle/input/melanoma-classification-weights/submission_fold4.csv')

sub4.sort_values(['image_name'], inplace=True)

sub4
sub = sub0.copy()

sub['target'] = (sub0['target']+sub1['target']+sub2['target']+sub3['target']+sub4['target'])/5

sub
print('Generating submission file...')

sub.to_csv('submission_fold_ensemble.csv', index=False)