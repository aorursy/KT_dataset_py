import numpy as np

import pandas as pd

import tensorflow as tf

from bs4 import BeautifulSoup

from functools import reduce

from IPython.display import display



# Convert csv files into dataframes

biology_pd = pd.read_csv('../input/biology.csv')

cooking_pd = pd.read_csv('../input/cooking.csv')

cryptology_pd = pd.read_csv('../input/crypto.csv')

diy_pd = pd.read_csv('../input/diy.csv')

robotics_pd = pd.read_csv('../input/robotics.csv')

travel_pd = pd.read_csv('../input/travel.csv')

test_pd = pd.read_csv('../input/test.csv')



# Print dataframe heads

print('Biology: %i questions' % biology_pd.shape[0])

print('Cooking: %i questions' % cooking_pd.shape[0])

print('Crytology: %i questions' % cryptology_pd.shape[0])

print('DIY: %i questions' % diy_pd.shape[0])

print('Robotics: %i questions' % robotics_pd.shape[0])

print('Travel: %i questions' % travel_pd.shape[0])

print('Test: %i questions' % test_pd.shape[0])
