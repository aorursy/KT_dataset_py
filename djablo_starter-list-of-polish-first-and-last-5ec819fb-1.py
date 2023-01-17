from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
male = pd.read_csv('../input/polish_male_firstnames.txt')
male
surnames = pd.read_csv('../input/polish_surnames.txt')
surnames
female = pd.read_csv('../input/polish_female_firstnames.txt')
female