#load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
#Load Dataset

ed=pd.read_csv('../input/xAPI-Edu-Data.csv')
ed.head()
ed.describe()
ed.shape
ed.info()
ed['Class'].value_counts()
ed.groupby(['gender'])['Class'].count().reset_index()
ed.groupby(['NationalITy'])['Class'].count().reset_index().sort(['Class'],ascending=False)
ed.groupby(['GradeID'])['Class'].count().reset_index().sort(['GradeID'])
ed.groupby(['GradeID','Semester'])['Class'].count().reset_index().sort(['GradeID'])