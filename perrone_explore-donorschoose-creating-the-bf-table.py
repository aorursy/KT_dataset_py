# https://www.kaggle.com/donorschoose/io/data

%matplotlib inline
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

donors = pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID')
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID")
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID")

projects_light = projects.drop(columns='Project Essay',axis=1)
donations = pd.read_csv('../input/Donations.csv')
megadf = donations.join(projects_light,on='Project ID',how='left')
megadf = megadf.join(donors,on='Donor ID')
megadf = megadf.join(schools,on='School ID')
megadf
