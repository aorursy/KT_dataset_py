!pip install jovian --upgrade --quiet
!pip install jovian --upgrade --quiet

!pip install jovian opendatasets --upgrade --quiet

!pip install pandas --upgrade --quiet

import pandas as pd

import jovian

dataset_url = 'https://www.kaggle.com/nehaprabhavalkar/indian-food-101' 

import opendatasets as od

od.download(dataset_url) 
data_dir = './indian-food-101'
import os

os.listdir(data_dir)
project_name = "zerotopandas-course-project-indian-food-101"
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)


print( 'We are Loading the Indian Foods csv file for learning more about Pandas :: \n')

pd.read_csv( filepath_or_buffer =   '../input/indian-food-101/indian_food.csv' )  
