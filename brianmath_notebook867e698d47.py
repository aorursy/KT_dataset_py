!pip install jovian opendatasets --upgrade --quiet
# Change this
dataset_url = 'https://www.kaggle.com/spscientist/students-performance-in-exams' 
import opendatasets as od
od.download(dataset_url)
# Change this
data_dir = './data-analysis-course'
import os
os.listdir(data_dir)
project_name = "zerotopandas-course-project-starter" # change this (use lowercase letters and hyphens only)
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)







import jovian
jovian.commit()
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'










import jovian
jovian.commit()















import jovian
jovian.commit()
import jovian
jovian.commit()
import jovian
jovian.commit()
