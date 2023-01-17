# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

admission_data_filepath = "/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv"

admission_data = pd.read_csv(admission_data_filepath)

admission_data.head()
#How are Gre Scores?

#Firstly,Plotting scatter plot for 'GRE Score' and 'Chance of Admit'

sns.scatterplot(x=admission_data['GRE Score'],y=admission_data['Chance of Admit '])
#The above plot suggests that there is GRE scores and Admission chances are positively corelated.Thus we can say that students with high GRE scores are likely

#to have high probability of getting admission.

#To double check the strength of the relationship, We will add a regression line.

sns.regplot(x=admission_data['GRE Score'],y=admission_data['Chance of Admit '])#We get a high slope for this regression line and thus confirms our result.