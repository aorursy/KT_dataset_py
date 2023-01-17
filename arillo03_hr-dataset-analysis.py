# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn#for visuals

sn.set(style="white", color_codes=True)#customizes the graphs

import matplotlib.pyplot as mp #for visuals

%matplotlib inline

import warnings #suppress certain warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

hr_data = pd.read_csv("../input/HR_comma_sep.csv")

hr_data
hr_data.shape
hr_data.head(11)
hr_data.corr()
#Focusing on employees that are still in the company

hr_data = hr_data[hr_data.left == 1]



#Building the plot

mp.figure(figsize=(15,10))

ax = mp.axes()

ax.set_title('How employee perfomance affects their satisfaction level',fontsize=20,fontweight="bold")

ax.set_xlim(120,320)

ax.set_ylim(0,1)

mp.xlabel('Average Monthly Hours')

mp.ylabel('Satisfaction Level')



hr_data['color'] = 'none'



#Identifying employees salaries

hr_data.loc[hr_data.salary == 'low', ['color']] = 'r'

hr_data.loc[hr_data.salary == 'medium', ['color']] = 'b'

hr_data.loc[hr_data.salary == 'high', ['color']] = 'g'



#Showing the plot

mp.scatter(hr_data['average_montly_hours'], hr_data['satisfaction_level'],

	s=hr_data['number_project']**3, color=hr_data['color'], alpha=0.5)

mp.show()
