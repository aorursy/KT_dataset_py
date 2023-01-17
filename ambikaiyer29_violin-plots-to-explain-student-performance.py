# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))
import warnings  
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(11.7,8.27)})
# Any results you write to the current directory are saved as output.
student_data = pd.read_csv("../input/StudentsPerformance.csv")
student_data.head()
student_data['overall_average'] = round((student_data['math score']+student_data['reading score']+student_data['writing score'])/3.0,2)
# sns.distplot(student_data.groupby(['gender']).reset_index().overall_average)
sns.violinplot(x="gender", y="overall_average", hue="lunch",split=True,data=student_data)
sns.violinplot(x="gender", y="overall_average", hue="parental level of education",data=student_data)
sns.catplot(x="gender", y="overall_average",
                hue="lunch", col="parental level of education",split=True,
               data=student_data, kind="violin");
sns.violinplot(x="test preparation course", y="overall_average", hue="gender",split=True,data=student_data)
sns.violinplot(x="race/ethnicity", y="overall_average", hue="gender",split=True,data=student_data)