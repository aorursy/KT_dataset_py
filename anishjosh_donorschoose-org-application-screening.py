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

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
sample = train_data.iloc[:,:30]
print("shape of the data : {}".format(train_data.shape))
#hypothesis test of teacher_id and approval
teacher_id_unique = train_data.teacher_id.unique()
# len(teacher_id_unique)
teacher_ids = train_data.teacher_id.value_counts()
try:
    for i in range(5):
        split_by_high = train_data[train_data["teacher_id"] == teacher_ids.keys()[i]]
        print("ID: ",train_data["teacher_id"][i]," approved:",split_by_high.project_is_approved.value_counts()[1])
        print("unapproved:",split_by_high.project_is_approved.value_counts()[0])
except:
        "KeyError: 0"
#     print("ID: ",train_data["teacher_id"][i]," approved:",split_by_high.project_is_approved.value_counts()[1])
#     print("unapproved:",split_by_high.project_is_approved.value_counts()[0])

preview = train_data.columns
xp = train_data.project_grade_category.value_counts()
perc = []
for i in xp.keys():
    max_acceptance_grades = train_data[train_data["project_grade_category"] == i].project_is_approved
    vc_grades_co = max_acceptance_grades.value_counts()
    calc_perc = vc_grades_co.values[0]/np.sum(vc_grades_co.values[:])*100
    print("In Grade",i," the acceptance rate is: {}%".format(calc_perc))
    perc.append(calc_perc)
labels = xp.keys()
sizes = perc
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.2, 0, 0) 
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=150)
 
plt.axis('equal')
plt.show()
