# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from functools import reduce



students = pd.read_csv("../input/enrollement_schoolmanagement_2.csv")

schools = pd.read_csv("../input/managementwise_schools_0.csv")

teachers = pd.read_csv("../input/no.ofteachers_0.csv")



students["District"] = [d.title() if "Total" not in d else "Total" for d in students["District"]]

teachers["District"] = [d.title() if "Total" not in d else "Total" for d in teachers["District"]]

schools["District"] = [d.title() if "Total" not in d else "Total" for d in schools["District"]]



ts = pd.merge(teachers,students,on='District', how='outer')

student_ratios = pd.DataFrame(ts, columns=["District"])

#print(ts.columns)



student_ratios["Government"] = ts["Govt Total"] / ts["Govt"]

student_ratios["Aided"] = ts["Private Aided Total"] / ts["Pvt Aided"]

student_ratios["Private"] = ts["Private Unaided Total"] / ts["Pvt Unaided"]

student_ratios["Total"] = ts["Grand Total "] / ts["Total Teachers"]



print(student_ratios)

student_ratios.to_csv("student_teacher_ratios.csv")



ax = student_ratios.plot(title="Student to Teacher Ratio", figsize=(15,7.5), fontsize=11, grid=True)

ax.set_xticks(student_ratios.index)

ax.set_xticklabels(student_ratios["District"], rotation=90)

ax.get_figure().savefig("student_teacher_ratios.png")