# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/student-mat.csv')

df = pd.DataFrame(data)

alc_weekend = df.loc('Walc')

fin_grades = df.loc('G3')

#Plotting alcohol consumption vs final grades

plt.title('Weekly alcohol consumption vs final grades in Portugese secondary students')

plt.xlabel('Alcohol')

plt.ylabel('Final Grades')

plt.plot([alc_weekend], [fin_grades])

plt.show()