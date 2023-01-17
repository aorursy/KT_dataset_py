# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

insurance = pd.read_csv("../input/insurance/insurance.csv")
insurance
import matplotlib.pyplot as plt
font = {'size': 20}
plt.hist(insurance["age"], color="pink",edgecolor="blue", linewidth="4")

plt.xlabel("age")

plt.ylabel("frequency")

plt.title("Ages of Policy Holders")
#This histogram is moderatelty uniform with the exception of the early 20s agegroup, which is the mode. 

#The median of this histogram is 40, and the range is approximately 50. There are no outliers nor are there any gaps 

#in this histogram



insurance["smoker"].value_counts()
labels ="Non-Smokers","Smokers"

explode = (0,.1)

sizes =[1064, 274]

plt.pie(sizes, explode, labels, autopct="%1.1f%%")

plt.title("Smokers vs. Non-Smokers")
#This pie chart illustrates that the majority of policy holders are non-smokers, being that less than a quarter of the

#individuals sampled are smokers.
insurance["region"].value_counts()
labels ="southwest","northwest","southeast","northeast"

sizes=[325,325,364,324]

plt.bar(labels, sizes, color="pink")
#This bar chart is uniform and policy holders from the southeast are the mode. Considering that this chart contains categorical 

#data, there is no way to find a center tendancy. There are no outliers and no gaps in this data.
import seaborn as sns

sns.boxplot(x="bmi", data=insurance, color="pink")

plt.xlabel("Body mass index")

plt.title("BMI of Insurance Policy Holders")

#This boxplot has a median of approx. 31 and a range of 32. There are multiple outliers past a bmi of 47, meaning that the mean

#would be higher than the median. The boxplot is slightly asymmetrical, considering that the distance from the median to the max

#is greater than the distance from the median to minimum. The IQR is is approximately 8.
plt.hist(insurance["age"])

plt.xlabel("age")

plt.ylabel("frequency")

plt.title("Ages of Policy Holders")
plt.hist(insurance["bmi"])

plt.xlabel("BMI")

plt.ylabel("frequency")

plt.title("BMI of Policy Holders")
#These 2 histograms are different in the sense that the first is uniform while the second is symmetrical. The second 

#does not contain any gaps, however there are outliers in the 50 BMI range. The range of the first histogram is 50, and the 

#second has a range of approximately 36.