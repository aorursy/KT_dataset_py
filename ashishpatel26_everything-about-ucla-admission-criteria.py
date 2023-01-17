# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter("ignore")



plt.style.use("seaborn-muted")





tableau_20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),

         (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),

         (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),

         (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),

         (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]



# Scaling avove RGB values to [0, 1] range, which is Matplotlib acceptable format:

for i in range(len(tableau_20)):

    r, g, b = tableau_20[i]

    tableau_20[i] = (r / 255., g / 255., b / 255.)



# Any results you write to the current directory are saved as output.
print(plt.style.available)
admission  = pd.read_csv("../input/Admission_Predict.csv")

admission_v1 = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
display(admission.head())

display(admission_v1.head())
display(admission.isnull().sum())
display(admission_v1.isnull().sum())
admission.shape
admission.index
admission.columns.to_frame().T
admission.count().to_frame().T
admission.info(verbose=True)
admission_v1.shape
admission_v1.index
admission_v1.columns.to_frame().T
admission_v1.count().to_frame().T
admission_v1.info(verbose=True)
admission.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Admissions")
admission_v1.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="PiYG")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Admissions v1.1")
sns.set(color_codes=True)

sns.pairplot(admission_v1, kind="scatter", palette="Set2")
a = admission_v1.pop('Serial No.')

del a
plt.figure(figsize=(20,25))

i = 0



for item in admission_v1.columns:

    i += 1

    plt.subplot(4, 2, i)

    sns.distplot(admission_v1[item], rug=True, rug_kws={"color": "m"},kde=True,

                 kde_kws={"color": "red", "lw": 3, "label": "KDE"},

                 hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "orange"},label="{0}".format(item))

#     sns.distplot(admission_v1[item], kde=True,label="{0}".format(item))



plt.show()
admission_v1.columns
plt.figure(figsize=(20,25))

sns.jointplot(x="GRE Score", y=admission_v1['Chance of Admit '], data=admission_v1, kind="reg", height=12,color='m')

plt.xlabel("\nClutser of GRE Score is Belong to  300 to 330\n and Highest Admission of Score Range in Between 310 to 320")

plt.show()
plt.figure(figsize=(20,25))

sns.jointplot(x="TOEFL Score", y=admission_v1['Chance of Admit '], data=admission_v1, kind="reg", height=12,color='Red')

plt.xlabel("\nClutser of TOEFL Score Most value Belongs 97 to 120\n and Highest Student of Admission Belong in Between 105 to 110")

plt.show()
plt.figure(figsize=(20,25))

sns.jointplot(x="CGPA", y=admission_v1['Chance of Admit '], data=admission_v1, kind="scatter", height=12, color=tableau_20[14])

plt.show()