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
reason=pd.read_csv("/kaggle/input/suicides-in-sri-lanka/reasons_of_suicides.csv")

nature=pd.read_csv("/kaggle/input/suicides-in-sri-lanka/nature_of_occupation.csv")
employ=nature[["Age group","2014_Unemployed persons_M","2014_Unemployed persons_F"]]

employ
over71=employ[employ["Age group"]=="Over 71 Yrs"]

total=employ[employ["Age group"]=="Total"]

print("Female Without over 71: ",float(total["2014_Unemployed persons_F"])-float(over71["2014_Unemployed persons_F"]))

print("Male Without over 71: ",float(total["2014_Unemployed persons_M"])-float(over71["2014_Unemployed persons_M"]))
nature.columns[0:20]
srilankapopulation=20359439
placeholder="2017_Professional Technical & related workers (Doctors/Engineers/Accountants/ Teachers/Authors/ Photographers)_"

profes=nature[["Age group",placeholder+"M",placeholder+"F"]]

profes
placeholder="2017_Clerical & related workers (Stenographers/ Typists etc)_"

clerics=nature[["Age group",placeholder+"M",placeholder+"F"]]

clerics
placeholder1="2017_Armed Services_"

workwithgun=nature[["Age group",placeholder1+"M",placeholder1+"F"]]

placeholder2="2017_Police_"

workwithgun2=nature[["Age group",placeholder+"M",placeholder+"F"]]

male=workwithgun[placeholder1+"M"]+workwithgun2[placeholder2+"M"]

female=workwithgun[placeholder1+"F"]+workwithgun2[placeholder2+"F"]

armed=pd.DataFrame({

    "Age Group":workwithgun["Age group"],

    "male":male,

    "female":female

})

armed
nature.groupby(by="Age group").sum()["2014_Totals_M"]
nature.groupby(by="Age group").sum()["2014_Totals_F"]
education=pd.read_csv("/kaggle/input/suicides-in-sri-lanka/Education_level.csv")

education.columns
placeholder="2017_University Degree or above_"

clerics=education[["Age group",placeholder+"M",placeholder+"F"]]

clerics
placeholder="2016_University Degree or above_"

clerics=education[["Age group",placeholder+"M",placeholder+"F"]]

clerics