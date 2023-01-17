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
titanic = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic[:3]
total_number_passenger = len(titanic)

total_number_passenger
print("จำนวนคนทั้งหมดที่มีใน Dataset {}".format(len(titanic)))
print("จำนวนคนทั้งหมดที่ ขึ้นเรือ Pclass 1: {} คน".format(len(titanic[titanic["Pclass"] == 1])))
titanic["Sex"].unique()
titanic["Survived"].unique()
answer_2 = len(titanic[(titanic["Sex"] == 'female') & (titanic["Survived"] == 0)])

answer_2
print("มีผู้หญิงตายทั้งหมด: {} คน".format(answer_2))
print("มีผู้หญิงตายทั้งหมด: {:.2f} %".format(answer_2 *100 / total_number_passenger))
answer_3 = len(titanic[(titanic["Sex"] == 'male') & (titanic["Survived"] == 0)])

answer_3
print("มีผู้ชายตายทั้งหมด: {} คน".format(answer_3))

print("มีผู้ชายตายทั้งหมด: {:.2f} %".format(answer_3 *100 / total_number_passenger))
most_dead_age = titanic[titanic["Survived"] == 0]["Age"].mode()[0]

most_dead_age
print("อายุใดที่มีการตายเยอะที่สุดึคือ {} ปี".format(most_dead_age))
female_avg_dead_age = titanic[(titanic["Survived"] == 0) & (titanic["Sex"] == "female")]["Age"].mean()

female_avg_dead_age
male_avg_dead_age = titanic[(titanic["Survived"] == 0) & (titanic["Sex"] == "male")]["Age"].mean()

male_avg_dead_age
print("อายุเฉลี่ยของ เพศ ญ ที่ตาย {:.2f} ปี".format(female_avg_dead_age))

print("อายุเฉลี่ยของ เพศ ช ที่ตาย {:.2f} ปี".format(male_avg_dead_age))
titanic.isnull().sum()
avg_price_pclass = titanic.groupby("Pclass")[["Fare"]].mean()

avg_price_pclass
avg_price_pclass_dead = titanic[titanic["Survived"] == 0].groupby("Pclass")[["Fare"]].mean()

avg_price_pclass_dead
avg_price_pclass_survived = titanic[titanic["Survived"] == 1].groupby("Pclass")[["Fare"]].mean()

avg_price_pclass_survived
temp_df01 = titanic.groupby(["Pclass"])[["Fare"]].mean().rename(columns={"Fare":"all_passengers_avg_fare"})

temp_df02 = titanic[titanic["Survived"] == 0].groupby(["Pclass"])[["Fare"]].mean().rename(columns={"Fare":"dead_passengers_avg_fare"})

temp_df03 = titanic[titanic["Survived"] == 1].groupby(["Pclass"])[["Fare"]].mean().rename(columns={"Fare":"survived_passengers_avg_fare"})





represent_q10 = pd.concat([temp_df01, temp_df02, temp_df03], axis=1)

represent_q10.style.background_gradient(axis=1, cmap='YlOrRd')
titanic.columns
tmp_embark01 = titanic.groupby(["Survived", "Embarked"])[["Embarked"]].count().rename(columns={"Embarked":"cnt_passenger"})

tmp_embark01["percent_survived"] = tmp_embark01["cnt_passenger"] * 100 / len(titanic) 

tmp_embark01.style.background_gradient(axis=0, cmap='YlOrRd')
tmp_embark02 = titanic.groupby(["Survived", "Embarked"])[["Embarked"]].count().rename(columns={"Embarked":"cnt_passenger"})

tmp_embark02["cnt_pas_in_port"] = [75+93, 47+30, 427+217, 75+93, 47+30, 427+217]

tmp_embark02["percent_survived"] = tmp_embark02["cnt_passenger"] * 100 / tmp_embark02["cnt_pas_in_port"]

tmp_embark02.style.background_gradient(axis=0, cmap='YlOrRd')
tmp_avg_fare = titanic.groupby(["Survived", "Embarked"])[["Fare"]].mean().rename(columns={"Fare":"avg_fare"})

tmp_avg_fare.style.background_gradient(axis=0, cmap='YlOrRd')
lowest_age_idx = titanic["Age"].idxmin()

print("แถวที่พบอายุคนน้อยที่สุด: {}".format(lowest_age_idx))
maxest_age_idx = titanic["Age"].idxmax()

print("แถวที่พบอายุคนน้อยที่สุด: {}".format(maxest_age_idx))
titanic[titanic.index == lowest_age_idx ]
titanic[titanic.index == maxest_age_idx ]
titanic[["Age"]].describe()
cnt_zero_sibsp_passengers = titanic["SibSp"].ne(0).sum()

print("จำนวนผู้โดยสารที่ขึ้นเรือมาโดยที่ SibSp เท่ากับ 0: {} rows".format(cnt_zero_sibsp_passengers))
pay_largest_fare = titanic[["Name", "Fare"]].nlargest(3, "Fare")["Name"].tolist()

print("\n".join(pay_largest_fare))
pay_smallest_fare = titanic[["Name", "Fare"]].nsmallest(3, "Fare")["Name"].tolist()

print("\n".join(pay_smallest_fare))