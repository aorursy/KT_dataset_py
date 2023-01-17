import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

dataset.head()
table1 = pd.crosstab(dataset.YearsSinceLastPromotion , dataset.Attrition)

table1.div(table1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de YearsSinceLastPromotion")

plt.xlabel("YearsSinceLastPromotion")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'YearsSinceLastPromotion', hue ='Attrition',data= dataset,palette = "Set2").set_title('YearsSinceLastPromotion vs Attrition')
table2 = pd.crosstab(dataset.YearsAtCompany , dataset.Attrition)

table2.div(table2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de YearsAtCompany")

plt.xlabel("YearsAtCompany")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'YearsAtCompany', hue ='Attrition',data= dataset,palette = "Set2").set_title('YearsAtCompany vs Attrition')
table2 = pd.crosstab(dataset.JobSatisfaction , dataset.Attrition)

table2.div(table2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de JobSatisfaction")

plt.xlabel("JobSatisfaction")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'JobSatisfaction', hue ='Attrition',data= dataset,palette = "Set2").set_title('JobSatisfaction vs Attrition')
table3 = pd.crosstab(dataset.Age , dataset.Attrition)

table3.div(table3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de Age")

plt.xlabel("Age")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x="Age",hue="Attrition", data=dataset,palette="Set1").set_title('Employee Age vs Attrition')
table5 = pd.crosstab(dataset.BusinessTravel  , dataset.Attrition)

table5.div(table5.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de BusinessTravel ")

plt.xlabel("BusinessTravel")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'BusinessTravel', hue ='Attrition',data= dataset,palette = "Set2").set_title(' BusinessTravel vs Attrition')
table6 = pd.crosstab(dataset.Department  , dataset.Attrition)

table6.div(table6.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de Department ")

plt.xlabel("Department")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(y = 'Department', hue ='Attrition',data= dataset,palette = "Set2").set_title(' Department vs Attrition')
table7 = pd.crosstab(dataset.DistanceFromHome  , dataset.Attrition)

table7.div(table7.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de DistanceFromHome ")

plt.xlabel("DistanceFromHome")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'DistanceFromHome', hue ='Attrition',data= dataset,palette = "Set2").set_title('DistanceFromHome vs Attrition')
table8 = pd.crosstab(dataset.EducationField  , dataset.Attrition)

table8.div(table8.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de EducationField ")

plt.xlabel("EducationField")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(y = 'EducationField', hue ='Attrition',data= dataset,palette = "Set2").set_title('EducationField vs Attrition')
table9 = pd.crosstab(dataset.OverTime  , dataset.Attrition)

table9.div(table9.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.title("Attrition en fonction de OverTime ")

plt.xlabel("OverTime")

plt.ylabel("Proportion d'employés")

plt.show()

#sns.countplot(x = 'OverTime', hue ='Attrition',data= dataset,palette = "Set2").set_title('OverTime vs Attrition')