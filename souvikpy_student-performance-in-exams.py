import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.shape
df.isnull().sum()
df.describe()
df["Total marks"] = df["math score"] + df["reading score"] + df["writing score"]
df["Percentage"] = df["Total marks"] / 3
df.head()
def Grade(marks):
    if marks >= 90:
        grade = 'A'
    elif marks >= 80:
        grade = 'B'
    elif marks >= 70:
        grade = 'C'
    elif marks >= 60:
        grade = 'D'
    elif marks >= 50:
        grade = 'E'
    else:
        grade = 'F'
    return grade
        
        
df["Grade_math"] = df["math score"].apply(lambda s: Grade(s))
df["Grade_reading"] = df["reading score"].apply(lambda s: Grade(s))
df["Grade_writing"] = df["writing score"].apply(lambda s: Grade(s))
df["Overall_grade"] = df["Percentage"].apply(lambda s: Grade(s))
df.head()
sns.set(style = "white")
sns.countplot(x = "math score", data = df)
df.Grade_math.value_counts()
order_grade = ["A","B","C","D","E","F"]
sns.countplot(x = "Grade_math", data = df, order = order_grade, palette = "GnBu_d")
_ = plt.xlabel("Grades in Mathematics")
sns.countplot(x = "reading score", data = df)
df.Grade_reading.value_counts()
sns.countplot(x= "Grade_reading",data = df, order = order_grade, palette = "BuGn_r")
_ = plt.xlabel("Grades in reading")
sns.countplot(x = "writing score", data = df)
df.Grade_writing.value_counts()
sns.countplot(x = "Grade_writing", data = df, order = order_grade, palette = sns.light_palette("navy", reverse=True))
_ = plt.xlabel("Grades in writing")
df.Overall_grade.value_counts()
sns.countplot(x = 'Overall_grade', order = order_grade, data = df, palette = 'Paired')
_ = plt.xlabel("Overall Grade")
sns.relplot(x='reading score', y = 'writing score', data = df)
sns.relplot(x='writing score', y = 'reading score', data = df)
r = np.corrcoef(df["reading score"], df["writing score"])[0, 1]
print(r)
df["race/ethnicity"].value_counts()
sns.set(style = "ticks")
order_race = ["group A","group B", "group C", "group D", "group E"]
sns.boxplot(x = "Percentage", y = "race/ethnicity", data = df, palette = "vlag", order = order_race)
sns.swarmplot(x = "Percentage", y = "race/ethnicity", data = df, size = 2, color = ".3", linewidth = 0, order = order_race)
sns.despine(trim = True, left = True)
sns.set(style = "whitegrid")
sns.violinplot(x= 'race/ethnicity', y = 'Percentage', data = df, palette = "Set3", order = order_race)
df["lunch"].value_counts()
sns.boxenplot(x="lunch", y="Percentage", data=df)
sns.countplot(x = "lunch", data = df, hue = "Overall_grade", hue_order = ["A","B","C","D","E","F"], palette = "Paired")

df["test preparation course"].value_counts()
sns.boxenplot(x='test preparation course', y='Percentage',data = df, palette = "hls")
sns.countplot(x = "test preparation course", hue = "Overall_grade",data = df, hue_order = order_grade, palette = 'Paired')
_ = plt.legend()
df["parental level of education"].value_counts()
order_edu = ['some high school','high school',"associate's degree","some college","bachelor's degree","master's degree"]
p = sns.countplot(x='parental level of education', hue='Overall_grade',data=df, order= order_edu, hue_order = order_grade, palette = 'Paired')
_ = plt.xlabel('Parents level of education')
_ = plt.setp(p.get_xticklabels(), rotation = 60)
q = sns.violinplot(x="parental level of education", y="Percentage", data = df, order = order_edu,palette = "Paired")
_ = plt.setp(q.get_xticklabels(), rotation = 60)