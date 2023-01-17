import pandas as pd

sp = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

sp
sp.head()
sp.describe()
print(sp)
import datetime



x = datetime.datetime.now()
x
sp.columns
sp.iloc[0:,6].dtype
scores = sp["math score"]

print(scores)

scores.head()
scores.max()
sp.dtypes
scores.head(10)
sp.info()
All_scores = sp[["gender","reading score","math score","writing score"]]

All_scores
All_scores.shape
math_passed = [sp["math score"]>=50]

math_passed
math_passed_genderwise = All_scores[All_scores["math score"]>=50]

math_passed_genderwise
failed = sp["math score"]<50

failed
#Students who passed(>=50) in writing score, actually completed the test preparation course

course_completed = sp.loc[sp["writing score"]>=50,"test preparation course"]

course_completed

course_completed
All_scores.plot()
All_scores["math score"].plot()
All_scores.plot.scatter(x="reading score",y="writing score",alpha=0.5)
sp.plot.box()
axs = sp.plot.area(figsize=(20,16),subplots = True)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize=(20,16));



sp.plot.area(ax=axs);



axs.set_ylabel("Scores for different Subjects");



fig.savefig("Scores.png")
#Mean of math scores

sp["math score"].mean()
sp["writing score"].mean()
sp["reading score"].mean()
sp[["math score","reading score","writing score"]].describe()
sp.agg({'math score': ['min','mean','max', 'median', 'skew'],

             'writing score': ['min', 'max', 'median', 'mean','skew']})
#Average math score for male and female comparison

sp[["gender","math score"]].groupby("gender").mean()
sp[["test preparation course","math score","reading score","writing score"]].groupby("test preparation course").mean()

a = sp[["race/ethnicity","math score","reading score","writing score"]].groupby("race/ethnicity").mean()

a
a.plot()
b = sp[["parental level of education","math score","reading score","writing score"]].groupby("parental level of education").mean()

b
b.plot(figsize=(20,16))
c = sp[["lunch","math score","reading score","writing score"]].groupby("lunch").mean()

c
c.plot()
sp["gender"].value_counts()
sp["parental level of education"].value_counts()
sp["test preparation course"].value_counts()
sp["lunch"].value_counts()
sp["race/ethnicity"].value_counts()