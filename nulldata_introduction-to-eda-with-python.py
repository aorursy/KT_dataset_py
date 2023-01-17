import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#import os
#print(os.listdir("../input"))
survey = pd.read_csv("../input/survey_results_public.csv")
survey.shape
survey.head()
[column for column in survey.columns]
len(survey.index)
survey.groupby("Gender").size()
survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size()
(survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size()/len(survey[survey["Gender"].isin(["Male","Female"])].index))*100
survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size().transform(lambda x: (x/sum(x))*100)
survey["Gender"].value_counts(normalize=True) * 100 
survey["Gender"][survey["Gender"].isin(["Male","Female"])].value_counts(normalize=True) * 100 
survey["Age"][1:10]
print("Total number of rows is ", len(survey["Age"]), " and \nNumber of missing values is ", np.count_nonzero(survey["Age"].isnull()))
survey.groupby('Age').size()
sns.countplot(x="Age", data=survey)
plt.style.use('ggplot')
sns.countplot(x="Age", data=survey)
sns.countplot(y="Age", data=survey)
sns.countplot(y="Age", data=survey)
plt.title("Age Count of StackOverflow Survey Data")
sns.countplot(y="Age", data=survey).set_title("Age Count of StackOverflow Survey Data")
sns.set(rc={'figure.figsize':(12,8)})
sns.countplot(y="Age", data=survey).set_title("Age Count of StackOverflow Survey Data")
sns.countplot(x="Age", hue = "JobSatisfaction", data=survey).set_title("Age Count of StackOverflow Survey Data")
filtered = survey[survey["Gender"].isin(["Male","Female"])].dropna(subset = ['Age'])
sns.set(rc={'figure.figsize':(12,8)})
sns.catplot(y="Age", hue = "JobSatisfaction", col = "Gender", data= filtered, kind="count", height=10, aspect = 0.9)
sns.set(style="whitegrid")
sns.violinplot(x="Gender", y="ConvertedSalary", data=filtered);
plt.style.use("fivethirtyeight")
sns.violinplot(y="JobSatisfaction", x="ConvertedSalary", data=filtered).set_title("Slightly More Beautiful")