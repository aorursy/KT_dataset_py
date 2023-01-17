# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/StudentsPerformance.csv")
print("Data Loaded")
df.head()
df.rename(columns={'parental level of education': 'parental_level_of_education', 'test preparation course': 'test_preparation_course', 'math score': 'math_score', 'reading score': 'reading_score', 'writing score': 'writing_score'}, inplace = True)
# PLOT THE GENDERS OF THE DATASET
df['gender'].value_counts().plot.bar()
df['parental_level_of_education'].value_counts().plot.bar()
#sns.countplot(x=df.parental_level_of_education)
df.groupby("parental_level_of_education", as_index=True)[["math_score", "reading_score", "writing_score"]].mean()
score_grouped = df.groupby("parental_level_of_education", as_index=True)[["math_score", "reading_score", "writing_score"]].mean().sort_values(by='writing_score',ascending=False)
score_grouped.plot.bar(title = "Students Score Average per Parental Level of Education", figsize=(20,10))
with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "math_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Math Score");

with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "writing_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Writing Score");
with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "reading_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Reading Score");
df["overall_score"] = np.nan
df.head()
df["overall_score"] = round((df["math_score"] + df["writing_score"] + df["reading_score"]) / 3, 2)
df.head()
df.groupby(["gender", "test_preparation_course"]).size()
df.groupby(["gender", "lunch"]).size()
gender_test_preparation = df.groupby(["gender", "test_preparation_course"]).size().unstack(fill_value=0).plot.bar()
gender_test_preparation.plot(figsize=(10, 5))
gender_test_preparation = df.groupby(["gender", "lunch"]).size().unstack(fill_value=0).plot.bar()
gender_test_preparation.plot(figsize=(10, 5))
with sns.axes_style(style='ticks'):
    b = sns.catplot("test_preparation_course", "overall_score", "gender", data=df, kind="box", height=5, aspect=2)
    b.set_axis_labels("Test Preparation", "Test Scores Average")