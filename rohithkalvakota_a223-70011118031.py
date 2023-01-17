import pandas as pd
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv", index_col=["race/ethnicity"])
df.shape
df.head(3)
df.iloc[1]
df.loc["group B"]
df.loc[["group A","group B","group C"]]
df.iloc[[5,6,28]]
df[:3]
df[3:6]
df.columns
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)

df["parental_level_of_education"].head(10)
df.parental_level_of_education.head(10)
df[["parental_level_of_education","lunch"]][:3]
df.lunch.iloc[2]
df.lunch.iloc[[2]]
(df.gender=="female").head()
df[df.gender=="female"].head()
df[df.lunch.isin(["standard"])].head()
(df["gender"].value_counts().head(10)/len(df)).plot.bar()
df["parental_level_of_education"].value_counts().sort_index().plot.bar()
df['parental_level_of_education'].value_counts().sort_index().plot.area()
df['math_score'].plot.hist()
import seaborn as sns
sns.countplot(df['lunch'])