import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(15,30)})
sns.set(font_scale=3)
%matplotlib inline
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df.head()
gender_race_group = df.groupby(['gender', 'race/ethnicity']).mean().reset_index()
gender_race_group.head()
sns.catplot(x = 'gender', y = 'average_score', data = gender_race_group,height=10, aspect = 2.1, kind = "bar", hue = "race/ethnicity", palette="Blues_d")
sns.catplot(x = 'race/ethnicity', y = 'average_score', hue = 'gender',height=10, aspect = 2.1, data = gender_race_group, kind='bar', palette="Blues_d")
gender_lunch_group = df.groupby(['gender', 'lunch']).mean().reset_index()
gender_lunch_group
sns.catplot(x = 'lunch', y = 'average_score',height=10, aspect = 2.1, hue = 'gender', data = gender_lunch_group, palette="Blues_d", kind = "bar")
gender_parent_group = df.groupby(['gender', 'parental level of education']).mean().reset_index()
gender_parent_group.head()
sns.catplot(x = 'gender', y = 'average_score', data = gender_parent_group,height=10, aspect = 2.1,palette="Blues_d",  hue = 'parental level of education', kind = 'bar')
gender_test_group = df.groupby(['gender', 'test preparation course']).mean().reset_index()
gender_test_group.head()
sns.catplot(x = 'gender', y = 'average_score', hue = 'test preparation course',height=10, aspect = 2.1,palette="Blues_d", data = gender_test_group, kind = 'bar')
gender_race_lunch = df.groupby(['gender', 'race/ethnicity', 'lunch']).mean().reset_index()
gender_race_lunch.head()
sns.catplot(x = 'gender', y = 'average_score', hue = 'lunch',height=10, aspect = 2.1,palette="Blues_d", data = gender_race_lunch, kind = 'bar')
race_parent_group = df.groupby(['race/ethnicity', 'parental level of education']).mean().reset_index()
sns.catplot(x = 'parental level of education', y = 'average_score', hue = 'race/ethnicity',height=10, aspect = 2.1,palette="Blues_d", data = race_parent_group, kind = 'bar').set_xticklabels(rotation=30)
