import pandas as pd

df = pd.read_csv("../input/data-scientist-skills/skills.csv", sep=';')

df
df.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt



labels = ['Degree not required', 'Bachelor', 'Masters', 'PhD']

degree_not_required_count = len(df)-df.Bachelor.sum()-df.Masters.sum()-df.PhD.sum()

sizes = [degree_not_required_count,

         df.Bachelor.sum(),

         df.Masters.sum(),

         df.PhD.sum()

        ]



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%')

ax1.axis('equal')

plt.show()
soft_skills = df[['Communication','English','German']].sum().to_frame().reset_index()

soft_skills.columns = ['Soft skill', 'count']

soft_skills['percentage']=soft_skills['count']*100/len(df)

soft_skills=soft_skills.sort_values(by=['percentage'], ascending=False)

soft_skills
ax = sns.barplot(x='Soft skill', y='percentage',data=soft_skills)

ax.set(ylim=(0, 100))
columns_to_drop = ['Company','Country','Platform','Bachelor','Masters','PhD','Communication','English','German']

technical_skills = df.copy().drop(columns_to_drop, axis = 1)

technical_skills = technical_skills.sum().to_frame().reset_index()

technical_skills.columns = ['Technical skill', 'count']

technical_skills['percentage']=technical_skills['count']*100/len(df)

technical_skills=technical_skills.sort_values(by=['percentage'], ascending=False)

technical_skills
ax = sns.barplot(y='Technical skill', x='percentage', data=technical_skills, orient="h")

sns.set(rc={'figure.figsize':(20,16)})

ax.set(xlim=(0, 100))