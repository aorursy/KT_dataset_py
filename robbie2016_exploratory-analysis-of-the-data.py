# Importing libraries and the dataset

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorsys
plt.style.use('seaborn-talk')

df = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv", sep=',')
df.Age.hist(bins=100)
plt.xlabel("Age")
plt.title("Distribution of Age")
plt.show()
labels = df.Gender.value_counts().index
N = len(df.EmploymentField.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
patches, texts = plt.pie(df.Gender.value_counts(), colors=RGB_tuples, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(patches, labels, bbox_to_anchor=(1.05,1))
plt.title("Gender")
plt.show()
N = len(df.JobRoleInterest.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
labels = df.JobRoleInterest.value_counts().index
colors = ['OliveDrab', 'Orange', 'OrangeRed', 'DarkCyan', 'Salmon', 'Sienna', 'Maroon', 'LightSlateGrey', 'DimGray']
patches, texts = plt.pie(df.JobRoleInterest.value_counts(), colors=RGB_tuples, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(patches, labels, bbox_to_anchor=(1.25, 1))
plt.title("Job Role Interest")
plt.show()
N = len(df.EmploymentField.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
labels = df.EmploymentField.value_counts().index
patches, texts = plt.pie(df.EmploymentField.value_counts(), colors=RGB_tuples, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(patches, labels, bbox_to_anchor=(1.3, 1))
plt.title("Employment Field")
plt.show()
df4 = pd.crosstab(df_ageranges.EmploymentField,df_ageranges.IsUnderEmployed).apply(lambda r: r/r.sum(), axis=1)
df4 = df4.sort_values(by=1.0)
N = len(df_ageranges.EmploymentField.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
ax1 = df4.plot(kind="bar", stacked=True, color= RGB_tuples, title="Under-employed per Employment Field")
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines,["No", "Yes"], bbox_to_anchor=(1.51, 1))
df_ageranges = df.copy()
bins=[0, 20, 30, 40, 50, 60, 100]
df_ageranges['AgeRanges'] = pd.cut(df_ageranges['Age'], bins, labels=["< 20", "20-30", "30-40", "40-50", "50-60", "< 60"]) 
df2 = pd.crosstab(df_ageranges.AgeRanges,df_ageranges.JobPref).apply(lambda r: r/r.sum(), axis=1)
N = len(df_ageranges.AgeRanges.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
ax1 = df2.plot(kind="bar", stacked=True, color= RGB_tuples, title="Job preference per Age")
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines,labels, bbox_to_anchor=(1.51, 1))
df3 = pd.crosstab(df_ageranges.AgeRanges,df_ageranges.JobRelocateYesNo).apply(lambda r: r/r.sum(), axis=1)
N = len(df_ageranges.AgeRanges.value_counts().index)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
ax1 = df3.plot(kind="bar", stacked=True, color=RGB_tuples, title="Relocation per Age")
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines,["No", "Yes"], loc='best')
df_aux = df.copy()
values = df_aux.BootcampName.value_counts(dropna=True)
values = values[values > 10]
values = values[values.index != 'Free Code Camp is not a bootcamp - please scroll up and change answer to "no"']
index = np.arange(0, len(values.index))
plt.bar(index, values)
plt.xticks(index + 0.5, values.index, rotation="vertical")
plt.xlim(0, len(values))
plt.title("Bootcamp preference")
plt.show()
#df_aux2 = df.copy()
#df_aux2["values"] = pd.Series()
#values2 = df_aux2.BootcampName.value_counts(dropna=True)
#values2 = values2[values2 > 10]
#df_aux2.values2 = values2
#df_aux2.values2 = df_aux2.values2[values2.index != 'Free Code Camp is not a bootcamp - please scroll up and change answer to "no"']

# Revisar
df_aux = df_aux[df_aux.BootcampFinish == 1.0]
df10 = pd.crosstab(df_aux.BootcampName,df_aux.BootcampRecommend).apply(lambda r: r/r.sum(), axis=1)
df10 = df10.ix[values.index,:] 
df10 = df10
df10.plot(kind="bar", stacked=True, title="")

df5 = df.copy()
df5 = df5.dropna(subset=["ExpectedEarning"])
df5 = df5[df['MoneyForLearning'].isin(range(0,60000))]

x = df5.MoneyForLearning
y = df5.ExpectedEarning

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Money for learning")
plt.ylabel("Expected earning")
plt.title("Money for learning vs Expected earning")
plt.show()
df6 = df.copy()
df6 = df6.dropna(subset=["Income"])
df6 = df6[df['MoneyForLearning'].isin(range(0,60000))]

x = df6.Income
y = df6.MoneyForLearning

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.title("Income vs Money for learning")
plt.xlabel("Income")
plt.ylabel("Money for learning")
plt.show()
df7 = df.copy()
df7 = df7.dropna(subset=["HoursLearning"])
df7 = df7.dropna(subset=["ExpectedEarning"])

x = df7.HoursLearning
y = df7.ExpectedEarning

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Hours learning")
plt.ylabel("Expected earning")
plt.title("Hours learning vs Expected earning")
plt.show()
df8 = df.copy()
df8 = df8.dropna(subset=["HoursLearning"])
df8 = df8[df['MonthsProgramming'].isin(range(0,500))]

x = df8.MonthsProgramming
y = df8.HoursLearning

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Months Programming")
plt.ylabel("Hours learning")
plt.title("Months programming vs Hours learning")
plt.show()
df9 = df.copy()
df9 = df9.dropna(subset=["HoursLearning"])
df9 = df9[df['Age'].isin(range(0,70))]

x = df9.Age
y = df9.HoursLearning

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Age")
plt.ylabel("Hours learning")
plt.title("Age vs Hours learning")
plt.show()