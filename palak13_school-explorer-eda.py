import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplot
%matplotlib inline
student = pd.read_csv('../input/2016 School Explorer.csv')
student.head(2)
student.shape
student.columns = student.columns.str.replace(' ','_')
student.columns = student.columns.str.replace('?','')
student.columns = student.columns.str.replace('%','')
student.head(1)
student = student.drop(['Adjusted_Grade','New','Other_Location_Code_in_LCGMS'], axis = 1)
student = student.dropna(how = "any")
def convert(x):
    return float(x.strip('%'))/100


student['Percent_Asian'] = student['Percent_Asian'].astype(str).apply(convert)
student['Percent_White'] = student['Percent_White'].astype(str).apply(convert)
student['Percent_Black'] = student['Percent_Black'].astype(str).apply(convert)
student['Percent_Hispanic'] = student['Percent_Hispanic'].astype(str).apply(convert)
student['Percent_of_Students_Chronically_Absent'] = student['Percent_of_Students_Chronically_Absent'].astype(str).apply(convert)
student['Trust_'] = student['Trust_'].astype(str).apply(convert)
f, axes = plt.subplots(ncols=4, figsize=(20, 6))

sns.distplot(student['Percent_Hispanic'], kde=False, color="b", ax=axes[0], bins=35).set_title('No. of Hispanic Distribution (%)')
sns.distplot(student['Percent_Black'], kde=False, color="g", ax=axes[1], bins=35).set_title('No. of Black Distribution (%)')
sns.distplot(student['Percent_Asian'], kde=False, color="y", ax=axes[2], bins=25).set_title('No. of Asian Distribution (%)')
sns.distplot(student['Percent_White'], kde=False, color="r", ax=axes[3], bins=25).set_title('No. of White Distribution (%)')

plt.show()
sns.stripplot(y="Effective_School_Leadership_Rating", x="Percent_of_Students_Chronically_Absent", data=student)
plt.show()
temp = sns.distplot(student[['Economic_Need_Index']].values, kde=False, color = 'c')
temp= plt.title("ENI distribution")
temp = plt.xlabel("ENI")
temp = plt.ylabel("No. of Schools")
plt.show()
df = student.groupby('Community_School')['Trust_'].mean()
print(df.head())