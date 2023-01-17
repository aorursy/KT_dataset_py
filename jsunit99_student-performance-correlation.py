import pandas as pd

import numpy as np



from sklearn.preprocessing import OneHotEncoder



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
encoder=OneHotEncoder(sparse=False)



df_encoded = pd.DataFrame(encoder.fit_transform(df[['gender']]))

df_encoded.columns = encoder.get_feature_names(['gender'])

df.drop(['gender'] ,axis=1, inplace=True)

df_new= pd.concat([df, df_encoded], axis=1)

df_new
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df_new.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
df_new.groupby(["test preparation course"])["math score", "reading score", "writing score"].mean()
df_new.groupby(["parental level of education"])["math score", "reading score", "writing score"].mean()
df_new.groupby(["lunch"])["math score", "reading score", "writing score"].mean()
df_new.groupby(["race/ethnicity"])["math score", "reading score", "writing score"].mean()
df_new['test preparation course'].value_counts()
df_new['lunch'].value_counts()
df_new.groupby(["lunch", "test preparation course"])["math score", "reading score", "writing score"].mean()
df_encoded = pd.DataFrame(encoder.fit_transform(df_new[['lunch']]))

df_encoded.columns = encoder.get_feature_names(['lunch'])

df_new.drop(['lunch'] ,axis=1, inplace=True)

df_new = pd.concat([df_new, df_encoded], axis=1)

df_new
df_encoded = pd.DataFrame(encoder.fit_transform(df_new[['test preparation course']]))

df_encoded.columns = encoder.get_feature_names(['test preparation course'])

df_new.drop(['test preparation course'] ,axis=1, inplace=True)

df_new = pd.concat([df_new, df_encoded], axis=1)

df_new
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df_new.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()