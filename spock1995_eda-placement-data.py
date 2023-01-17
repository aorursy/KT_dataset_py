import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.shape
df.describe()
df.info()
df.drop('sl_no', axis=1, inplace=True);
df.head()
df.rename(columns={'ssc_p':'grade_X', 'ssc_b':'X_board', 'hsc_p':'grade_XII', 'hsc_b':'XII_board', 'hsc_s':'stream', 'degree_p':'grade_UG', 'degree_t':'field_UG', 'etest_p':'grade_PP', 'mba_p':'grade_MBA' }, inplace=True)
df.head()
df.status.value_counts()
plt.bar(df.status.unique(), df.status.value_counts(), alpha=0.5);
plt.title('Placement Status');
plt.xlabel('Status');
plt.ylabel('Number of students');

for i,v in enumerate(df.status.value_counts()):
    plt.text(i-0.05,v-10,v);
    plt.text(i-0.08,v-25,str(round((v/df.shape[0])*100,2))+'%');
plt.bar(df.gender.unique(), df.gender.value_counts(), alpha=0.5);
plt.title('Gender');
plt.xlabel('Gender');
plt.ylabel('Number of students');

for i,v in enumerate(df.gender.value_counts()):
    plt.text(i-0.2,v-10,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.X_board.unique(), df.X_board.value_counts(), alpha=0.5);
plt.title('X_board');
plt.xlabel('X_board');
plt.ylabel('Number of students');

for i,v in enumerate(df.X_board.value_counts()):
    plt.text(i-0.2,v-10,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.XII_board.unique(), df.XII_board.value_counts(), alpha=0.5);
plt.title('XII_board');
plt.xlabel('XII_board');
plt.ylabel('Number of students');

for i,v in enumerate(df.XII_board.value_counts()):
    plt.text(i-0.2,v-10,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.stream.unique(), df.stream.value_counts(), alpha=0.5);
plt.title('stream');
plt.xlabel('stream');
plt.ylabel('Number of students');

for i,v in enumerate(df.stream.value_counts()):
    plt.text(i-0.30,v-8,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.field_UG.unique(), df.field_UG.value_counts(), alpha=0.5);
plt.title('field_UG');
plt.xlabel('field_UG');
plt.ylabel('Number of students');

for i,v in enumerate(df.field_UG.value_counts()):
    plt.text(i-0.30,v-8,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.workex.unique(), df.workex.value_counts(), alpha=0.5);
plt.title('workex');
plt.xlabel('workex');
plt.ylabel('Number of students');

for i,v in enumerate(df.workex.value_counts()):
    plt.text(i-0.2,v-10,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
plt.bar(df.specialisation.unique(), df.specialisation.value_counts(), alpha=0.5);
plt.title('specialisation');
plt.xlabel('specialisation');
plt.ylabel('Number of students');

for i,v in enumerate(df.specialisation.value_counts()):
    plt.text(i-0.2,v-10,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
sns.distplot(df.grade_X);
df.grade_X.skew()
sns.boxplot(df.grade_X);
sns.distplot(df.grade_XII);
df.grade_XII.skew()
sns.boxplot(df.grade_XII);
q3, q1 = np.percentile(df.grade_XII, [75,25])
q3-q1

min = q1-(1.5)*(q3-q1)
max = q3+(1.5)*(q3-q1)

for i,x in enumerate(df.grade_XII):
    if x<min:
        print(i,x,"Min")
    elif x>max:
        print(i,x,"Max")
    
q3, q1 = np.percentile(df.grade_XII, [75,25])
q3-q1
min = q1-(1.5)*(q3-q1)
max = q3+(1.5)*(q3-q1)
for i,x in enumerate(df.grade_XII):
    if x<min:
        print(i,x,"Min")
    elif x>max:
        print(i,x,"Max")
    
df.iloc[206,:]
sns.distplot(df.grade_UG);
df.grade_UG.skew()
sns.boxplot(df.grade_UG)
q3, q1 = np.percentile(df.grade_UG, [75,25])
min = q1-(1.5)*(q3-q1)
max = q3+(1.5)*(q3-q1)
for i,x in enumerate(df.grade_UG):
    if x<min:
        print(i,x,"Min")
    elif x>max:
        print(i,x,"Max")
df.iloc[197,:]
sns.distplot(df.grade_MBA);
df.grade_MBA.skew()
sns.boxplot(df.grade_MBA);
sns.distplot(df.grade_PP);
df.grade_PP.skew()
sns.boxplot(df.grade_PP);
df.head(5)
gender=pd.crosstab(df.gender,df.status)
gender.div(gender.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
X=pd.crosstab(df.X_board,df.status)
X.div(X.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
XII=pd.crosstab(df.XII_board,df.status)
XII.div(XII.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
stream=pd.crosstab(df.stream,df.status)
stream.div(stream.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
stream.div(stream.sum(1).astype(float),axis=0)
field_UG=pd.crosstab(df.field_UG,df.status)
field_UG.div(field_UG.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
field_UG.div(field_UG.sum(1).astype(float),axis=0)
specialisation=pd.crosstab(df.specialisation,df.status)
specialisation.div(specialisation.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True);
specialisation.div(specialisation.sum(1).astype(float),axis=0)
df.head()
df.status.replace('Placed',1,inplace=True);
df.status.replace('Not Placed',0,inplace=True);
df.head()
df.corr()
sns.heatmap(df.corr(),vmax=.8, cmap="BuPu");
df_new = df.copy()
df_new.head()
df_new.drop([24,42,49,120,134,169,177,206,197],axis=0, inplace = True)








