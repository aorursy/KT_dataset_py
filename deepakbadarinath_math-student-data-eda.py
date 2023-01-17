



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



url='/kaggle/input/math-students/student-mat.csv'

df=pd.read_csv(url)

df.head()
df.describe()
df.info()
def pass_classify(row):

    if row.G3>=10:

        return 1

    else:

        return 0

    

pass_=df.apply(pass_classify,axis='columns')

#print(pass_fail)

print(pass_.value_counts())
def grade_classify(row):

    if row.G3>=16:

        return 'A'

    elif row.G3>=14:

        return 'B'

    elif row.G3>=12:

        return 'C'

    elif row.G3>=10:

        return 'D'

    else:

        return 'F'

    



grades=df.apply(grade_classify,axis='columns')

print(grades.value_counts())

def get_percent(col):

    return (col.value_counts()/col.value_counts().sum())*100

pass_percent=get_percent(pass_)

print(pass_percent)
grade_percent=get_percent(grades)

print(grade_percent)
sns.distplot(a=df['G3'], kde=False)
df['grades']=grades

print(grades)
df.info()
df_cat = df[[i for i in df.columns if i not in ('G1','G2','G3','absences')]]

df_cat.head()
from sklearn import preprocessing



label = preprocessing.LabelEncoder()

data_encoded = pd.DataFrame() 



for i in df_cat.columns :

  data_encoded[i]=label.fit_transform(df_cat[i])
data_encoded.head()
from scipy.stats import chi2_contingency

import numpy as np









def cramers_V(var1,var2) :

  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building

  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test

  obs = np.sum(crosstab) # Number of observations

  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table

  return (stat/(obs*mini))
rows= []



for var1 in data_encoded:

  col = []

  for var2 in data_encoded :

    cramers =cramers_V(data_encoded[var1], data_encoded[var2]) # Cramer's V test

    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  

  rows.append(col)

  

cramers_results = np.array(rows)

df_var = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)







df_var
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(20,20))

plt.title("Heatmap of categorical variables")

sns.heatmap(data=df_var,vmin=0, vmax=1,annot=True)



plt.show()

x1=df['G1']

x2=df['G2']

y=df['G3']

plt.figure(figsize=(10,6))

plt.title("Year 3 grade VS Year1 and Year2 grade")

g1=plt.scatter(x1,y,marker='x')

g2=plt.scatter(x2,y,marker='o')

plt.legend((g1, g2),

           ('Year1 Grade', 'Year2 Grade'),

           scatterpoints=1,

           loc='upper left',

           ncol=3,

           fontsize=8)

plt.xlabel("Marks(out of 20) of year1/year2")

plt.ylabel("Marks(out of 20) of year 3")

plt.show() 



plt.figure(figsize=(10,6))

plt.title("No of absent days VS Marks")

plt.scatter(df["absences"],df["G3"])



plt.ylabel("Marks(out of 20) of year 3")

plt.xlabel("Absent days")

plt.show()
#plt.figure(figsize=(12,7))

plt.title("Sex vs Math marks in final exam")

ax = sns.boxplot(x="sex", y="G3", data=df)

plt.ylabel("Marks in final exam")

plt.show()





plt.title("Daily drinking VS Math marks")

ax = sns.boxplot(x="Dalc", y="G3",data=df)

plt.xlabel("Daily alcohol consumption")

plt.ylabel("Marks in final exam")

plt.show()



plt.title("Quality of family relationship VS Math marks")

ax=sns.boxplot(x="famrel",y="G3",data=df)

plt.ylabel("Marks in final exam")

plt.xlabel("Quality of family relationship")

plt.show()





ax = sns.boxplot(x="romantic", y="G3", hue="goout",

                 data=df)

plt.ylabel("Marks in final exam")

plt.xlabel("Involved in a romantic relationship")

plt.show()





ax=sns.boxplot(x="reason",y="G3",data=df)

plt.ylabel("Marks in final exam")

plt.xlabel("Reason to choose this school")

plt.show()









sns.jointplot(x=df['G1'], y=df['G3'], kind="kde")

#plt.title("2D KDE plot b/w marks in first exam vs marks in final exam")

plt.xlabel('Marks in first exam')

plt.ylabel('Marks in final exam')

plt.show()







sns.jointplot(x=df['G2'], y=df['G3'], kind="kde")

#plt.title("2D KDE plot b/w marks in second exam vs marks in final exam")

plt.xlabel('Marks in second exam')

plt.ylabel('Marks in final exam')

plt.show()

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in columns



romantic_no=label_encoder.fit_transform(df['romantic'])



print(romantic_no[0])

print(df['romantic'][0])

df['romantic']=romantic_no
famsize_no=label_encoder.fit_transform(df['famsize'])

df['famsize']=1-famsize_no



activities_no=label_encoder.fit_transform(df['activities'])

df['activites']=activities_no



df['Pstatus']=label_encoder.fit_transform(df['Pstatus'])

df['nursery']=label_encoder.fit_transform(df['nursery'])

df['internet']=label_encoder.fit_transform(df['internet'])

df['higher']=label_encoder.fit_transform(df['higher'])

df['schoolsup']=label_encoder.fit_transform(df['schoolsup'])

df['famsup']=label_encoder.fit_transform(df['famsup'])

df['paid']=label_encoder.fit_transform(df['paid'])

print(df.head())
df.info()
grouped_df=df.groupby('grades')

print(grouped_df['freetime','famrel','goout','romantic','Pstatus','activities','paid'].mean())

plt.figure(figsize=(10,6))

plt.title("Relationship Quotient Vs Grade")

sns.barplot(x=["A","B","C","D","F"], y=grouped_df['romantic'].mean())

plt.ylabel("Relationship Quotient")

plt.xlabel("Grade")
plt.figure(figsize=(10,6))

plt.title("Free Time Vs Grade")

sns.barplot(x=["A","B","C","D","F"], y=grouped_df['freetime'].mean())

plt.ylabel("Free Time")

plt.xlabel("Grade")
plt.figure(figsize=(10,6))

plt.title("Goout Vs Grade")

sns.barplot(x=["A","B","C","D","F"], y=grouped_df['goout'].mean())

plt.ylabel("GO out")

plt.xlabel("Grade")