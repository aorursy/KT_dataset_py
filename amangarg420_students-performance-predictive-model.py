# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.gender.value_counts()
#df["gender"]=df["gender"].astype('category').cat.codes
df.head(4)
df=df.rename(columns={
                   'gender':'gender',
                   'race/ethnicity':'group',
                   'parental level of education':'highest_degree',
                   'lunch':'lunch',
                   'test preparation course':'coaching',
                   'math score':'math_score',
                   'reading score':'reading_score',
                   'writing score':'writing_score'
})
df.head()
df.group.value_counts()
df.head()
df.highest_degree.value_counts()

#df["highest_degree"]=df["highest_degree"].astype('category').cat.codes
df.highest_degree.value_counts()
df.lunch.value_counts()
df.lunch.value_counts()
df.coaching.value_counts()
df.coaching.value_counts()
df.isnull().any()
corr=df.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
corr
#here we clearly see strong positive linear relationship between math_score and reading_score
plt.scatter(df.math_score,df.reading_score)
plt.title("Relationship between math_score and reading")
plt.xlabel("math_scores")
plt.ylabel("reading_scores")
#writing and reading scores are also corelated to each other
plt.scatter(df.writing_score,df.reading_score)
plt.title("Relationship between writing_score and reading")
plt.xlabel("writing_scores")
plt.ylabel("reading_scores")
#writing and math scores are also positively related to each other
plt.scatter(df.writing_score,df.math_score)
plt.title("Relationship between writing_score and math")
plt.xlabel("writing_scores")
plt.ylabel("math_scores")
g=sns.PairGrid(df)
g.map(plt.scatter)
f,axes=plt.subplots(ncols=3,figsize=(15,6))
#graph math_score distribution
sns.distplot(df.math_score,kde=True,color="g",ax=axes[0]).set_title("math_score distribution of students")
axes[0].set_label('student_count')

#graph reading_score distribution
sns.distplot(df.reading_score,kde=True,color="r",ax=axes[1]).set_title("reading_score distribution of students")
axes[1].set_label('student_count')

#graph writing_score distribution
sns.distplot(df.writing_score,kde=True,color="b",ax=axes[2]).set_title("writing_score distribution of students")
axes[2].set_label('student_count')
#here we clearly see the distributuion of students in the groups
f,ax=plt.subplots(figsize=(15,4))
sns.countplot(y='group',hue='gender',data=df).set_title("students distribution in the gropus on the basis of gender")
color_types=['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']
#countplot a.k.a bar_plot
sns.countplot(x='group',data=df,palette=color_types).set_title('students group distribution');


cross_table=pd.crosstab(index=df['highest_degree'],columns=['gender'])
cross_table.plot(kind="bar",
                figsize=(7,7),
                stacked=False)
df.gender.value_counts()
df['total_score']=(df['math_score']+df['writing_score']+df['reading_score'])/3
df['total_score']=df['total_score'].astype('int')
df.head()
df["group"]=df["group"].astype('category').cat.codes
df["coaching"]=df["coaching"].astype('category').cat.codes
df["lunch"]=df["lunch"].astype('category').cat.codes
df["highest_degree"]=df["highest_degree"].astype('category').cat.codes
df["gender"]=df["gender"].astype('category').cat.codes
df.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
X=df.drop('total_score',axis=1)
y=df.total_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2018)
linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print((mean_absolute_error(y_test,y_pred)))
plt.scatter(y_pred,y_test,color='red',edgecolors=(0,0,0))
plt.title("difference between predicted and actual values")
plt.xlabel("predicted values")
plt.ylabel("actual values")