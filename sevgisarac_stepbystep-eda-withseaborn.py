# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore') 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# load data

df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

#check columns' names

df.columns
#column rename 

df.rename(columns = {'math score':'math_score','reading score':'reading_score', 'writing score':'writing_score'},inplace =True)

df.rename(columns = {'race/ethnicity':'race_ethnicity', 'parental level of education':'parental_level_of_education','test preparation course':'test_preparation_course'},inplace = True)
#check first 5 entries

df.head()
#check basic statistics values

df.describe()
df.info()
category1 = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

for i in category1:

    

    plt.figure(figsize = (9,15))

    sns.set_style("whitegrid") #for grid background

    sns.catplot(x=i, kind="count", palette="ch:.25", dodge=True, data=df)

    plt.xticks(rotation= 90)

    plt.ylabel("Frequency")

    plt.title(i)

    plt.show()

   

 
#new mean of scores column

df["meanofscores"] = (df["math_score"]+df["reading_score"]+df["writing_score"])/3

df.head()
numericVar = ["math_score", "reading_score", "writing_score","meanofscores"]

for n in numericVar:

    sns.set(style="whitegrid") #for grid background

    ax = sns.distplot(df[n], color="y")

    plt.show()

    
#gender vs meanofscore

df[["gender","meanofscores"]].groupby(["gender"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
df[["race_ethnicity","meanofscores"]].groupby(["race_ethnicity"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
df[["lunch","meanofscores"]].groupby(["lunch"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
df[["parental_level_of_education","meanofscores"]].groupby(["parental_level_of_education"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
df[["test_preparation_course","meanofscores"]].groupby(["test_preparation_course"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
def detect_outliers(df, features):

    outlier_indices = []

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3-Q1

        #Outlier step

        outlier_step = IQR*1.5

        #detect outlier and their indeces

        outlier_list_col = df[(df[c]<Q1-outlier_step)|(df[c]>Q3+outlier_step)].index

        #store indices

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>1)

    return multiple_outliers
df.loc[detect_outliers(df,["math_score","reading_score","writing_score","meanofscores"])]
#Before decide to drop all outliers, let look at boxplots

plt.figure(figsize = (9,12))

sns.boxplot(x="gender", y="meanofscores", hue="lunch", data=df, palette="PRGn")

ax.legend(loc='lower right',frameon = True) 

sns.set(style="whitegrid")



sns.despine(left=True, bottom=True)

plt.show()
#even if we have just little outliers, lets drop all to catch normalization easily

# drop outliers

df = df.drop(detect_outliers(df,["math_score","reading_score","writing_score","meanofscores"]),axis = 0).reset_index(drop = True)
#missing value check codes; we will check meaningless value like "-" or losts of zero

#but our data set has none of them

#df["math score"].value_counts()
#df["reading score"].value_counts()
#df["writing score"].value_counts()
#df["meanofscores"].value_counts()
#to check drop

df.info()
#race success ratio

race_list = list(df['race_ethnicity'].unique()) # for finding each rece type

race_success_ratio = [] #to sort these values

for i in race_list:

    x = df[df['race_ethnicity'] == i]

    race_success_rate = sum(x.meanofscores)/len(x)

    race_success_ratio.append(race_success_rate)

#for the ratio, we will create new data set

data = pd.DataFrame({'race_list':race_list, 'race_success_ratio':race_success_ratio})

#sort our data

new_index = (data['race_success_ratio'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)



#visualization

plt.figure(figsize=(10,5))

sns.barplot(x = sorted_data['race_list'],y = sorted_data['race_success_ratio'],palette="ch:.25")

plt.xticks(rotation=0)

plt.xlabel('race/ethnicity')

plt.ylabel('success ratio')

plt.title('Success rate given race/ethnicity')

sns.set(style="whitegrid")

ax.set(xlim=(0, 24))

sns.despine(left=True, bottom=True)
df[["race_ethnicity","meanofscores"]].groupby(["race_ethnicity"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
#make meanofscore integer

df['meanofscores'] = df['meanofscores'].astype('int')

# calculate score interval and add new column

df['grading_scale'] = ['A ' if 100>=i>=93 else 'A-' if 92>=i>=90 else 'B+' if 89>=i>=87 else 'B ' if 86>=i>=83 else 'B-' if 82>=i>=80 else 'C+' if 79>=i>=77 else 'C ' if 76>=i>=73 else 'C-' if 72>=i>=70 else 'D+' if 69>=i>=67 else 'D ' if 66>=i>=63 else 'D-' if 62>=i>=60 else 'F ' for i in df.meanofscores]
df['grading_scale'].value_counts()
grade_list = list(df['grading_scale'].unique()) # for finding each grade type

grade_ratio = [] #to sort these values

for i in grade_list:

    x = df[df['grading_scale'] == i]

    grade_rate = sum(x.meanofscores)

    grade_ratio.append(grade_rate)

#for the ratio, we will create new data set

data = pd.DataFrame({'grade_list':grade_list, 'grade_ratio':grade_ratio})

#sort our data

new_index = (data['grade_ratio'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)



#visualization

plt.figure(figsize=(10,5))



sns.barplot(x = sorted_data['grade_ratio'] ,y = sorted_data['grade_list'],palette="pastel")

plt.xticks(rotation=0)

plt.xlabel('Sum of Grade')

plt.ylabel('Grade Scale')

plt.title('Sum of Grade  given grade scale')

sns.set(style="whitegrid")

ax.set(xlim=(0, 24))

sns.despine(left=True, bottom=True)

df[["parental_level_of_education","meanofscores"]].groupby(["parental_level_of_education"], as_index = False).mean().sort_values(by="meanofscores",ascending = False)
math_score_count =Counter(df.math_score)

most_common_scores = math_score_count.most_common(15)

print(most_common_scores)
math_score_count =Counter(df.math_score)

most_common_scores = math_score_count.most_common(15)



x,y = zip(*most_common_scores)

x,y = list(x),list(y)



# visualization

plt.figure(figsize=(10,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Most Common Scores')

plt.ylabel('Frequency')

plt.title('Most common 15 Score')

sns.set(style="whitegrid")



sns.despine(left=True, bottom=True)
plt.subplots(figsize =(15,10))

ax = sns.pointplot(x="test_preparation_course", y="meanofscores", hue="lunch",

                   data=df,

                   markers=["o", "x"],

                   linestyles=["-", "--"])

sns.set(style="whitegrid")



sns.despine(left=True, bottom=True)




g = sns.jointplot( 'math_score','reading_score', data = df, kind="kde", space=0, color="g") #kde: kernel density estimation #size: graph size

sns.set(style="whitegrid")

plt.savefig('graph.png')#it is related to kaggle notebook visual

plt.show()



# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one



g = sns.jointplot("math_score", "reading_score", data=df, kind="reg")

sns.set(style="whitegrid")

plt.show()
#make meanofscore integer

df['meanofscores'] = df['meanofscores'].astype('int')

# calculate score interval and add new column

df['grading_scale2'] = ['A ' if 100>=i>=90 else 'B' if 89>=i>=80 else 'C' if 79>=i>=70  else 'D' if 69>=i>=60 else 'F ' for i in df.meanofscores]
df.grading_scale2.value_counts()
labels = df.grading_scale2.value_counts().index



explode = [0.1,0,0,0,0]



sizes= df.grading_scale2.value_counts().values



#visualization



#add colors

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ffcc00']



plt.figure(figsize = (8,8))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.title('Percentage of Grade Scale',color = 'grey',fontsize = 15)





plt.show()
sns.lmplot(x='math_score', y='writing_score',data=df)

sns.set(style="whitegrid")



plt.show()
sns.kdeplot(df.math_score, df.reading_score, shade=True,cut=3)

sns.set(style="whitegrid")

plt.show()


sns.violinplot(data=df, palette="cool", inner="points")

sns.set(style="whitegrid")

plt.show()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="white", fmt= '.1f',ax=ax)

plt.show()
plt.figure(figsize = (8,8))

plt.xticks(rotation=90)

sns.set(style="whitegrid")

sns.swarmplot(x="parental_level_of_education", y="meanofscores",hue="grading_scale2", data=df)

ax.legend(loc='lower right',frameon = True) 

plt.show()
plt.figure(figsize = (8,8))

plt.xticks(rotation=90)

sns.set(style="whitegrid")

sns.swarmplot(x="test_preparation_course", y="meanofscores",hue="grading_scale2", data=df)

ax.legend(loc='lower right',frameon = True) 

plt.show()
plt.figure(figsize = (8,8))

plt.xticks(rotation=90)

sns.set(style="whitegrid")

sns.swarmplot(x="lunch", y="meanofscores",hue="grading_scale2", data=df)

ax.legend(loc='lower right',frameon = True) 

plt.show()
sns.pairplot(df)

plt.show()
#Count Plot

plt.figure(figsize=(10,7))

sns.countplot(df.grading_scale)

plt.title('Number of Each Grade',color = 'grey',fontsize=15)

plt.show()
#Bar Plot

grade = df.grading_scale.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=grade.index,y=grade.values)

plt.title('Number of Each Grade',color = 'grey',fontsize=15)

plt.show()