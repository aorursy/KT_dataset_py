import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/inputt/students_performance.csv')

df.head(5)
print(df.info())

print(df.describe())

print(df.shape)
import seaborn as sns 

import matplotlib.pyplot as plt 

sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")

plt.show()
count  = 0

for i in df['gender'].unique():

    count = count + 1

    print(count,'- ',i)

print('Number of Gender : ', df['gender'].nunique())
count = 0

for i in sorted(df['race/ethnicity'].unique()):

    count = count + 1

    print(count, '-',i)

print('Number of different races/ethnicity of people: ', df['race/ethnicity'].nunique())
count = 0

for i in sorted(df['parental level of education'].unique()):

    count = count + 1

    print(count, '-',i)

print('Number of Parental Level of Education : ', df['parental level of education'].nunique())
count = 0

for i in sorted(df['lunch'].unique()):

    count = count + 1

    print(count, '-',i)

print('Number of lunch: ', df['lunch'].nunique())
count = 0

for i in sorted(df['test preparation course'].unique()):

    count = count + 1

    print(count, '-',i)

print('Number of test preparation course: ', df['test preparation course'].nunique())
#histogram of writing score 

df["writing score"].plot.hist()
plt.figure(figsize=(12,6))

sns.distplot(df['writing score'], kde = False, color='red', bins = 30)

plt.ylabel('Frequency')

plt.title('Writing Score Analysis')

plt.show()
#histogram of reading score 

df["reading score"].plot.hist()
plt.figure(figsize=(12,6))

sns.distplot(df['reading score'], kde = False, color='red', bins = 30)

plt.ylabel('Frequency')

plt.title('Reading Score Analysis')

plt.show()
#histogram of math score 

df["math score"].plot.hist()
plt.figure(figsize=(12,6))

sns.distplot(df['math score'], kde = False, color='red', bins = 30)

plt.ylabel('Frequency')

plt.title('Math Score Analysis')

plt.show()
print('Maximum score in Writing is: ',max(df['writing score']))

print('Minimum score in Writing is: ',min(df['writing score']))

print('Average score in Writing is: ',df['writing score'].mean())
print('Maximum score in Reading is: ',max(df['reading score']))

print('Minimum score in Reading is: ',min(df['reading score']))

print('Average score in Reading is: ',df['reading score'].mean())
print('Maximum score in math is: ',max(df['math score']))

print('Minimum score in math is: ',min(df['math score']))

print('Average score in math is: ',df['math score'].mean())
print('Number of students having AA in Writing: ', len(df[df['writing score'] >=90]))

print('Number of students having AA in Reading: ', len(df[df['reading score'] >=90]))

print('Number of students having AA in Math: ', len(df[df['math score'] >=90]))
perfect_writing = df['writing score'] >= 90

perfect_reading = df['reading score'] >= 90

perfect_math = df['math score'] >= 90



perfect_score = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

print('Number of students having maximum marks in all three subjects: ',len(perfect_score))

male_perfectScore=perfect_score[perfect_score['gender']=='male']

print('Male : ', len(male_perfectScore))

female_perfectScore=perfect_score[perfect_score['gender']=='female']

print('Female : ', len(female_perfectScore))

perfect_score
minimum_math = df['math score'] <= 40

minimum_reading = df['reading score'] <= 40

minimum_writing = df['writing score'] <= 40



minimum_score = df[(minimum_math) & (minimum_reading) & (minimum_writing)]

print('Number of students having minimum marks in all three subjects: ',len(minimum_score))

male_minimumScore=minimum_score[minimum_score['gender']=='male']

print('Male : ', len(male_minimumScore))

female_minimumcore=minimum_score[minimum_score['gender']=='female']

print('Female : ', len(female_minimumcore))

minimum_score
plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.barplot(x = 'gender', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'gender', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'gender', y = 'math score', data = df)



plt.tight_layout()
plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.barplot(x = 'race/ethnicity', y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'race/ethnicity', y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'race/ethnicity', y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.tight_layout()

plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.barplot(x = 'test preparation course', y = 'reading score', hue = 'gender', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'test preparation course', y = 'writing score',hue = 'gender', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'test preparation course', y = 'math score',hue = 'gender', data = df)



plt.tight_layout()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.barplot(x = 'parental level of education', y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'parental level of education', y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'parental level of education', y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.tight_layout()
plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.barplot(x = 'lunch', y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'lunch', y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'lunch', y = 'math score', data = df)

plt.xticks(rotation = 90)







plt.tight_layout()
print('----Females----')

print('Max. Reading Score: ', df[df['gender'] == 'female']['reading score'].max())

print('Min. Reading Score: ', df[df['gender'] == 'female']['reading score'].min())

print('Average Reading Score: ', df[df['gender'] == 'female']['reading score'].mean())

print('----Males----')

print('Max. Reading Score: ', df[df['gender'] == 'male']['reading score'].max())

print('Min. Reading Score: ', df[df['gender'] == 'male']['reading score'].min())

print('Average Reading Score: ', df[df['gender'] == 'male']['reading score'].mean())
print('----Females----')

print('Max. Writing Score: ', df[df['gender'] == 'female']['writing score'].max())

print('Min. Writing Score: ', df[df['gender'] == 'female']['writing score'].min())

print('Average Writing Score: ', df[df['gender'] == 'female']['writing score'].mean())

print('----Males----')

print('Max. Writing Score: ', df[df['gender'] == 'male']['writing score'].max())

print('Min. Writing Score: ', df[df['gender'] == 'male']['writing score'].min())

print('Average Writing Score: ', df[df['gender'] == 'male']['writing score'].mean())
print('----Females----')

print('Max. math Score: ', df[df['gender'] == 'female']['math score'].max())

print('Min. math Score: ', df[df['gender'] == 'female']['math score'].min())

print('Average math Score: ', df[df['gender'] == 'female']['math score'].mean())

print('----Males----')

print('Max. math Score: ', df[df['gender'] == 'male']['math score'].max())

print('Min. math Score: ', df[df['gender'] == 'male']['math score'].min())

print('Average math Score: ', df[df['gender'] == 'male']['math score'].mean())
plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'gender', y = 'math score', data = df,palette = ['blue', 'red'])



plt.subplot(1,3,2)

sns.boxplot(x = 'gender', y = 'reading score', data = df,palette = ['blue', 'red'])



plt.subplot(1,3,3)

sns.boxplot(x = 'gender', y = 'writing score', data = df,palette = ['blue', 'red'])



plt.tight_layout()
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Reading Max. marks: ', df[df['race/ethnicity'] == i]['reading score'].max())

    print('Reading Min. marks: ', df[df['race/ethnicity'] == i]['reading score'].min())

    print('Reading Average marks: ', df[df['race/ethnicity'] == i]['reading score'].mean())
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Writing Max. marks: ', df[df['race/ethnicity'] == i]['writing score'].max())

    print('Writing Min. marks: ', df[df['race/ethnicity'] == i]['writing score'].min())

    print('Writing Average marks: ', df[df['race/ethnicity'] == i]['writing score'].mean())
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Math. Max. marks: ', df[df['race/ethnicity'] == i]['math score'].max())

    print('Math. Min. marks: ', df[df['race/ethnicity'] == i]['math score'].min())

    print('Math. Average marks: ', df[df['race/ethnicity'] == i]['math score'].mean())
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)

sns.boxplot(x = 'race/ethnicity', y = 'math score', data = df)



plt.subplot(1,3,2)

sns.boxplot(x = 'race/ethnicity', y = 'reading score', data = df)



plt.subplot(1,3,3)

sns.boxplot(x = 'race/ethnicity', y = 'writing score', data = df)



plt.tight_layout()
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Reading Max. marks: ', df[df['parental level of education'] == i]['reading score'].max())

    print('Reading Min. marks: ', df[df['parental level of education'] == i]['reading score'].min())

    print('Reading Average. marks: ', df[df['parental level of education'] == i]['reading score'].mean())
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Writing Max. marks: ', df[df['parental level of education'] == i]['writing score'].max())

    print('Writing Min. marks: ', df[df['parental level of education'] == i]['writing score'].min())

    print('Writing Average. marks: ', df[df['parental level of education'] == i]['writing score'].mean())
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)

sns.boxplot(x ='parental level of education' , y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.boxplot(x ='parental level of education' , y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.boxplot(x ='parental level of education' , y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.tight_layout()
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Math Max. marks: ', df[df['parental level of education'] == i]['math score'].max())

    print('Math Min. marks: ', df[df['parental level of education'] == i]['math score'].min())

    print('Math Average. marks: ', df[df['parental level of education'] == i]['math score'].mean())
def get_grade(marks):

    if marks >= 90:

        return 'AA'

    elif marks >= 85 and marks < 90:

        return 'BA'

    elif marks >=80 and marks < 85:

        return 'BB'

    elif marks >=75 and marks < 80:

        return 'CB'

    elif marks >= 70 and marks < 75:

        return 'CC'

    elif marks >=60 and marks < 70:

        return 'DC'

    elif marks >= 50 and marks < 60:

        return 'DD'

    elif marks < 50:

        return 'FF'
df['writing_grade'] = df['writing score'].apply(get_grade)

df['math_grade'] = df['math score'].apply(get_grade)

df['reading_grade'] = df['reading score'].apply(get_grade)
sns.set_style('whitegrid')

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

sns.countplot(x ='math_grade', data = df,order = ['AA','BA','BB','CB','CC','DC','DD','FF'],palette='magma')

plt.title('Grade Count in Math')





plt.subplot(1,3,2)

sns.countplot(x ='reading_grade', data = df,order = ['AA','BA','BB','CB','CC','DC','DD','FF'],palette='magma')

plt.title('Grade Count in Reading')



plt.subplot(1,3,3)

sns.countplot(x ='writing_grade', data = df,order = ['AA','BA','BB','CB','CC','DC','DD','FF'],palette='magma')

plt.title('Grade Count in Writing')



plt.tight_layout()



print('Number of students having maximum grade in reading: ', len(df[df['reading_grade'] == 'AA']))

print('Number of students having maximum grade in writing: ', len(df[df['writing_grade'] == 'AA']))

print('Number of students having maximum grade in math: ', len(df[df['math_grade'] == 'AA']))

print('Number of students having minimum grade in reading: ', len(df[df['reading_grade'] == 'FF']))

print('Number of students having minimum grade in writing: ', len(df[df['writing_grade'] == 'FF']))

print('Number of students having minimum grade in math: ', len(df[df['math_grade'] == 'FF']))
minimum_math = df['math_grade'] == 'FF'

minimum_reading = df['reading_grade'] == 'FF'

minimum_writing = df['writing_grade'] == 'FF'



failed_grade = df[(minimum_math) & (minimum_reading) & (minimum_writing)]

print('Number of students having minimum grade(F) in all three subjects: ',len(failed_grade))
perfect_writing = df['writing_grade'] == 'AA'

perfect_reading = df['reading_grade'] == 'AA'

perfect_math = df['math_grade'] == 'AA'



perfect_grade = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

print('Number of students having maximum grade(AA) in all three subjects: ',len(perfect_grade))

passed_students = len(df) - len(failed_grade)

print('Total Number of students who passed are: {}' .format(passed_students))
labels = ['Passed', 'Failed']

failed_students = len(failed_grade)

sizes = [passed_students, failed_students]

explode = (0, 0.1)  

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
sns.pairplot(df, hue = 'gender')

plt.show()
#correlation

corr = df.corr()

mask = np.triu(np.ones_like(corr,dtype = bool))

sns.heatmap(df.corr(),mask=mask,annot = True, cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
df['total'] = (df['math score']+df['reading score']+df['writing score'])/3

df.sample()
df.groupby(['race/ethnicity','parental level of education']).mean()
df.head(3)
Y = df.iloc[:,5:8]

print(Y.head(1))
Gender=pd.get_dummies(df["gender"],drop_first=True)

print(Gender.head(5))
ParentLevel=pd.get_dummies(df["parental level of education"],drop_first=True)

print(ParentLevel.head(5))
Lunch=pd.get_dummies(df["lunch"],drop_first=True)

print(Lunch.head(5))
TestPreperation=pd.get_dummies(df["test preparation course"],drop_first=True)

print(TestPreperation.head(5))
RaceEthnicity=pd.get_dummies(df['race/ethnicity'],drop_first=True)

print(RaceEthnicity.head(5))
print(df.head(2))
X=pd.concat([Gender,RaceEthnicity,ParentLevel,TestPreperation,Lunch],axis=1)

print(X.head(1))
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import neighbors

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.1)

model = neighbors.KNeighborsRegressor()

model.fit(X_train,Y_train)

predictions=model.predict(X_test)



mean_squared_error(predictions,Y_test)





scores = df.loc[:,["math score","reading score","writing score"]]

scores.rename(index = int, columns = {"math score":"mthscore","reading score":"readscr","writing score":"writingscr"},inplace=True)



from sklearn.cluster import KMeans

wcss = []



for i in range(1,15):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(scores)

    wcss.append(kmeans.inertia_)





plt.plot(range(1,15),wcss,"-o")

plt.xlabel("Number of K value")

plt.ylabel("WCSS Value")

plt.show()





kmeans2 = KMeans(n_clusters = 5)

clusters = kmeans2.fit_predict(scores)

scores["examscores"] = clusters





plt.scatter(scores.mthscore[scores.examscores == 0],scores.readscr[scores.examscores == 0],color="red")

plt.scatter(scores.mthscore[scores.examscores == 1],scores.readscr[scores.examscores == 1],color="green")

plt.scatter(scores.mthscore[scores.examscores == 2],scores.readscr[scores.examscores == 2],color="blue")

plt.scatter(scores.mthscore[scores.examscores == 3],scores.readscr[scores.examscores == 3],color="black")

plt.scatter(scores.mthscore[scores.examscores == 4],scores.readscr[scores.examscores == 4],color="brown")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(scores, method = "ward")

dendrogram(merg,leaf_rotation=90)

plt.xlabel("Scores")

plt.ylabel("Distances")

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")

cluster = hc.fit_predict(scores)



scores["examscores"] = cluster

plt.scatter(scores.mthscore[scores.examscores == 0],scores.readscr[scores.examscores == 0],color="red")

plt.scatter(scores.mthscore[scores.examscores == 1],scores.readscr[scores.examscores == 1],color="green")

plt.scatter(scores.mthscore[scores.examscores == 2],scores.readscr[scores.examscores == 2],color="blue")

plt.scatter(scores.mthscore[scores.examscores == 3],scores.readscr[scores.examscores == 3],color="black")

plt.scatter(scores.mthscore[scores.examscores == 4],scores.readscr[scores.examscores == 4],color="brown")

#plt.scatter(scores.mthscore[scores.examscores == 5],scores.readscr[scores.examscores == 5],color="purple")

#plt.scatter(scores.mthscore[scores.examscores == 6],scores.readscr[scores.examscores == 6],color="orange")



plt.show()