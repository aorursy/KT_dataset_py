import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd

from pylab import rcParams
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head(10)
df.describe()
df.info()
# to check presence of missing observations

df.isna().sum()
# to print unique values in all columns

unique_columns = [col+" "+df[col].unique() for col in df.select_dtypes(exclude=np.number)]

unique_columns
#Creating a column Gender1 - where it assumes value 1 when gender = female



gender1 = [1 if each == "female" else 0 for each in df.gender]



df['Gender1']=gender1



df.head(10)
# correlation b/w variables



corr = df.corr()

print(corr)



sns.heatmap(corr, annot=True)

plt.show()
# Plotting densities for female vs male basis score in various subjects

f=df[['math score','reading score','writing score','Gender1']]

def plot_densities(data):

    '''

    Plot features densities depending on the outcome values

    '''

    # change fig size to fit all subplots

    rcParams['figure.figsize'] = 20, 7

    fig, axs = plt.subplots(3, 1)

    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95,

                        wspace = 0.2, hspace = 0.9)



    # plot densities

    for column_name in names[:-1]: 

        ax = axs[names.index(column_name)]

        data[data['Gender1'] == 0][column_name].plot(kind='density', ax=ax, subplots=True, 

                                    sharex=False, color="red", legend=True,

                                    label=column_name + ' for Male')

        data[data['Gender1'] == 1][column_name].plot(kind='density', ax=ax, subplots=True, 

                                     sharex=False, color="green", legend=True,

                                     label=column_name + ' for Female')

        ax.set_xlabel(column_name + ' values')

        ax.set_title(column_name + ' density')

        ax.grid('on')

    plt.show()



names = list(f.columns)



# plot correlation & densities

plot_densities(f)
# distribution of test preparation

df['average score']=df[['math score','reading score','writing score']].mean(axis=1)

print(df.std())







sns.catplot(x="average score", y="test preparation course", hue="gender",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=df)



sns.catplot(x="average score", y="parental level of education", hue="gender",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=df)

sns.catplot(x="average score", y="lunch", hue="gender",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=df)

sns.catplot(x="average score", y="race/ethnicity", hue="gender",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=df)


sns.pairplot(f,hue = 'Gender1')
#Creating Dummy Values

dummy = pd.get_dummies(df[['race/ethnicity','lunch','test preparation course']])

dummy.head(10)
# concatenate dummy df with original df

df1 = pd.concat([df, dummy], axis = 1)

df1.info()
# average of values by gender

df_group = df1.groupby('gender').mean()

df_group
# drop columns for which dummies are created and y variables

x = df1.drop(['race/ethnicity', 'lunch', 'test preparation course', 'parental level of education','gender','average score','Gender1'], axis = 1)



#standardize data

x= (x-np.min(x)) / (np.max(x)-np.min(x))



x.head()
y = df['Gender1']

y.head()
# split into train / test data

from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 21)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state= 0,max_depth=10 )

classifier.fit(x_train, y_train)

y_pre = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pre)

sum1 = np.sum(cm, axis=1, keepdims=True)

perc1 = cm / sum1.astype(float) * 100

annot = np.empty_like(cm).astype(str)



nrows, ncols = cm.shape

for i in range(nrows):

            for j in range(ncols):

                c = cm[i, j]

                p = perc1[i, j]

                if i == j:

                    s = sum1[i]

                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

                elif c == 0:

                    annot[i, j] = ''

                else:

                    annot[i, j] = '%.1f%%\n%d' % (p, c)







sns.heatmap(cm, annot=annot, fmt='')

print(classification_report(y_test,y_pre))
from sklearn import tree

text_representation = tree.export_text(classifier)

print(text_representation)
with open("decistion_tree.log", "w") as fout:

    fout.write(text_representation)

fig = plt.figure(figsize=(40,20))

_ = tree.plot_tree(classifier, filled=True)
from sklearn.neighbors import KNeighborsClassifier

error = []

for i in range(2,20):

 

 knn = KNeighborsClassifier(n_neighbors=i)

 knn.fit(x_train,y_train)

 pred = knn.predict(x_test)

 error.append(np.mean(pred != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(2,20),error,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title('Error vs. K Value')

plt.xlabel('K')

plt.ylabel('Error')



#optimal neighbour count=7

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_pre = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pre)

sum1 = np.sum(cm, axis=1, keepdims=True)

perc1 = cm / sum1.astype(float) * 100

annot = np.empty_like(cm).astype(str)



nrows, ncols = cm.shape

for i in range(nrows):

            for j in range(ncols):

                c = cm[i, j]

                p = perc1[i, j]

                if i == j:

                    s = sum1[i]

                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

                elif c == 0:

                    annot[i, j] = ''

                else:

                    annot[i, j] = '%.1f%%\n%d' % (p, c)







sns.heatmap(cm, annot=annot, fmt='')

print(classification_report(y_test,y_pre))
error=[]

from sklearn.ensemble import RandomForestClassifier



for i in range(1,100):

 rfc = RandomForestClassifier(n_estimators = i,random_state = 21,bootstrap = "False",criterion="gini",min_samples_split = 10 , min_samples_leaf = 2)

 rfc.fit(x_train,y_train)

 pred = rfc.predict(x_test)

 error.append(np.mean(pred != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,100),error,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title('Error vs. No of Trees')

plt.xlabel('No of Trees')

plt.ylabel('Error')



rfc = RandomForestClassifier(n_estimators = 50,random_state = 21,bootstrap = "False",criterion="gini",min_samples_split = 10 , min_samples_leaf = 2)

rfc.fit(x_train,y_train)

y_pre = rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pre)

sum1 = np.sum(cm, axis=1, keepdims=True)

perc1 = cm / sum1.astype(float) * 100

annot = np.empty_like(cm).astype(str)



nrows, ncols = cm.shape

for i in range(nrows):

            for j in range(ncols):

                c = cm[i, j]

                p = perc1[i, j]

                if i == j:

                    s = sum1[i]

                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

                elif c == 0:

                    annot[i, j] = ''

                else:

                    annot[i, j] = '%.1f%%\n%d' % (p, c)







sns.heatmap(cm, annot=annot, fmt='')
print(classification_report(y_test,y_pre))