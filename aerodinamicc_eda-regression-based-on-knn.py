import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
#Data loading
csvFile = '../input/StudentsPerformance.csv'
st = pd.read_csv(csvFile)
print(st.info())
st = st.rename(columns = {'parental level of education' : "parentLE", 'race/ethnicity' : 'race', 'test preparation course': 'prepCourse', 'math score' : 'maths', 'reading score': 'reading', 'writing score' : 'writing'})
st['total'] = st.maths + st.reading + st.writing
st['average'] = round((st.maths + st.reading + st.writing) / 3, 1)
#st['race'].replace(regex = True, inplace = True, to_replace = r'\group ', value = r'')
st.parentLE = st.parentLE.replace("bachelor's degree", "BSc") 
st.parentLE = st.parentLE.replace("master's degree", 'MSc')
st.parentLE = st.parentLE.replace("associate's degree", 'Associate')
st.prepCourse = st.prepCourse.replace('none', 'False')
st.prepCourse = st.prepCourse.replace('completed', 'True')

#st['race'] = st['race'].astype('category')
st['prepCourse'] = st['prepCourse'].map({'False':False, 'True':True})
#st['lunch'] = st['lunch'].astype('category')
#st['gender'] = st['gender'].astype('category')
#st['parentLE'] = st['parentLE'].astype('category')

print(st.info())
fig, axs = plt.subplots(3, 1, sharex=True, figsize = (8,7))
sns.swarmplot(x="gender", y="maths",hue="prepCourse", data=st, size = 3, ax = axs[0])
sns.swarmplot(x="gender", y="reading",hue="prepCourse", data=st, size = 3, ax = axs[1])
sns.swarmplot(x="gender", y="writing",hue="prepCourse", data=st, size = 3, ax = axs[2])
plt.show()
#gender performance across subjects
scores = st.drop(['total', 'average', 'parentLE', 'race', 'lunch'], 1)
scores = pd.melt(scores, ('gender','prepCourse'), ('maths', 'reading', 'writing'))
sns.boxplot(x = "variable", y = 'value', hue = 'gender', data = scores)
plt.xlabel('Subject')
plt.ylabel('Score')
plt.show()
#reading vs writing scores
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot(st.reading, st.writing, kind="reg", stat_func=r2)
plt.xlabel('Reading scores')
plt.ylabel('Writing scores')
plt.show()
#Math scores higher than average
plt.clf()
maths_higher_than_average = st.gender[st.maths > st.average]
sns.countplot(maths_higher_than_average)
plt.ylabel('Number of students with maths scores higher than their average')
plt.show()
#scores related to race/ethnicity
sns.boxplot(x='gender', y='total', hue = 'race', data = st)
plt.xlabel('Total score')
plt.show()
#scores related to parents' education
sns.boxplot(x='gender', y='total', hue = 'parentLE', data = st)
plt.xlabel('Total score')
plt.show()
#This last one represents the number of students with parents divided into categories of race/etnicity and level of education
raceEdu = st[['race', 'parentLE']].groupby(['race', 'parentLE']).size().reset_index()
raceEdu.rename(columns = {0 : 'count'}, inplace = True)
raceEdu = raceEdu.pivot_table(values='count', index='race', columns='parentLE')
raceEdu.plot(kind='bar', stacked=True, figsize=(18.5, 10.5))
print(st.gender.unique())
print(st.race.unique())
print(st.parentLE.unique())
print(st.lunch.unique())
print(st.prepCourse.unique())
students = st
students['gender'] = students['gender'].apply(lambda g: 1 if g == "male" else 0)
students['lunch'] = students['lunch'].apply(lambda l: 1 if l == "standard" else 0)
students['race'] = students['race'].apply(lambda r: 0 if r == "group A" else 
                                                     1 if r == "group B" else
                                                     2 if r == "group C" else
                                                     3 if r == "group D" else 4)
students['parentLE'] = students['parentLE'].apply(lambda r: 0 if r == "some high school" else 
                                                     1 if r == "high school" else
                                                     2 if r == "some college" else
                                                     3 if r == "Associate" else
                                                     4 if r == "BSc" else 5)

students['prepCourse'] = students['prepCourse'].astype('int64')
students['average'] = students['average'].astype('int64')
print(students.info())
plt.matshow(students.corr())
plt.gca().xaxis.tick_bottom()
plt.xticks(range(len(students.columns)), students.columns)
plt.xticks(rotation = 90)
plt.yticks(range(len(students.columns)), students.columns)
plt.colorbar()
plt.title("Correlation of all variables")
plt.show()
#source - https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
def optimalK(maxK, X_train, X_test, y_train, y_test):
    from math import sqrt
    from sklearn.metrics import mean_squared_error

    rmse_val = [] #to store rmse values for different k
    for K in range(maxK):
        K = K+1
        model = KNeighborsRegressor(n_neighbors = K)

        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values

    #plotting the rmse values against k values
    curve = pd.DataFrame(rmse_val) #elbow curve 
    curve.plot()
    
    print("Optimal k value is: %s" % (rmse_val.index(min(rmse_val)) + 1))
    print('RMSE is %s' % (min(rmse_val)))
    
    return(rmse_val.index(min(rmse_val)) + 1)
X_train, X_test, y_train, y_test = train_test_split(students[['race','parentLE', 'gender', 'prepCourse', 'lunch']],
                                                    students.average, test_size=0.30)

optimalK(100, X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(students[['writing']],
                                                    students.average, test_size=0.30)

optK_maths = optimalK(100, X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(students[['race','parentLE', 'gender', 'prepCourse', 'lunch', 'writing']],
                                                    students.average, test_size=0.30)

optK_maths = optimalK(100, X_train, X_test, y_train, y_test)
#so RFE DOES NOT WORK ON KNN regression
#knn = KNeighborsRegressor(n_neighbors=optK_maths)
#knn.fit(X_train,y_train)

#from sklearn.feature_selection import RFE
#
#rfe = RFE(knn, 3)
#rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)

#pred_knn = knn.predict(X_test)
#print('Actual:    %s' % y_test[1:7].values)
#print('Predicted: %s' % pred_knn[1:7])
X_train, X_test, y_train, y_test = train_test_split(students[['gender', 'prepCourse', 'writing']],
                                                    students.reading, test_size=0.33)

optK_reading = optimalK(100, X_train, X_test, y_train, y_test)
knn = KNeighborsRegressor(n_neighbors=optK_reading)
knn.fit(X_train,y_train)

pred_knn = knn.predict(X_test)
print('Actual:    %s' % y_test[1:7].values)
print('Predicted: %s' % pred_knn[1:7])
print("Score is: %s" % (knn.score(X_test, y_test)))
print("Correlation bn predicted and actual values: %s" % (round(np.corrcoef(pred_knn, y_test)[0, 1], 3)))