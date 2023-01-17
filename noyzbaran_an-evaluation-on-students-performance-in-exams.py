import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



stud_perf = pd.read_csv('../input/studentsperformance/datasets_74977_169835_StudentsPerformance.csv')
#Obtain total number of students for each 'gender' (Entire DataFrame)

p_race = stud_perf['race/ethnicity'].value_counts()

p_race_height = p_race.values.tolist() #Provides numerical values

p_race.axes #Provides row labels

p_race_labels = p_race.axes[0].tolist() #Converts index object to list



#=====PLOT Preparations and Plotting====#

ind = np.arange(5)  # the x locations for the groups

width = 0.7        # the width of the bars

colors = ['#FD1414','#FFF012','#11F237','#1155F2','#B611F2']

fig, ax = plt.subplots(figsize=(5,7))

stud_perf_bars = ax.bar(ind, p_race_height , width, color=colors)



#Add some text for labels, title and axes ticks

ax.set_xlabel("Ethnity Group",fontsize=20)

ax.set_ylabel('Count',fontsize=20)

ax.set_title('Race/Ethnicity',fontsize=22)

ax.set_xticks(ind) #Positioning on the x axis

ax.set_xticklabels(('group C', 'group D','group B','group E','group A'),

                  fontsize = 12)



#Auto-labels the number of mushrooms for each bar color.

def autolabel(rects,fontsize=14):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),

                ha='center', va='bottom',fontsize=fontsize)

autolabel(stud_perf_bars)        

plt.show() #Display bars. 
female_count = [] #female

male_count = []    #male

for genCount in p_race_labels:

    size = len(stud_perf[stud_perf['race/ethnicity'] == genCount].index)

    f_c = len(stud_perf[(stud_perf['race/ethnicity'] == genCount) & (stud_perf['gender'] == 'female')].index)

    female_count.append(f_c)

    male_count.append(size-f_c)

                        

#=====PLOT Preparations and Plotting====#

width = 0.40

fig, ax = plt.subplots(figsize=(12,7))

female_bar_value = ax.bar(ind, female_count , width, color='#FF3A75')

male_bar_value = ax.bar(ind+width, male_count , width, color='#0A0AFF')



#Add some text for labels, title and axes ticks

ax.set_xlabel("Ethnicity Group",fontsize=20)

ax.set_ylabel('Count',fontsize=20)

ax.set_title('Gender Comparison By Group',fontsize=22)

ax.set_xticks(ind + width / 2) #Positioning on the x axis

ax.set_xticklabels(('group C', 'group D','group B','group E','group A'),

                  fontsize = 12)

ax.legend((female_bar_value,male_bar_value),('Female Count','Male Count'),fontsize=17)

autolabel(female_bar_value, 10)

autolabel(male_bar_value, 10)

plt.show()
plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

sns.barplot(x='test preparation course',y='math score',data=stud_perf,hue='gender',palette='summer')

plt.title('MATH SCORES')

plt.subplot(1, 3, 2)

sns.barplot(x='test preparation course',y='reading score',data=stud_perf,hue='gender',palette='summer')

plt.title('READING SCORES')

plt.subplot(1, 3, 3)

sns.barplot(x='test preparation course',y='writing score',data=stud_perf,hue='gender',palette='summer')

plt.title('WRITING SCORES')

plt.show()
stud_perf['Total Score']=stud_perf['math score']+stud_perf['reading score']+stud_perf['writing score']
fig,ax=plt.subplots()

sns.barplot(x=stud_perf['parental level of education'],y='Total Score',data=stud_perf,palette='Wistia')

fig.autofmt_xdate()
plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

sns.barplot(x='test preparation course',y='math score',data=stud_perf,hue='gender',palette='summer')

plt.title('MATH SCORES')

plt.subplot(1, 3, 2)

sns.barplot(x='test preparation course',y='reading score',data=stud_perf,hue='gender',palette='summer')

plt.title('READING SCORES')

plt.subplot(1, 3, 3)

sns.barplot(x='test preparation course',y='writing score',data=stud_perf,hue='gender',palette='summer')

plt.title('WRITING SCORES')

plt.show()
stud_perf_sample = stud_perf.loc[np.random.choice(stud_perf.index, 1000, False)]
#Get all unique race/ethnicity

stud_perf_sample['race/ethnicity'].unique()
#mushrooms_sample.groupby('cap-color', 0).nunique()



#Get 'race/ethnicity' Series

genCount = stud_perf_sample['race/ethnicity']



#Get the total number of mushrooms for each unique cap color. 

genCount.value_counts()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



stud_perf = pd.read_csv('../input/studentsperformance/datasets_74977_169835_StudentsPerformance.csv')



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

 #       print(os.path.join(dirname, filename))

        



#df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv') # df usually is used to abbreviate "Data Frame" from pandas library



#print(f'Data Frame Shape (rows, columns): {df.shape}') 



sns.countplot(data=stud_perf, x="gender").set_title("Class Outcome - Female-F/Male-M")
import numpy as np 

import pandas as pd

import warnings

warnings.simplefilter("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
stud_perf.isnull().sum()
stud_perf.describe()
X=stud_perf.drop('gender',axis=1) #Predictors

y=stud_perf['gender'] #Response

X.head()
from sklearn.preprocessing import LabelEncoder

Encoder_X = LabelEncoder() 

for col in X.columns:

    X[col] = Encoder_X.fit_transform(X[col])

Encoder_y=LabelEncoder()

y = Encoder_y.fit_transform(y)
X.head()
y
X=pd.get_dummies(X,columns=X.columns,drop_first=True)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
def visualization_train(model):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_train, y_train

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title("%s Training Set" %(model))

    plt.xlabel('PC 1')

    plt.ylabel('PC 2')

    plt.legend()

def visualization_test(model):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_test, y_test

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('red', 'green')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title("%s Test Set" %(model))

    plt.xlabel('PC 1')

    plt.ylabel('PC 2')

    plt.legend()
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(8, kernel_initializer='uniform', activation= 'relu', input_dim = 2))

classifier.add(Dense(6, kernel_initializer='uniform', activation= 'relu'))

classifier.add(Dense(5, kernel_initializer='uniform', activation= 'relu'))

classifier.add(Dense(4, kernel_initializer='uniform', activation= 'relu'))

classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid'))

classifier.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
visualization_train(model='ANN')
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):

    if train == True:

        print("Training results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))

        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))

        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')

        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))

        print('Standard Deviation:\t{0:.4f}'.format(res.std()))

    elif train == False:

        print("Test results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))

        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('Logistic Reg')
visualization_test('Logistic Reg')
from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=42)



classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('SVC')
visualization_test('SVC')
from sklearn.neighbors import KNeighborsClassifier as KNN



classifier = KNN()

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('K-NN')
visualization_test('K-NN')
from sklearn.naive_bayes import GaussianNB as NB



classifier = NB()

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('Naive Bayes')
visualization_test('Naive Bayes')
from sklearn.tree import DecisionTreeClassifier as DT



classifier = DT(criterion='entropy',random_state=42)

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('Decision Tree')
visualization_test('Decision Tree')