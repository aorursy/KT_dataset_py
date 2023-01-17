import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
adult_p="../input/adult_data.csv"
adult=pd.read_csv(adult_p, sep=',', decimal='.', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'annual_salary'])
adult
adult.dtypes
adult.describe()
adult['workclass'].value_counts()
adult['workclass']=adult['workclass'].str.strip()
adult['workclass'].replace( '?', "Private", inplace=True)
adult['workclass'].value_counts()
adult['education'].value_counts()
adult['marital-status'].value_counts()
adult['sex'].value_counts()
adult['sex'] = adult['sex'].map({' Male':1,' Female':0}).astype(int)
adult['occupation'].value_counts()
adult['occupation']=adult['occupation'].str.strip()
adult['occupation'].replace('?', 'Other-service', inplace=True)
adult['occupation'].value_counts()
adult['relationship'].value_counts()
adult['race'].value_counts()
adult['native-country'].value_counts()
adult['native-country']=adult['native-country'].str.strip()
adult['native-country'].replace('?', 'United-States', inplace=True)
adult['native-country'].value_counts()
adult['annual_salary'].value_counts()
adult['capital-gain'].value_counts()
adult['age'] .hist(bins=6,color='darkblue', figsize=(5, 5), label='Age')
plt.title("Distribution of age")
plt.ylabel('Number of people')
plt.xlabel('Age in years')
plt.legend()
adult['workclass'].value_counts().sort_index().plot.bar()
adult['education'].value_counts().sort_index().plot.bar()
adult['marital-status'] = adult['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
adult['marital-status'] = adult['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
adult['marital-status'].value_counts().plot(kind='pie',figsize=(6, 6),autopct='%.2f')
plt.ylabel('Percentage of people')
plt.title('Distribution of Marital status')
adult['relationship'].value_counts().plot(kind='pie',figsize=(7, 7),autopct='%.2f')
plt.ylabel('People')
plt.title('Distribution of relationship column')
adult['race'].value_counts().sort_index().plot.bar()
adult['sex'] .hist(bins=10,color='pink', figsize=(5, 5), label=('Sex'))
plt.legend()
plt.title("Distribution of sex")
plt.ylabel('Number of people')
plt.xlabel('Sex - 0:Female and 1:Male')
adult['native-country'] = adult['native-country'].replace(['Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],'Non-US')
adult['native-country']= adult['native-country'].replace(['United-States'], 'US')
adult['native-country'].value_counts().plot(kind='pie',figsize=(5, 5),autopct='%.2f')
plt.ylabel('Number of People')
plt.title('Distribution of people from US and Non-US countries')
adult['annual_salary']=adult['annual_salary'].map({' <=50K':0,' >50K':1}).astype(int)
adult['annual_salary'] .hist(bins=10,color='purple', figsize=(5, 5), label='Annual Income')
plt.legend()
plt.title("Distribution of annual income")
plt.ylabel('Number of people')
plt.xlabel('Per annuam income- 0:>50k and 1:<50k')
adult['hours-per-week'] .hist(bins=8,color='green', figsize=(5, 5), label='hours-per-week')
plt.legend()
plt.title("Distribution of hours per week")
plt.ylabel('Number of people')
plt.xlabel('Hours per week')
adult.boxplot(column='age', by='annual_salary', figsize=(6,6))
adult[['sex','annual_salary']].groupby(['sex']).mean() #Where 0 is for Females and 1 is for Males
adult[['race','annual_salary']].groupby(['race']).mean()
adult.boxplot(column='hours-per-week', by='sex', figsize=(6,6))
adult.boxplot(column='hours-per-week', by='annual_salary', figsize=(6,6))
adult.boxplot(column='education-num', by='annual_salary', figsize=(6,6))
adult[['occupation','annual_salary']].groupby(['occupation']).mean()
import seaborn as sns
sns.countplot(y='workclass', hue='annual_salary', data = adult)
sns.countplot(y='native-country', hue='annual_salary', data = adult)
adult[['native-country','annual_salary']].groupby(['native-country']).mean()
sns.countplot(y='marital-status', hue='annual_salary', data = adult)
adult[['marital-status','annual_salary']].groupby(['marital-status']).mean()
def plot_correlation(adult, size=15):
    corr= adult.corr()
    fig, ax =plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()
plot_correlation(adult)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
adult['workclass'] = adult['workclass'].map({ 'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 
                                             'State-gov':5, 'Without-pay':6, 'Never-worked':7}).astype(int)
adult['occupation'] = adult['occupation'].map({'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 
                                               'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 
                                               'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13}).astype(int)
adult['native-country'] = adult['native-country'].map({'US':1,'Non-US':0}).astype(int)
adult['marital-status'] = adult['marital-status'].map({'Couple':0,'Single':1}).astype(int)
adult['relationship'] = adult['relationship'].map({' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}).astype(int)
adult['race']= adult['race'].map({' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}).astype(int)
adult.drop(labels=['education'],axis=1,inplace=True)
adult.head(10)
adult.dtypes
X = adult.drop(['annual_salary'], axis =1)  #input is entire table except for annual_salary because annual_salary is the target
y = adult['annual_salary']        #target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=0)   #50% data for testing
clf=KNeighborsClassifier(5)  #K Nearest Neighbour
fit=clf.fit(X_train, y_train)
predicted=fit.predict(X_test)
cm=confusion_matrix(y_test,predicted)
cm
from sklearn.metrics import classification_report
classification_report(y_test,predicted)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))  #Accuracy of the Model for k=5
t=DecisionTreeClassifier(criterion='entropy', max_depth=8)     #Decision Tree
fit=t.fit(X_train, y_train)
y_pre= fit.predict(X_test)
cm=confusion_matrix(y_test, y_pre)
cm
classification_report(y_test,y_pre)
scores=cross_val_score(t, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))  #Accuracy for max_depth=8
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)   #40% data for testing
clf=KNeighborsClassifier(5)
fit=clf.fit(X_train, y_train)

predicted=fit.predict(X_test)

cm=confusion_matrix(y_test,predicted)
cm
classification_report(y_test,predicted)
scores=cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))
t=DecisionTreeClassifier(criterion='entropy', max_depth=8)
fit=t.fit(X_train, y_train)
y_pre= fit.predict(X_test)
cm=confusion_matrix(y_test, y_pre)
cm
classification_report(y_test,y_pre)
scores=cross_val_score(t, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)  #Test data size 20%
clf=KNeighborsClassifier(5)
fit=clf.fit(X_train, y_train)
predicted=fit.predict(X_test)
cm=confusion_matrix(y_test,predicted)
cm
classification_report(y_test,predicted)
scores=cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))
t=DecisionTreeClassifier(criterion='entropy', max_depth=8)
fit=t.fit(X_train, y_train)
y_pre= fit.predict(X_test)
cm=confusion_matrix(y_test, y_pre)
cm
classification_report(y_test,y_pre)
scores=cross_val_score(t, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))