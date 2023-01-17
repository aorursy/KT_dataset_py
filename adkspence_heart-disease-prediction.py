# Import of libraries we will be using in analyzing the data and creating the model

import numpy as np # Library for Linear algebra

import pandas as pd # Data processing functionalities

import matplotlib.pyplot as plt # Plotting graphs

import seaborn as sns # Customizing graphs
# Load the dataset which will be used for analysis and training model

heart_data = pd.read_csv("../input/heart.csv")
n_rows, n_cols = heart_data.shape

print(f"There exists {n_rows} domain instances with {n_cols} features in the dataset.")
features = list(heart_data.columns)



for feature in range(len(features)):

    print("Column {0} in the dataset is {1}".format(feature+1, features[feature].title()))
# The head method operates on a dataframe by displaying a number of rows. The first 5 rows are displayed if no arguments are passed.

heart_data.head()
# Checking to see if there are any null values in our dataset.

heart_data.isnull().any()
# Checking to see if there are any duplicated data in dataset

heart_data[heart_data.duplicated() == True]
# Removing duplicate data

heart_data.drop_duplicates(inplace=True)



n_rows, n_cols = heart_data.shape

print(f"After removing duplicate data we now have {n_rows} domain instances.")
# The pandas dataframe object created i.e. heart_data enables us to retrieve data using column headers

# First thing we are interested in is the distribution of patients on both ends. i.e. diseased and not diseased

heart_data.groupby('target').size()
not_diseased = len(heart_data[heart_data.target == 0])

diseased = len(heart_data[heart_data.target == 1])

print(f"The percentage of diseased patients within this dataset is {round((diseased/len(heart_data.target)), 2)*100}% leaving {round((not_diseased/len(heart_data.target)), 2)*100}% of the subjects as patients diagnosed to not have the heart disease.")
sns.countplot(x='target', data=heart_data, palette='mako_r')

plt.xlabel("Class Labels: 0 = not diseased, 1 = diseased")

plt.show()
male_gender = len(heart_data[heart_data.sex == 1])

female_gender = len(heart_data[heart_data.sex == 0])



print("In this dataset there exists {0} male subjects and {1} female subjects which computes to {2}% for males and {3}% for females.".format(male_gender, female_gender, round((male_gender/len(heart_data.sex)), 2)*100, round((female_gender/len(heart_data.sex)), 2)*100))
# Visualizing the distribution of Male and Female genders in the data

sns.countplot(x='sex', data=heart_data, palette='gist_rainbow')

plt.xlabel("Sex: 0 => Female, 1 => Male")

plt.show()
# We can utilize the crosstab method in the pandas library to analyze how gender impacts a person's chance of getting a heart disease

gender_impact = pd.crosstab(heart_data['sex'], heart_data['target'])

gender_impact
# We can go further to visualize these stats for a clearer view

gender_impact.plot(kind='bar', stacked=False, color=['#00e676', '#d50000'])
heart_data.describe()
# This provides us with the 'mean' subset of the describe() method

heart_data.groupby('target').mean()
pd.crosstab(heart_data.age,heart_data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
# Population distribution for heart disease



x = heart_data.groupby(['age','target']).agg({'sex':'count'})

y = heart_data.groupby(['age']).agg({'sex':'count'})

z = (x.div(y, level='age') * 100)

q= 100 - z

bin_x = range(25,80,2)



fig, axes = plt.subplots(2,2, figsize = (20,12))

plt.subplots_adjust(hspace = 0.5)



axes[0,0].hist(heart_data[heart_data['target']==1].age.tolist(),bins=bin_x,rwidth=0.8)

axes[0,0].set_xticks(range(25,80,2))

axes[0,0].set_xlabel('Age Range',fontsize=15)

axes[0,0].set_ylabel('Population Count',fontsize=15)

axes[0,0].set_title('People suffering from heart disease',fontsize=20)



axes[0,1].hist(heart_data[heart_data['target']==0].age.tolist(),bins=bin_x,rwidth=0.8)

axes[0,1].set_xticks(range(25,80,2))

axes[0,1].set_xlabel('Age Range',fontsize=15)

axes[0,1].set_ylabel('Population Count',fontsize=15)

axes[0,1].set_title('People not suffering from heart disease',fontsize=20)



axes[1,0].scatter(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex,s=(x.xs(1,level=1).sex)*30,edgecolors = 'r',c = 'yellow')

axes[1,0].plot(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex)

axes[1,0].set_xticks(range(25,80,2))

axes[1,0].set_yticks(range(0,110,5))

axes[1,0].set_xlabel('Age',fontsize=15)

axes[1,0].set_ylabel('%',fontsize=15)

axes[1,0].set_title('% of people with heart disease by age',fontsize=20)



axes[1,1].scatter(z.xs(1,level=1).reset_index().age,q.xs(1,level=1).reset_index().sex,s=(x.xs(0,level=1).sex)*30,edgecolors = 'r',c = 'yellow')

axes[1,1].plot(z.xs(1,level=1).reset_index().age,q.xs(1,level=1).reset_index().sex)

axes[1,1].set_xticks(range(25,80,2))

axes[1,1].set_yticks(range(0,110,5))

axes[1,1].set_xlabel('Age',fontsize=15)

axes[1,1].set_ylabel('%',fontsize=15)

axes[1,1].set_title('% of people with no heart disease by age',fontsize=20)



plt.show()
fig, axes = plt.subplots(6,2, figsize = (20,40))

plt.subplots_adjust(hspace = 0.5)



axes[0,0].scatter(heart_data[heart_data['target']==0][['age','thalach']].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','thalach']].sort_values(by = ['age']).thalach, c = 'g',label = 'target=0')

axes[0,0].scatter(heart_data[heart_data['target']==1][['age','thalach']].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','thalach']].sort_values(by = ['age']).thalach, c = 'r',label = 'target=1')

axes[0,0].set_title('thalach distribution',fontsize=20)

axes[0,0].set_xticks(range(25,80,2))

axes[0,0].set_xlabel('Age',fontsize=15)

axes[0,0].set_ylabel('thalach',fontsize=15)

axes[0,0].axhline(np.mean(heart_data['thalach']),xmin=0,xmax=1,linewidth=1, color='black',linestyle = '--')

axes[0,0].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')

axes[0,0].legend()



axes[0,1].scatter(heart_data[heart_data['target']==0][['age','trestbps']].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','trestbps']].sort_values(by = ['age']).trestbps, c = 'g',label = 'target=0')

axes[0,1].scatter(heart_data[heart_data['target']==1][['age','trestbps']].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','trestbps']].sort_values(by = ['age']).trestbps, c = 'r',label = 'target=1')

axes[0,1].set_title('trestbps distribution',fontsize=20)

axes[0,1].set_xticks(range(25,80,2))

axes[0,1].set_xlabel('Age',fontsize=15)

axes[0,1].set_ylabel('trestbps',fontsize=15)

axes[0,1].axhline(np.mean(heart_data['trestbps']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[0,1].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



# heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age'])

axes[1,0].scatter(heart_data[heart_data['target']==0][['age','chol',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','chol',]].sort_values(by = ['age']).chol,c = 'g',label = 'target=0')

axes[1,0].scatter(heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age']).chol,c = 'r',label = 'target=1')

axes[1,0].set_title('chol distribution',fontsize=20)

axes[1,0].set_xticks(range(25,80,2))

axes[1,0].set_xlabel('Age',fontsize=15)

axes[1,0].set_ylabel('chol',fontsize=15)

axes[1,0].axhline(np.mean(heart_data['chol']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[1,0].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



axes[1,1].scatter(heart_data[heart_data['target']==0][['age','oldpeak',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','oldpeak',]].sort_values(by = ['age']).oldpeak,c = 'g',label = 'target=0')

axes[1,1].scatter(heart_data[heart_data['target']==1][['age','oldpeak',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','oldpeak',]].sort_values(by = ['age']).oldpeak,c = 'r',label = 'target=1')

axes[1,1].set_title('oldpeak distribution',fontsize=20)

axes[1,1].set_xticks(range(25,80,2))

axes[1,1].set_xlabel('Age',fontsize=15)

axes[1,1].set_ylabel('oldpeak',fontsize=15)

axes[1,1].axhline(np.mean(heart_data['oldpeak']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[1,1].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



fbs_count = heart_data['fbs'].value_counts()

labels = [('fbs = '+ str(x)) for x in fbs_count.index]

axes[2,0].pie(fbs_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[2,0].axis('equal')

axes[2,0].set_title('fbs share',fontsize=15)



restecg_count = heart_data['restecg'].value_counts()

labels = [('restecg = '+ str(x)) for x in restecg_count.index]

axes[2,1].pie(restecg_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45,explode = [0,0,0.5])

axes[2,1].axis('equal')

axes[2,1].set_title('restecg share',fontsize=15)



exang_count = heart_data['exang'].value_counts()

labels = [('exang = '+ str(x)) for x in exang_count.index]

axes[3,0].pie(exang_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[3,0].axis('equal')

axes[3,0].set_title('exang share',fontsize=15)



slope_count = heart_data['slope'].value_counts()

labels = [('slope = '+ str(x)) for x in slope_count.index]

axes[3,1].pie(slope_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[3,1].axis('equal')

axes[3,1].set_title('slope share',fontsize=15)



ca_count = heart_data['ca'].value_counts()

labels = [('ca = '+ str(x)) for x in ca_count.index]

axes[4,0].pie(ca_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[4,0].axis('equal')

axes[4,0].set_title('ca share',fontsize=15)



thal_count = heart_data['thal'].value_counts()

labels = [('thal = '+ str(x)) for x in thal_count.index]

axes[4,1].pie(thal_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[4,1].axis('equal')

axes[4,1].set_title('thal share',fontsize=15)



cp_count = heart_data['cp'].value_counts()

labels = [('cp = '+ str(x)) for x in cp_count.index]

axes[5,0].pie(cp_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[5,0].axis('equal')

axes[5,0].set_title('CP share',fontsize=15)



target_count = heart_data['target'].value_counts()

labels = [('target = '+ str(x)) for x in target_count.index]

axes[5,1].pie(target_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[5,1].axis('equal')

axes[5,1].set_title('target share',fontsize=15)



plt.show()
# Produce a correlation matrix to reveal how independent features within the data affect the target

correlations = heart_data.corr()

pd.DataFrame(correlations['target']).sort_values(by='target', ascending=False)
# Visual representation

plt.figure(figsize=(14,10))

sns.heatmap(heart_data.corr(), linewidths=.01, annot = True, cmap='coolwarm')

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



X = heart_data.iloc[:, :-1]

y = heart_data.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Model

heart_model = LogisticRegression()

heart_model.fit(X_train, y_train)



# Making predictions

predictions = heart_model.predict(X_test)



# Checking the Accuracy of predictions

print("Accuracy ", heart_model.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(predictions, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
print(heart_model.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]))
print(heart_model.predict([[65, 1, 0, 135, 254, 0, 0, 127, 0, 2.8, 1, 1, 3]]))
# Model using decision trees

heart_model_dt = DecisionTreeClassifier()



#fiting the model

heart_model_dt.fit(X_train, y_train)



#prediction

dt_predictions = heart_model_dt.predict(X_test)



#Accuracy

print("Accuracy ", heart_model_dt.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(dt_predictions, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
print(heart_model_dt.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]))
print(heart_model_dt.predict([[65, 1, 0, 135, 254, 0, 0, 127, 0, 2.8, 1, 1, 3]]))