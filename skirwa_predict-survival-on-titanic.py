import numpy as np

import pandas as pd

from numpy import log



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import gridspec

import matplotlib

from matplotlib.colors import ListedColormap



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import math

from collections import Counter

import pandas_profiling as pp

import scipy.stats as stats



#configuration settings

%matplotlib inline

sns.set(color_codes=True)

titanic_survival_test_df = pd.read_csv("../input/titanic/test.csv")

titanic_survival_training_df = pd.read_csv("../input/titanic/train.csv")
#view the top 5 records for the training set

titanic_survival_training_df.head(5)
#view the top 5 records for the test set

titanic_survival_test_df.head(5)
#profile report generation

pp.ProfileReport(titanic_survival_training_df)
pp.ProfileReport(titanic_survival_test_df)
titanic_survival_training_df = titanic_survival_training_df.drop(["Name", "Cabin", "Embarked", "Ticket", "Age"], axis=1)

titanic_survival_test_df = titanic_survival_test_df.drop(["Name", "Cabin", "Embarked", "Ticket", "Age"], axis=1)
titanic_survival_training_df.head()
titanic_survival_test_df.head()
#check the shape of the records to know how many records are in the training dataset

titanic_survival_training_df.shape
# Check for rows containing duplicate data in the training set

duplicate_rows_df = titanic_survival_training_df[titanic_survival_training_df.duplicated()]

print("Number of duplicate rows: ", duplicate_rows_df.shape)
# Finding the null values in the training set.

titanic_survival_training_df.isnull().sum()
# Finding the null values in the test set.

titanic_survival_test_df.isnull().sum()
#Filling the missing value in the training set

titanic_survival_test_df['Fare'].fillna((titanic_survival_test_df['Fare'].mean()), inplace=True)
# Finding the null values in the test set.

titanic_survival_test_df.isnull().sum()
#plotting a boxplot

sns.boxplot(x=titanic_survival_training_df["Fare"])
titanic_survival_training_df.boxplot(column=['Fare'], by=["Survived"])
#proportion of the 'Survived' variable

survived_vc = titanic_survival_training_df['Survived'].value_counts()

survived_df = survived_vc.rename_axis('survived').reset_index(name='counts')

survived_df
#ploting a pie chart and a bar graph

# Define the labels

survived_label =  '0', '1'



#Choose which proportion to explode

survived_explode = (0,0.1)



# Create the container which will hold the subplots

survived_fig = plt.figure(figsize = (25,12))



# Create a frame using gridspec

gs = gridspec.GridSpec(6,7)



# Create subplots to visualize the pie chart

pie_ax01 = plt.subplot(gs[0:,:-3])

pie_ax01.set_title(label="Survival Rate",fontdict={"fontsize":25})

pie_ax01.pie(survived_df["counts"],

            explode = survived_explode,

            autopct = "%1.1f%%",

            shadow = True,

            startangle = 90,

            textprops ={"fontsize":22})

pie_ax01.legend(survived_label, loc = 0, fontsize = 18, ncol=2)



# Set subplot to visualize the bargraph

bar_ax01 = plt.subplot(gs[:6,4:])

survived_label_list = survived_df["survived"]

survived_freq = survived_df["counts"]

index = np.arange(len(survived_label_list))

width = 1/1.5



bar_ax01.set_title(label="Survival Rate",fontdict={"fontsize":25})

bar_ax01.set_xlabel(xlabel="Survived",fontdict={"fontsize":25})

bar_ax01.set_ylabel(ylabel="Count",fontdict={"fontsize":25})

bar_ax01.set_xticklabels(survived_label_list,rotation="vertical",fontdict={"fontsize":25})

bar_ax01.bar(survived_label_list,survived_freq,width,color="blue")



plt.tight_layout(pad=5)
# Checking for imbalance in the target variable

def balance_calc(data, unit='natural'):

    base = {

        'shannon' : 2.,

        'natural' : math.exp(1),

        'hartley' : 10.

    }

    if len(data) <= 1:

        return 0

    

    counts = Counter()

    

    for d in data:

        counts[d] += 1

    

    ent = 0

    

    probs = [float(c) / len(data) for c in counts.values()]

    for p in probs:

        if p > 0.:

            ent -= p * math.log(p, base[unit])

            

    return ent/math.log(len(data))
balance_calc(titanic_survival_training_df["Survived"],'shannon')
#plotting a histogram with a fitted normal distribution

sns.distplot(titanic_survival_training_df["Survived"], fit=stats.norm, color='red', kde = False)
#skewness

titanic_survival_training_df["Survived"].skew(axis = 0)
#kurtosis

titanic_survival_training_df["Survived"].kurt()
#Finding the relations between the variables.

plt.figure(figsize = (20,10))

correlation = titanic_survival_training_df.corr()

sns.heatmap(correlation, cmap='BrBG', annot=True)

correlation
titanic_survival_training_df.head()
#training set and test set

X = titanic_survival_training_df.iloc[:, [6]].values

y = titanic_survival_training_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting SVM to the training set

svm_fare_model = SVC(kernel='rbf', random_state=0)

svm_fare_model.fit(X_train, y_train)
# Predicting the test set results

y_pred = svm_fare_model.predict(X_test)
# Accuracy score

svm_fare_model.score(X_train, y_train)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test, y_pred))
#training set and test set

X = titanic_survival_training_df.iloc[:, [2]].values

y = titanic_survival_training_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting SVM to the training set

svm_class_model = SVC(kernel='rbf', random_state=0)

svm_class_model.fit(X_train, y_train)
# Predicting the test set results

y_pred = svm_class_model.predict(X_test)
# Accuracy score

svm_class_model.score(X_train, y_train)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test, y_pred))
#training set and test set

X = titanic_survival_training_df.iloc[:, [4,5]].values

y = titanic_survival_training_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting SVM to the training set

svm_sp_model = SVC(kernel='rbf', random_state=0)

svm_sp_model.fit(X_train, y_train)
# Predicting the test set results

y_pred = svm_sp_model.predict(X_test)
# Accuracy score

svm_sp_model.score(X_train, y_train)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test, y_pred))
#Encoding the 'Sex' variable

#Frequency encoding

fe = titanic_survival_training_df.groupby('Sex').size()/len(titanic_survival_training_df)

titanic_survival_training_df.loc[:,'Sex_Enc'] = titanic_survival_training_df['Sex'].map(fe)

titanic_survival_training_df.sample(5)
#training set and test set

X = titanic_survival_training_df.iloc[:, [7]].values

y = titanic_survival_training_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting SVM to the training set

svm_sex_model = SVC(kernel='rbf', random_state=0)

svm_sex_model.fit(X_train, y_train)
# Predicting the test set results

y_pred = svm_sex_model.predict(X_test)
# Accuracy score

svm_sex_model.score(X_train, y_train)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test, y_pred))
titanic_survival_training_df.sample(5)
#training set and test set

X = titanic_survival_training_df.iloc[:, [2,4,5,7]].values

y = titanic_survival_training_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting SVM to the training set

svm_all_model = SVC(kernel='rbf', random_state=0)

svm_all_model.fit(X_train, y_train)
# Predicting the test set results

y_pred = svm_all_model.predict(X_test)
# Accuracy score

svm_all_model.score(X_train, y_train)
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test, y_pred))
titanic_survival_test_df.head()
#Encoding the 'Sex' variable in the test set

#Frequency encoding

fe = titanic_survival_test_df.groupby('Sex').size()/len(titanic_survival_test_df)

titanic_survival_test_df.loc[:,'Sex_Enc'] = titanic_survival_test_df['Sex'].map(fe)

titanic_survival_test_df.sample(5)
X_test2 = titanic_survival_test_df.iloc[:, [1,3,4,6]].values
# Predicting the test set results using the svm_all_model

y_pred = svm_all_model.predict(X_test2)
y_pred