import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix,accuracy_score



#exploratory

df = pd.read_csv('../input/heart.csv')



pd.set_option('display.max_columns', None)

df.info()

df.describe()

df.columns



plt.figure(figsize=(10, 10))

sns.heatmap(df.corr(), annot = True, cbar = True)

sns.heatmap(df[['age', 'sex', 'trestbps', 'chol','thalach', 'oldpeak']].corr(), annot = True, cbar = True)

plt.title("corr")



# basic da / age-sex

df.groupby(['sex'])['sex'].count()



plt.figure(figsize=(10, 10))

sns.catplot(x="age", col="sex", data=df, kind="count", height=11, aspect=.7); 



bins = [0, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]

names =   ['<29', '29-34', '34-39', '39-44', '44-49', '49-54', '54-59', '59-64', '64-69','69+']

df['AgeRange'] = pd.cut(df['age'], bins, labels=names)



df.groupby(['AgeRange'])['age'].count()

df.groupby(['AgeRange'])['age'].count().plot()



df[df['sex']==0].groupby(['AgeRange'])['age'].count().plot()

plt.title('Frequency for Ages / sex = 0')

plt.xlabel('Age')

plt.ylabel('Count')

plt.savefig('heartDiseaseAndAges.png')

plt.show()





df[df['sex']==1].groupby(['AgeRange'])['age'].count().plot()

plt.title('Frequency for Ages / sex=1')

plt.xlabel('Age')

plt.ylabel('Count')

plt.savefig('heartDiseaseAndAges.png')

plt.show()



# prepare RM

cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for i in cols:

    df[i]=df[i].astype(object);

    

df.info()



df=pd.get_dummies(df)

df.head





#logistic regression

# Importing the dataset

X = df.iloc[:, [2, 3]].values

y = df.iloc[:, 4].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = 0.25, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))










