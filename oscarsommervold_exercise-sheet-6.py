import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn import neighbors

import seaborn as sns
diabetes_df = pd.read_csv('../input/diabetes/diabetes.csv')

diabetes_df
diabetes_copy = diabetes_df.copy()
diabetes_df.Glucose = diabetes_df.Glucose.replace(0,np.nan)

diabetes_df.BloodPressure = diabetes_df.BloodPressure.replace(0,np.nan)

diabetes_df.SkinThickness = diabetes_df.SkinThickness.replace(0,np.nan)

diabetes_df.Insulin = diabetes_df.Insulin.replace(0,np.nan)

diabetes_df.BMI = diabetes_df.BMI.replace(0,np.nan)
diabetes_df
glucose = pd.DataFrame()

label = ['with zeros','with nan']

glucose['zero'] = diabetes_copy.Glucose

glucose['nan'] = diabetes_df.Glucose

plt.hist(glucose['zero'], alpha=0.5)

plt.hist(glucose['nan'], alpha=0.5)

plt.legend(label)

plt.title("Glucose")

plt.ylabel('Count')

plt.show()
bp = pd.DataFrame()

label = ['with zeros','with nan']

bp['zero'] = diabetes_copy.BloodPressure

bp['nan'] = diabetes_df.BloodPressure

plt.hist(bp['zero'], alpha=0.5)

plt.hist(bp['nan'], alpha=0.5)

plt.legend(label)

plt.title("BloodPressure")

plt.ylabel('Count')

plt.show()
st = pd.DataFrame()

label = ['with zeros','with nan']

st['zero'] = diabetes_copy.SkinThickness

st['nan'] = diabetes_df.SkinThickness

plt.hist(st['zero'], alpha=0.5)

plt.hist(st['nan'], alpha=0.5)

plt.legend(label)

plt.title("SkinThickness")

plt.ylabel('Count')

plt.show()
insulin = pd.DataFrame()

label = ['with zeros','with nan']

insulin['zero'] = diabetes_copy.Insulin

insulin['nan'] = diabetes_df.Insulin

plt.hist(insulin['zero'], alpha=0.5)

plt.hist(insulin['nan'], alpha=0.5)

plt.legend(label)

plt.title("Insulin")

plt.ylabel('Count')

plt.show()
bmi = pd.DataFrame()

label = ['with zeros','with nan']

bmi['zero'] = diabetes_copy.BMI

bmi['nan'] = diabetes_df.BMI

plt.hist(bmi['zero'], alpha=0.5)

plt.hist(bmi['nan'], alpha=0.5)

plt.legend(label)

plt.title("BMI")

plt.ylabel('Count')

plt.show()
X = diabetes_df.drop(columns = 'Outcome').values

y = diabetes_df.Outcome.values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.6, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,train_size = 0.6, test_size=0.4, random_state=42)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train_mean = imputer.fit_transform(X_train)

X_val_mean = imputer.fit_transform(X_val)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

X_train_median = imputer.fit_transform(X_train)

X_val_median = imputer.fit_transform(X_val)
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

X_train_mode = imputer.fit_transform(X_train)

X_val_mode = imputer.fit_transform(X_val)
k_list = [1, 5, 10, 20, 30]

val_accuracies_mean = []

val_accuracies_median = []

val_accuracies_mode= []

for idx,k in enumerate(k_list):

    knn_mean = neighbors.KNeighborsClassifier(k)

    knn_mean.fit(X_train_mean, y_train)

    val_accuracies_mean.append(accuracy_score(y_val,knn_mean.predict(X_val_mean)))

    

    knn_median= neighbors.KNeighborsClassifier(k)

    knn_median.fit(X_train_median, y_train)

    val_accuracies_median.append(accuracy_score(y_val,knn_median.predict(X_val_median)))

    

    knn_mode= neighbors.KNeighborsClassifier(k)

    knn_mode.fit(X_train_mode, y_train)

    val_accuracies_mode.append(accuracy_score(y_val,knn_mode.predict(X_val_mode)))

    
plt.figure(len(k_list), figsize=(12,10))

plt.plot(val_accuracies_mean) 

plt.plot(val_accuracies_median) 

plt.plot(val_accuracies_mode)

plt.legend(['Mean', 'Median','Mode'])

plt.show()

best = [('Mean',np.array(val_accuracies_mean).mean()),('Median',np.array(val_accuracies_median).mean()),('Most Frequent Value',np.array(val_accuracies_mode).mean())]

best
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

#X_train_median = imputer.fit_transform(X_train)

X_test_median = imputer.fit_transform(X_test)
val_accuracies = []

for idx,k in enumerate(k_list):

    knn = neighbors.KNeighborsClassifier(k)

    knn.fit(X_train_median, y_train)

    val_accuracies.append(accuracy_score(y_test,knn.predict(X_test_median)))
val_accuracies
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train_median, y_train)

clf.score(X_test_median,y_test)
titanic_df = pd.read_csv('../input/titanic/titanic.csv')
titanic_df['Title'] = titanic_df.Name.str.extract(" ([A-Za-z]+)\.")

plt.subplots(figsize=(15,10))

sns.barplot(x = titanic_df.Title.value_counts().index,y=titanic_df.Title.value_counts().values)

plt.show()
valid_titles = ["Master", "Miss", "Mr", "Mrs"]

titanic_df.loc[~titanic_df.Title.isin(valid_titles),'Title'] = "Other"

titanic_df.Title.value_counts()
# titlenum = [0,1,2,3,4]

titles = ["Master", "Miss", "Mr", "Mrs","Other"]

title_survival = []

for title in titles:

    title_survival.append(titanic_df.loc[titanic_df.Title == title,'Survived'].sum()/len(titanic_df.loc[titanic_df.Title == title]))
pd.DataFrame({'Title' : titles, 'Average Survival Rate': title_survival}).plot.bar(x = 'Title', y = 'Average Survival Rate', rot = 0, figsize = (10,5))
titanic_df.loc[titanic_df.Title == "Master",'Title'] = 0

titanic_df.loc[titanic_df.Title == "Miss",'Title'] = 1

titanic_df.loc[titanic_df.Title == "Mr",'Title'] = 2

titanic_df.loc[titanic_df.Title == "Mrs",'Title'] = 3

titanic_df.loc[titanic_df.Title == "Other",'Title'] = 4
X = titanic_df[['Fare','Title']].values

y = titanic_df.Survived.values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,train_size = 0.8, test_size=0.2, random_state=42)
k_list = [1, 5, 10, 20, 30]

accuracies_Pair = []

for idx,k in enumerate(k_list):

    knn_pair = neighbors.KNeighborsClassifier(k)

    knn_pair.fit(X_train, y_train)

    accuracies_Pair.append(accuracy_score(y_val,knn_pair.predict(X_val)))

    
X = titanic_df.Fare.values

y = titanic_df.Survived.values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,train_size = 0.8, test_size=0.2, random_state=42)
accuracies_Fare = []

for idx,k in enumerate(k_list):

    knn_fare = neighbors.KNeighborsClassifier(k)

    knn_fare.fit(X_train.reshape(-1, 1), y_train)

    accuracies_Fare.append(accuracy_score(y_val,knn_fare.predict(X_val.reshape(-1,1))))
plt.figure(len(k_list), figsize=(12,10))

plt.plot(accuracies_Fare) 

plt.plot(accuracies_Pair) 

plt.legend(['Fare', 'Fare And Title'])

plt.ylabel("Accuracy score on validation data")

plt.show()