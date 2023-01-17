import pandas as pd

 

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt



%matplotlib inline



import warnings



warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/titanic-dataset/train.csv')



df_train.head()
df_test = pd.read_csv('../input/titanic-dataset/test.csv')



df_test.head()
print("Shape of the Train data ::  ", df_train.shape)



print("\n")



print("Shape of the Test data ::  ", df_test.shape)

print("      ---- Train Data Info ----      \n")



df_train.info()
print("      ---- Test Data Info ----      \n")



df_test.info()
total_null_train = df_train.isnull().sum().sort_values(ascending = False)



total_null_test = df_test.isnull().sum().sort_values(ascending = False)



print("         ---- Total Null Values In Train Data -----")



print(total_null_train)



print("\n\n")



print("         ---- Total Null Values In Test Data -----")



print(total_null_test)
total_null_percent_train = round(df_train.isnull().sum()/df_train.isnull().count()*100, 2).sort_values(ascending = False)



total_null_percent_test = round(df_test.isnull().sum()/df_test.isnull().count()*100, 2).sort_values(ascending = False)



print("         ---- Percentage Of Total Null Values In Train Data (in %) -----")



print(total_null_percent_train)



print("\n\n")



print("         ---- Percentage Of Total Null Values In Test Data (in %) -----")



print(total_null_percent_test)
df_train.describe().head()
df_test.describe().head()
sns.countplot(df_train['Survived'])



plt.title('Count Of Survived Passengers', color = 'r')



plt.show()
sns.countplot(df_train['Pclass'])



plt.title('Count Of Pclass Passengers', color = 'r')



plt.show()
sns.barplot(x = 'Pclass', y = 'Survived', data = df_train)



plt.title('Pclass Vs Survived', color = 'r')



plt.show()
sns.countplot(df_train['Sex'])



plt.title('Count Of Male And Female Passengers', color = 'r')



plt.show()
sns.barplot(x = 'Sex', y = 'Survived', data = df_train)



plt.title('Gender Vs Survived', color = 'r')



plt.show()
plt.hist(df_train['Age'], edgecolor = 'y')



plt.xlabel('Age')



plt.ylabel('Frequency Count')



plt.title('Count Of The passengers With Different Ages', color = 'r')



plt.show()
sns.barplot(y = 'Age', x = 'Survived', data = df_train)



plt.title('Age Vs Survived', color = 'r')



plt.show()
sns.boxplot(y = 'Age', x = 'Survived', data = df_train)



plt.title('Age Vs Survived', color = 'r')



plt.show()
sns.countplot(df_train['SibSp'])



plt.title('Count Of Siblings And Spouse Along With Passengers', color = 'r')



plt.show()
sns.barplot(x = 'SibSp', y = 'Survived', data = df_train)



plt.title('SibSp Vs Survived', color = 'r')



plt.show()
sns.countplot(df_train['Parch'])



plt.title('Count Of Parch Along With Passengers', color = 'r')



plt.show()
sns.barplot(x = 'Parch', y = 'Survived', data = df_train)



plt.title('Parch Vs Survived', color = 'r')



plt.show()
plt.hist(df_train['Fare'], edgecolor = 'y')



plt.xlabel('Fare')



plt.ylabel('Frequency Count')



plt.title('Count Of The Passengers For Different Fare', color = 'r')



plt.show()
sns.barplot(y = 'Fare', x = 'Survived', data = df_train)



plt.title('Fare Vs Survived', color = 'r')



plt.show()
sns.boxplot(y = 'Fare', x = 'Survived', data = df_train)



plt.title('Fare Vs Survived', color = 'r')



plt.show()
sns.countplot(df_train['Embarked'])



plt.title('Count Of Embarked', color = 'r')



plt.show()
sns.barplot(x = 'Embarked', y = 'Survived', data = df_train)



plt.title('Embarked Vs Survived', color = 'r')



plt.show()
df_train_1 = df_train.drop(['PassengerId'], axis = 1)



df_test_1 = df_test.copy()



df_train_1.head()

total_null_train = df_train_1.isnull().sum().sort_values(ascending = False)



total_null_test = df_test_1.isnull().sum().sort_values(ascending = False)



print("         ---- Total Null Values In Train Data -----")



print(total_null_train)



print("\n\n")



print("         ---- Total Null Values In Test Data -----")



print(total_null_test)
print("Number of Unique Cabins :: ", df_train_1['Cabin'].nunique())



print("\n")



print("The Unique Cabins Are :: ", df_train_1['Cabin'].unique())
dict_cabins = {'A' : 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5, 'F' : 6, 'G' : 7, 'T' : 8, 'U' : 9} 



data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset['Cabin'] = dataset['Cabin'].fillna('U0')

    

    dataset['Cabin_Updated'] = dataset['Cabin'].str[0]

    

    dataset['Cabin_Updated'] = dataset['Cabin_Updated'].map(dict_cabins)

    

    dataset['Cabin_Updated'] = dataset['Cabin_Updated'].fillna(0)

    

    dataset['Cabin_Updated'] = dataset['Cabin_Updated'].astype(int)

    

df_train_1.head()
data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    mean_age = dataset['Age'].mean()

    

    std_age = dataset['Age'].std()

    

    is_null = dataset['Age'].isnull().sum()

    

    random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = is_null)

    

    age_updated = dataset['Age'].copy()

    

    age_updated[np.isnan(age_updated)] = random_age

    

    dataset['Age'] = age_updated

    

    dataset['Age'] = dataset['Age'].astype(int)



df_train_1.head()
data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset['Prefix'] = dataset['Name'].str.extract('([A-Z a-z]+)\.')

    

    print(dataset['Prefix'].value_counts(), "\n")



df_train_1.head()
dict_prefix = {" Mr" : 1, " Miss" : 2, " Mrs" : 3, " Master" : 4, " Others" : 5}



data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset['Prefix_Updated'] = dataset['Prefix'].replace([' Lady', ' Ms', ' Mme', ' Mlle', ' the Countess'], ' Mrs', inplace = True)

    

    dataset['Prefix_Updated'] = dataset['Prefix'].replace([' Sir', ' Rev'], ' Mr', inplace = True)

    

    dataset['Prefix_Updated'] = dataset['Prefix'].replace([' Capt', ' Col', ' Don', ' Dr', ' Major', ' Jonkheer', ' Dona'], ' Others', inplace = True)

    

    dataset['Prefix_Updated'] = dataset['Prefix'].map(dict_prefix)

    

df_train_1.head()
dict_embarked = {"S" : 0, "C" : 1, "Q" : 2}



data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset['Embarked_Updated'] = dataset['Embarked'].fillna('S')

    

    dataset['Embarked_Updated'] = dataset['Embarked_Updated'].map(dict_embarked)

    

df_train_1.head()
data_list = [df_train_1, df_test_1]



for dataset in data_list:

   

    dataset['Fare'] = dataset['Fare'].fillna(0)

    

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    print("\n--- Fare Value Count ----\n", dataset.Fare.value_counts())

    

    dataset.loc[dataset.Fare <= 7, 'Fare'] = 1

    

    dataset.loc[(dataset.Fare > 7) & (dataset.Fare <= 13), 'Fare'] = 2

    

    dataset.loc[(dataset.Fare > 13) & (dataset.Fare <= 25), 'Fare'] = 3

    

    dataset.loc[(dataset.Fare > 25) & (dataset.Fare <= 50), 'Fare'] = 4

    

    dataset.loc[dataset.Fare > 50, 'Fare'] = 5

    

    print("\n---- Updated Fare Value Count ---\n", dataset.Fare.value_counts())

    

df_train_1.head()
dict_sex = {"female" : 0, "male" : 1}



data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset['Sex'] = dataset['Sex'].map(dict_sex)

    

df_train_1.head()
data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    print("\n---- Age Value Count ----\n", dataset.Age.value_counts())

    

    dataset.loc[dataset.Age <= 20, 'Age'] = 1

    

    dataset.loc[(dataset.Age > 20) & (dataset.Age <= 30), 'Age'] = 2

    

    dataset.loc[(dataset.Age > 30) & (dataset.Age <= 40), 'Age'] = 3

    

    dataset.loc[dataset.Age > 40] = 4

    

    print("\n---- Updated Age Value Count ----\n", dataset.Age.value_counts())

    

df_train_1.head()
data_list = [df_train_1, df_test_1]



for dataset in data_list:

    

    dataset = dataset.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Prefix'], axis = 1, inplace = True)

    

df_train_1.head()
x_train = df_train_1.drop('Survived', axis = 1)



y_train = df_train['Survived']



x_test = df_test_1.drop('PassengerId', axis = 1)



print("Shape of x_train:: ", x_train.shape, "\ty_train:: ", y_train.shape, "\tx_test:: ",x_test.shape)



x_test.head()
from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression()



lr_model.fit(x_train, y_train)



lr_predicted = lr_model.predict(x_test)



lr_score = lr_model.score(x_train, y_train)



print("The Score Obtained By LogisticRegression Is :: ", lr_score)



lr_accuracy = round(lr_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By LogisticRegression Is :: ", lr_accuracy, "%")
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier()



rf_model.fit(x_train, y_train)



rf_predicted = rf_model.predict(x_test)



rf_score = rf_model.score(x_train, y_train)



print("The Score Obtained By RandomForestClassifier Is :: ", rf_score)



rf_accuracy = round(rf_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By RandomForestClassifier Is :: ", rf_accuracy, "%")
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier()



dt_model.fit(x_train, y_train)



dt_predicted = dt_model.predict(x_test)



dt_score = dt_model.score(x_train, y_train)



print("The Score Obtained By DecisionTreeClassifier Is :: ", dt_score)



dt_accuracy = round(dt_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By DecisionTreeClassifier Is :: ", dt_accuracy, "%")
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier()



knn_model.fit(x_train, y_train)



knn_predicted = knn_model.predict(x_test)



knn_score = knn_model.score(x_train, y_train)



print("The Score Obtained By KNeighborsClassifier Is :: ", knn_score)



knn_accuracy = round(knn_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By KNeighborsClassifier Is :: ", knn_accuracy, "%")
from sklearn.svm import SVC



sv_model = SVC()



sv_model.fit(x_train, y_train)



sv_predicted = sv_model.predict(x_test)



sv_score = sv_model.score(x_train, y_train)



print("The Score Obtained By SupportVectorClassifier Is :: ", sv_score)



sv_accuracy = round(sv_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By SupportVectorClassifier Is :: ", sv_accuracy, "%")
from sklearn.svm import LinearSVC



lsv_model = LinearSVC()



lsv_model.fit(x_train, y_train)



lsv_predicted = lsv_model.predict(x_test)



lsv_score = lsv_model.score(x_train, y_train)



print("The Score Obtained By SupportVectorClassifier Is :: ", lsv_score)



lsv_accuracy = round(lsv_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By SupportVectorClassifier Is :: ", lsv_accuracy, "%")
from sklearn.naive_bayes import GaussianNB



gnb_model = GaussianNB()



gnb_model.fit(x_train, y_train)



gnb_predicted = gnb_model.predict(x_test)



gnb_score = gnb_model.score(x_train, y_train)



print("The Score Obtained By GaussianNB Is :: ", gnb_score)



gnb_accuracy = round(gnb_model.score(x_train, y_train)*100, 2)



print("\nThe Accuracy Obtained By GaussianNB Is :: ", gnb_accuracy, "%")
accuracy = [("lr_accuracy", lr_accuracy), ("rf_accuracy", rf_accuracy), ("dt_accuracy", dt_accuracy),

            ("knn_accuracy", knn_accuracy), ("sv_accuracy", sv_accuracy), 

           ("lsv_accuracy", lsv_accuracy), ("gnb_accuracy", gnb_accuracy)]



labels, values = zip(*accuracy)

 

plt.figure(figsize = (16, 6))



plt.bar(labels, values, color = 'b')



plt.xticks(color = 'm', fontsize = 16)



plt.yticks(color = 'm', fontsize = 16)



plt.xlabel('Classifier', color = 'g', fontsize = 18)



plt.ylabel('Accuarcy in %', color = 'g', fontsize = 18)



plt.title('Accuracy Obtained for Different Classifiers', color = 'r', fontsize = 20)



plt.show()
titanic_final_submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'], 'Survived' : rf_predicted.astype(int)})



titanic_final_submission.to_csv('titanic_final_submission.csv', index = False)



titanic_final_submission['Survived'].value_counts()

output_data = pd.read_csv('./titanic_final_submission.csv')



output_data