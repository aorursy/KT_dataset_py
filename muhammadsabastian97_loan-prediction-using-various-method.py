# Importing required libraries

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer
# Importing required dataset

dataset = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
# Check the number of null values in each column

dataset.isnull().sum()
# Splitting data based on its category

cat_data = []

num_data = []



for i,c in enumerate(dataset.dtypes):

    if c == object:

        cat_data.append(dataset.iloc[:, i])

    else :

        num_data.append(dataset.iloc[:, i])



cat_data=pd.DataFrame(cat_data).transpose()

num_data=pd.DataFrame(num_data).transpose()
# Filling the missing data

cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))



imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')

imputer.fit(num_data.values[:,(2,3)])

num_data.values[:,(2,3)] = imputer.transform(num_data.values[:,(2,3)])





imputer_mf=SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_mf.fit(num_data.values[:, 4:5])

num_data.values[:,4:5]=imputer_mf.transform(num_data.values[:,4:5])
# Check whether there is still a null value in the dataset

cat_data.isnull().sum().any()

num_data.isnull().sum().any()
# Encoding the label and splitting data into independent and dependent variables

labelencoder = LabelEncoder()

cat_data.values[:, 2] = labelencoder.fit_transform(cat_data.values[:, 2])

cat_data.values[:, 4] = labelencoder.fit_transform(cat_data.values[:, 4])

cat_data.values[:, 5] = labelencoder.fit_transform(cat_data.values[:, 5])

cat_data.values[:, 6] = labelencoder.fit_transform(cat_data.values[:, 6])

cat_data.Dependents=cat_data.Dependents.replace({'3+':'3'})



target_values = {'Y': 1 , 'N' : 0}



target = cat_data['Loan_Status']

cat_data.drop('Loan_Status', axis=1, inplace=True)



target = target.map(target_values)



data = pd.concat([cat_data, num_data, target], axis=1)

X=data.iloc[:, 2:12]

y=data.iloc[:,12]

# Encoding the 'Property_Area' column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()                     

X.iloc[:, 4] = labelencoder_X.fit_transform(X.iloc[:, 4])     

transformer = ColumnTransformer(

        [('Property_Area', OneHotEncoder(), [4])],

        remainder='passthrough')

X = np.array(transformer.fit_transform(X), dtype=np.float)

# Splitting data into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
""" LOGISTIC REGRESSION """

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm_logreg = confusion_matrix(y_test, y_pred)

as_logreg=accuracy_score(y_test, y_pred)



""" K-NEAREST NEIGHBORS """

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_knn = confusion_matrix(y_test, y_pred)

as_knn=accuracy_score(y_test, y_pred)



""" SVM GAUSSIAN """

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_svm_gaussian = confusion_matrix(y_test, y_pred)

as_svm_gaussian = accuracy_score(y_test, y_pred)



""" SVM NO KERNEL """

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_svm_nokernel = confusion_matrix(y_test, y_pred)

as_svm_nokernel = accuracy_score(y_test, y_pred)



""" NAIVE BAYES """

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_nb = confusion_matrix(y_test, y_pred)

as_nb = accuracy_score(y_test, y_pred)



""" DECISION TREE CLASSIFICATION """

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_dtc = confusion_matrix(y_test, y_pred)

as_dtc = accuracy_score(y_test, y_pred)



""" RANDOM FOREST CLASSIFIER """

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating confusion matrix and calculating the accuracy score

cm_rfc = confusion_matrix(y_test, y_pred)

as_rfc = accuracy_score(y_test, y_pred)
# Evaluating the best method to use in this loan prediction case

score={'as_logreg':as_logreg, 'as_knn':as_knn, 'as_svm_gaussian':as_svm_gaussian, 'as_svm_nokernel':as_svm_nokernel, 'as_nb':as_nb, 'as_dtc':as_dtc, 'as_rfc':as_rfc}

score_list=[]

for i in score:

    score_list.append(score[i])

    u=max(score_list)

    if score[i]==u:

        v=i  

    print(f"{i}={score[i]}");   

print(f"The best accuracy score in this case is {v} with accuracy score {u}")