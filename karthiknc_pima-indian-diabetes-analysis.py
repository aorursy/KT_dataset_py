import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler



from sklearn import metrics

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.isna().sum()
data.describe().T
corr=data.corr()
plt.figure(figsize=[10,7])

sns.heatmap(corr,annot=True)
k=sns.countplot(data["Outcome"])

for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))
sns.pairplot(data)
features = ['Pregnancies','Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#replacing the 0s in the data

for feature in features:

    number = np.random.normal(data[feature].mean(), data[feature].std()/2)

    data[feature].fillna(value=number, inplace=True)
#Scaling the data

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = data.Outcome
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
#Logistic Regression

lr = LogisticRegression()

lr.fit(X_train,y_train)

acc = lr.score(X_test,y_test)*100

print("Logistic Regression Acc Score: ", acc)
#Naive Bayes

nb = GaussianNB()

nb.fit(X_train, y_train)

model_nb=nb.predict(X_test)

print("Naive Bayes Acc Score",(accuracy_score(y_test,model_nb))*100)
#K-Nearest Neigbors

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

model_knn = knn.predict(X_test)

print("KNN Acc Score",(accuracy_score(y_test,model_knn))*100)