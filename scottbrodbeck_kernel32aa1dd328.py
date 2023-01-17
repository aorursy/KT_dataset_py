import pandas as pd

SP = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
SP.describe()
SP.info()
SP.isna().sum()
y = SP['test preparation course']

y.head()
SP.corr()
X=SP[['gender','race/ethnicity','math score','lunch','parental level of education','writing score','reading score']]

X.head()
X.isna().sum()
X.corr()
X=pd.get_dummies(X,drop_first='True')

X.head()
X.corr()
from sklearn import preprocessing

cols=X.columns

x=X.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled=min_max_scaler.fit_transform(X)

X=pd.DataFrame(x_scaled,index=X.index,columns=cols)

X.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=25,random_state=0)





from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(random_state=0)



logreg.fit(X_train,y_train)

predictions=logreg.predict(X_test)

print(predictions)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
