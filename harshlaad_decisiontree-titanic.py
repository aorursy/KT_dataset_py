import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Binarizer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
data=pd.read_csv('../input/train.csv')

data.head()
y=data['Survived']

features=['Pclass','Sex','Age','SibSp','Fare','Embarked']

X=data[features]
X=pd.get_dummies(X)

X.head()
first_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

X=pd.DataFrame(first_imputer.fit_transform(X))
mean_pclass=X[0].mean()

bn_0 = Binarizer(threshold=mean_pclass)

X[0]= bn_0.transform([X[0]])[0]



mean_sex=X[1].mean()

bn_1=Binarizer(threshold=mean_sex)

X[1]=bn_1.transform([X[1]])[0]



mean_age=X[3].mean()

bn_2=Binarizer(threshold=mean_age)

X[3]=bn_2.transform([X[2]])[0]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=56)
clf=DecisionTreeClassifier(max_depth=10,min_samples_split=5,criterion='gini')

model=clf.fit(x_train,y_train)
print(accuracy_score(y_train,model.predict(x_train)))

print(accuracy_score(y_test,model.predict(x_test)))
