import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from pandas_profiling import ProfileReport
df=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.isnull().sum()
ProfileReport(df)
plt.figure(figsize=[14,9])

sns.heatmap(df.corr(),annot=True)
sns.barplot("quality","alcohol",data=df)
sns.barplot("quality","volatile acidity",data=df)
sns.countplot(x="quality",data=df)
quality = df["quality"].values

category = []

for num in quality:

    if num>5:

        category.append("Good")

    else:

        category.append("Bad")
category = pd.DataFrame(data=category, columns=["category"])

data = pd.concat([df,category],axis=1)

data.drop(columns="quality",axis=1,inplace=True)
data.head()
sns.countplot(x="category",data=data)
#histogram of alcohol

plt.hist(x=df["alcohol"],bins=20)

plt.show()
#histogram of sulphates

plt.hist(x=df["sulphates"],bins=20)

plt.show()
#histogram of citric acid

plt.hist(x=df["citric acid"],bins=20)

plt.show()
#histogram of fixed acidity

plt.hist(x=df["fixed acidity"],bins=20)

plt.show()
#histogram of volatile acidity

plt.hist(x=df["volatile acidity"],bins=20)

plt.show()
#creating a list of numeric features in the database.

numeric_values=[x for x in df.columns if df[x].dtypes!='O']

numeric_values
#plotting boxplot of all the numeric values.

for feature in numeric_values:

    plt.figure(figsize=[8,5])

    sns.boxplot(df[feature],palette="spring_r")
sns.jointplot(x=df["alcohol"],y=df["density"],kind="hex")
plt.figure(figsize=[10,8])

sns.scatterplot("alcohol","density",hue="category",data=data)
plt.figure(figsize=[10,8])

sns.scatterplot("chlorides","sulphates",hue="category",data=data)
sns.barplot("category","citric acid",data=data)
sns.barplot("category","volatile acidity",data=data)
#declaring X and y variables where X are features and y is our target variable.

X= data.iloc[:,:-1].values

y=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder

label_quality = LabelEncoder()
#splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
#scaling data for optimization

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test, pred_knn))
print("The accuracy of this model is ",accuracy_score(y_test,pred_knn)*100," %")