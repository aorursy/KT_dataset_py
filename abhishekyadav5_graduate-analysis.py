import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from sklearn.svm import SVC



import warnings

warnings.filterwarnings('ignore')
input_data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

input_data.head()
input_data.info()
#converting all variables between 0 and 1, and "Chance of Admit" to 0 to 100



input_data["GRE Score"] = (input_data["GRE Score"]/340)



input_data["TOEFL Score"] = (input_data["TOEFL Score"]/120)



input_data["University Rating"] = (input_data["University Rating"]/5)



input_data["SOP"] = (input_data["SOP"]/5)



input_data["LOR "] = (input_data["LOR "]/5)



input_data["CGPA"] = (input_data["CGPA"]/10)



input_data["Chance of Admit "] = (input_data["Chance of Admit "]*100).astype(int)
input_data.head()
train_data, test_data = train_test_split(input_data, test_size=0.08)
train_data.info()
f, ax = plt.subplots(nrows=4, ncols=2, figsize=[12,12])

sns.scatterplot(y = "GRE Score", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[0,0])

sns.scatterplot(y = "TOEFL Score", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[0,1])

sns.scatterplot(y = "University Rating", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[1,0])

sns.scatterplot(y = "SOP", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[1,1])

sns.scatterplot(y = "LOR ", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[2,0])

sns.scatterplot(y = "CGPA", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[2,1])

sns.scatterplot(y = "Research", x = "Chance of Admit ", data = train_data, color = "Red", ax = ax[3,0])

train_data.corr(method="pearson")
train_data.corr(method="spearman")
train_data.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
X_train = train_data.drop("Chance of Admit ", axis=1)

Y_train = train_data["Chance of Admit "]
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
#Logistic Regression



LR = LogisticRegression()

LR.fit(X_train, Y_train)
LR.predict(x_test)
print(y_test)
accu = LR.score(x_test, y_test)

accu
#Linear Regression



LinR = LinearRegression()

LinR.fit(X_train, Y_train)
LinR.predict(x_test)
LinR.score(x_test, y_test)
#SVM



svc = SVC(kernel = "rbf")

svc.fit(X_train, Y_train)
svc.predict(x_test)
svc.score(x_test, y_test)
#Lasso Regression



Lasso = Lasso()

Lasso.fit(X_train, Y_train)
Lasso.predict(x_test)
Lasso.score(x_test, y_test)
#Poly Regression with degree 4



poly = PolynomialFeatures(degree=4)

X_poly = poly.fit_transform(X_train)



lin2 = LinearRegression()

lin2.fit(X_poly, Y_train)
lin2.predict(poly.fit_transform(x_test))
lin2.score(poly.fit_transform(x_test), y_test)
# poly regression with degree 2



poly1 = PolynomialFeatures(degree=2)

X_poly1 = poly1.fit_transform(X_train)



lin3 = LinearRegression()

lin3.fit(X_poly1, Y_train)
lin3.predict(poly1.fit_transform(x_test))
lin3.score(poly1.fit_transform(x_test), y_test)
#Ridge Regression



Rid = Ridge()

Rid.fit(X_train, Y_train)
Rid.predict(x_test)
accu = Rid.score(x_test, y_test)

accu
#Elastic Net Regression



Enet = ElasticNet()

Enet.fit(X_train, Y_train)
Enet.predict(x_test)
accu = Enet.score(x_test, y_test)

accu