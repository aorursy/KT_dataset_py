# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot
import seaborn as sns

sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
sns.set()
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
for col in data.columns:
    if not np.issubdtype(data[col].dtype, np.number):
        if len(data[col].unique()) < 11:
            _dat = pd.get_dummies(data[col], prefix=col).iloc[:,1:]
            data = pd.concat([data, _dat], 1)
            data = data.drop(col, 1)
        else:
            if "Charges" in col:
                data[col] = pd.to_numeric(data[col].replace(" ", 0))

Y = data["Churn_Yes"]
X = data.drop(["Churn_Yes", "customerID"], 1)
X.head()
from sklearn.cluster import KMeans

wcss = []
number = range(2,10)

for n in number:
    kmeans = KMeans(n)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
plot.plot(number, wcss, "-o")
plot.show()

kmeans = KMeans(4)
kmeans.fit(X)

clusters = kmeans.predict(X)
def PlotClusters(X, v1, v2, clusters):
    Xc = pd.concat([X, pd.Series(clusters).rename("cluster")],1)

    Xc = Xc[[v1, v2, "cluster"]]

    for i in range(np.max(clusters)+1):
        _Xc = Xc[Xc["cluster"]==i]
        plot.scatter(_Xc[v1], _Xc[v2])

    plot.xlabel(v1)
    plot.ylabel(v2)
plot.figure(figsize=(15,15))
plot.subplot(221)
PlotClusters(X, "tenure", "TotalCharges", clusters)
plot.subplot(222)
PlotClusters(X, "tenure", "MonthlyCharges", clusters)
plot.subplot(223)
PlotClusters(X, "MonthlyCharges", "TotalCharges", clusters)

plot.show()
import statsmodels.api as sm

treshold=0.05
X2 = sm.add_constant(X.drop("TotalCharges", 1))
pdroplist=["TotalCharges"]

while True:
    ols = sm.OLS(Y, X2).fit()
    
    if ols.pvalues.max() > treshold:
        col = ols.pvalues.argmax()
        print("Dropping "+str(col))
        X2 = X2.drop(col,1)
        pdroplist.append(col)
    else:
        break
ols = sm.OLS(Y, X2).fit()
print(ols.summary())
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, StandardScaler

X3=X2.drop("const",1)

imputer = Imputer()
X3 = imputer.fit_transform(X3)

scaler = StandardScaler()
X3 = scaler.fit_transform(X3)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X3, Y, test_size=0.2)
from sklearn.metrics import accuracy_score

reg = LogisticRegression()
reg.fit(X_train, Y_train)

acc = accuracy_score(Y_val, reg.predict(X_val))
print("Accuracy: "+str(acc))
xvals = X2.columns[1:]
coeffs =np.exp(reg.coef_[0])-1.0

plot.figure(figsize=(10,5))
plot.bar(xvals, coeffs)
plot.xticks(rotation=90)
plot.ylabel("Coefficients [a.u.]")
plot.show()
inet_customers = data[data["InternetService_No"]==0]
fo_data = inet_customers[["InternetService_Fiber optic", "Churn_Yes"]]
c00 = fo_data[(fo_data["InternetService_Fiber optic"]==0) & (fo_data["Churn_Yes"]==0)].shape[0]
c01 = fo_data[(fo_data["InternetService_Fiber optic"]==0) & (fo_data["Churn_Yes"]==1)].shape[0]
c10 = fo_data[(fo_data["InternetService_Fiber optic"]==1) & (fo_data["Churn_Yes"]==0)].shape[0]
c11 = fo_data[(fo_data["InternetService_Fiber optic"]==1) & (fo_data["Churn_Yes"]==1)].shape[0]
from scipy.stats import chi2_contingency

c_table = np.array([[c00, c01], [c10, c11]])
chi2, p, dof, expected = chi2_contingency(c_table)

print("P-Value: "+str(p))
leaving = reg.predict(X_val)
leaving = np.sum(leaving)/len(leaving)

print("Customers currently leaving:\t\t\t"+str(np.round(leaving*100,2))+"%")

reg2 = LogisticRegression()

reg2.coef_=np.copy(reg.coef_)
reg2.coef_[0,7]=0
#reg2.coef_[0,2]=reg2.coef_[0,2]*np.exp(0.25)
reg2.intercept_ = reg.intercept_
reg2.classes_ = reg.classes_

leaving_new = reg2.predict(X_val)
leaving_new = np.sum(leaving_new)/len(leaving_new)


print("Customers leaving with better fiber optics:\t"+str(np.round(leaving_new*100,2))+"%")
