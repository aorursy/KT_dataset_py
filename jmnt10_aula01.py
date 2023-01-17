import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression 
pima_df = pd.read_csv("diabetes.csv")
pima_df.head(10)
#valores faltantes

print(pima_df[~pima_df.applymap(np.isreal).all(1)])

null_columns=pima_df.columns[pima_df.isnull().any()]

print(pima_df[pima_df.isnull().any(axis=1)][null_columns].head())
import seaborn as sns, numpy as np



pima_df.describe().T

ax = sns.distplot(pima_df['Glucose'])
#Piarplot Data Visualization, type this code and see the output

sns.pairplot(pima_df, diag_kind ='kde')
sns.pairplot(pima_df, diag_kind ='kde', hue="Outcome")
print("group by class:", pima_df.groupby(["Outcome"]).count())
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from sklearn import metrics
# select all rows and first 8 columns which are the #independent attributes

X = pima_df.iloc[:,0:7]

X.head()
y = pima_df.iloc[:,8]

y.head()
test_size = 0.30 

seed = 1 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
Logistic_model = LogisticRegression()

Logistic_model.fit(X_train, y_train)
y_predict= Logistic_model.predict(X_test)

print("Y predict/hat ", y_predict)
column_label = list(X_train.columns) # To label all the coefficient

model_Coeff = pd.DataFrame(Logistic_model.coef_, columns = column_label)

model_Coeff['intercept'] = Logistic_model.intercept_

print("Coefficient Values Of The Surface Are: ", model_Coeff)
print(metrics.confusion_matrix(y_test, y_predict))
pd.DataFrame({"Real": y_test, "Predito_modelo": y_predict, 

              "Acertou?": y_test==y_predict})