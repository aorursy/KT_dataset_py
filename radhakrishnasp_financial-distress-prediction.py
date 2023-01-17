import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('../input/financial-distress/Financial Distress.csv')
data.info()
data.head()
df = data.copy()
def missing_values_table(df):

    total_missing = df.isnull().sum().sort_values(ascending=False)

    percentage_missing = (100*df.isnull().sum()/len(df)).sort_values(ascending=False)

    missing_table = pd.DataFrame({'missing values':total_missing,'% missing':percentage_missing})

    return missing_table
missing_values_table(df)
df.shape
# Create correlation matrix

corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features 

df.drop(to_drop, axis=1, inplace=True)
df.shape
Y = df.iloc[:,2].values

for y in range(0,len(Y)):

       if Y[y] > -0.5:

              Y[y] = 0

       else:

              Y[y] = 1

X = df.iloc[:,3:].values
print(df['Financial Distress'].value_counts())

df['Financial Distress'].value_counts().plot(kind='bar')
X = pd.DataFrame(X)

Y = pd.DataFrame(Y)
X.head()
Y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.30, random_state = 0)
print(f"Shape of X_train is :{X_train.shape},\nShape of X_test is :{X_test.shape},\nShape of y_train is :{y_train.shape},\nShape of y_test is :{y_test.shape}")
#Importing Evaluation metrics.

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
BNB.fit(X_train,y_train)
BNB_pred = BNB.predict(X_test)
accuracy_score(BNB_pred,y_test)
BNB_CM = pd.DataFrame(confusion_matrix(BNB_pred,y_test), index = ['Actual No','Actual Yes'], columns=['Predicted No','Predicted Yes'])
BNB_CM
print(classification_report(BNB_pred,y_test))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train,y_train)
LDA_pred = LDA.predict(X_test)
LDA_pred = pd.DataFrame(LDA_pred)
accuracy_score(LDA_pred,y_test)
LDA_CM = pd.DataFrame(confusion_matrix(LDA_pred,y_test), index = ['Actual No','Actual Yes'], columns=['Predicted No','Predicted Yes'])

LDA_CM
LDA_pred.head(60)
LDA_pred.tail(60)