import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/indian_liver_patient.csv')

df.head(1)
df.shape
df.info()
# what is dataset  - Target column 

# 0 - no issue, 1 - disease 

df['Dataset'].unique()
# loooking for statistical properties 

df.describe()
# entire dataset is numerical except gender 

# gender can be labelencoded 

from sklearn.preprocessing import LabelEncoder



# create an instance of the labelencoder

le = LabelEncoder()



# apply 

df['Gender'] = le.fit_transform(df['Gender'])
df.head()
# how many na values in one column

df['Albumin_and_Globulin_Ratio'].isna().sum() # 4 
df['Albumin_and_Globulin_Ratio'].hist()



# fillna with mean

df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace = True)
# corr matrix 

corr = df.corr()

sns.heatmap(corr)
X = df.drop(['Dataset'], axis = 1)

X.shape
Y = df['Dataset']

Y.shape
# apply logistic regression 



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(classification_report(y_pred, y_test))
# countplot 

sns.countplot(df['Dataset'])