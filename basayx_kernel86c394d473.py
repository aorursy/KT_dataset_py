import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/50_Startups.csv", sep = ",")
df.head(5)
df
df.shape
df.info() 
corr = df.corr()
corr
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values
           )
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df);
plt.hist(df['R&D Spend'])
plt.hist(df['Profit'])
df.median()
df.mean()
df.std()
df.cov()
df["State"].unique()
pd.get_dummies(df['State']).mean()
"""New York ve California bölgelerinde eşit sayıda startup var. Bunun yanında Florida bölgesinde bu iki bölgeye kıyasla daha az
startup bulunuyor."""
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')
dfDummies
df.drop(["State", "State_California"], axis = 1, inplace = True)
df = pd.concat([df, dfDummies], axis=1)
df.head()
y = df['Profit']
X = df.drop(['Profit'], axis=1)
y
X
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 42)
X_train
X_test
y_train
y_test

from sklearn.feature_selection import RFE
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = model.predict(X_test)
df2 = pd.DataFrame({'Gercek': y_test, 'Tahmini': y_pred})
df2
MAE = mean_absolute_error(y_test, y_pred)
MAE
MSE = mean_squared_error(y_test, y_pred)
MSE

RMSE = math.sqrt(MSE)
RMSE
model.score(X_test, y_test)
model.score(X_train, y_train)




