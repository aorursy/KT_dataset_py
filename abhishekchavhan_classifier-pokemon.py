# Boiler plate tools :
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## For modelling :
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Modelling tools :
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score,RandomizedSearchCV,GridSearchCV 
data = pd.read_csv('/kaggle/input/pokemon/pokemon.csv')
data.head()
# Let's find out whether there is missing data or not...
data.isna().sum()
## The data looks quiet missing,Lets fill it.But before lets check the main metrics...
fig,axes = plt.subplots(figsize=(10,10))
axes.bar(data['name'][:10],data['attack'][:10],color='salmon');
plt.title('Pokemon attack');
plt.xlabel('Pokemon');
plt.ylabel('Attack');

data.plot(kind='scatter',x='name',y='sp_attack');
# Lets Check the missing values...
for label,content in data.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
# Fill them..
for label,content in data.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            data[label] = content.fillna(content.median())
            
data.dtypes
for label,content in data.items():
    if pd.api.types.is_float_dtype(content):
        data[label] = data[label].astype('int')
data.dtypes
for label,content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
        data[label] = data[label].astype('category')
data.dtypes
for label,content in data.items():
    if pd.api.types.is_categorical_dtype(content):
        data[label] = pd.Categorical(content).codes + 1
X = data.drop('is_legendary',axis=1)
y = data['is_legendary']
model_a = RandomForestClassifier()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model_a.fit(X_train,y_train)
model_a.score(X_test,y_test)
model_b = GradientBoostingClassifier()
model_b.fit(X_train,y_train)
model_b.score(X_test,y_test)
model_c = LogisticRegression()
model_c.fit(X_train,y_train)
model_c.score(X_test,y_test)
# Lets check the cross val score
y_preds = model_a.predict_proba(X_test)
cvm = cross_val_score(model_a,X,y,cv=10)
np.mean(cvm)
# Classification metrics :
y_preds = model_a.predict(X_test)

precision = precision_score(y_test,y_preds)
recall = recall_score(y_test,y_preds)
accuracy = accuracy_score(y_test,y_preds)
accuracy,recall,precision
## Lets get the legendary predictions : 
Pokemon = pd.DataFrame()
y_preds = model_a.predict(X)
Pokemon['Default values'] = y
Pokemon['Predictions'] = y_preds
Pokemon
fig,axes = plt.subplots()
axes.stackplot(Pokemon['Default values'],Pokemon['Predictions'],color=['red','blue']);