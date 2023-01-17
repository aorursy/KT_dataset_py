import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df=pd.read_csv("../input/security.csv")
df.shape
#Using Pandas get dummies function to convert the string columns into numerics

new_df=pd.get_dummies(df, columns=['DOMAIN_NAME','URL','WHOIS_COUNTRY','WHOIS_STATE_CITY','SERVER'])
pd.options.display.max_seq_items=100000
pd.options.display.max_seq_items
new_df['TIPO']= new_df['TIPO'].astype('category')
new_df['Target']=new_df['TIPO'].cat.codes
new_df.head()
target_df=new_df
target_df.columns
new_df=new_df.drop(['TIPO','CHARSET','CACHE_CONTROL','Target'], axis=1)

feature_cols=new_df.columns
x=new_df[feature_cols]
x.shape
#Benigna       0
#Maligna       1
y=target_df['Target']
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# print the size of the traning set:
print(x_train.shape)
print(y_train.shape)

# print the size of the testing set:
print(x_test.shape)
print(y_test.shape)
dec_tree=DecisionTreeClassifier() 
log_reg=LogisticRegression()
k=5
knn=KNeighborsClassifier(n_neighbors=k)
log_reg.fit(x_train,y_train)
dec_tree.fit(x_train,y_train)
knn.fit(x_train,y_train)
y_perdict_regression =log_reg.predict(x_test)
y_perdict_tree =dec_tree.predict(x_test)
y_predict_knn= knn.predict(x_test)
score_regression=accuracy_score(y_test,y_perdict_regression)
score_tree=accuracy_score(y_test, y_perdict_tree)
score_knn=accuracy_score(y_test, y_predict_knn)

print('Logistic Regression',score_regression)
print('Decision Tree',score_tree)
print('KNN',score_knn)
