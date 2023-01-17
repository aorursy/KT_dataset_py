import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
hr=pd.read_csv("../input/HR data.csv")
hr.head()
hr.isnull().sum()
hr.fillna(0,inplace=True)
hr.isnull().sum()
hr.drop(['EmployeeCount','EmployeeID','StandardHours',"Age"],axis=1, inplace=True)
catd=hr.select_dtypes(include=np.object)
catd.head(10)
catd.BusinessTravel.unique()
ndata=hr.select_dtypes(include=np.number)
ndata.head(4)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
#scaling down the numeric variables
df_numcols=pd.DataFrame(min_max.fit_transform(ndata.iloc[:,0:50]),columns=ndata.iloc[:,0:50].columns.tolist())
df_numcols.head()
# applying label encoding on categorical data as it seems ordinal data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded = catd.apply(le.fit_transform)
df_encoded.head()
# Concating Both the data

final=pd.concat([df_numcols,df_encoded],axis=1)
# spliting the data in to s_y ,s_x i.e. dependent and independent varibale
s_y = final['Attrition']
s_x = final.drop('Attrition', axis = 1)
# Spliting the data
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(s_x,s_y, test_size = 0.20, random_state=42)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#Accuracy : 0.8356
# Above accuracy based on minmax and label encoding technique.
# now will try to improve the accuracy by applying the standard scaler technique on X_train and X_test data.
from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
x_train = Scaler_X.fit_transform(X_train)
x_test = Scaler_X.fit_transform(X_test)
x_train
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
# Here we can see very slight improvment in accuracy when used standard scaler tecnique - 0.836
# will make model by using feature selection technique i.e Kbest,chi2 test.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(s_x,s_y)
feat_importances = pd.Series(fit.scores_, index=s_x.columns)
topFatures = feat_importances.nlargest(5).copy().index.values
print("TOP 10 Features (Best to worst) :\n")
print(topFatures)
select_features=SelectKBest(chi2,k=10).fit(s_x,s_y)
select_features_df=pd.DataFrame({'features':list(s_x.columns),'scores':select_features.scores_})
select_features_df.sort_values(by="scores",ascending=False)
# keeping only top 5 best feature or varibale whose impact is highest on attrition.
chi2_selector = SelectKBest(chi2, k=5)
X_kbest = chi2_selector.fit_transform(s_x,s_y)
X_clf_new=SelectKBest(score_func=chi2,k=5).fit_transform(s_x,s_y)
print(X_clf_new[:5])
e_dataframe = pd.DataFrame(X_clf_new[:5]) 
e_dataframe.head()
columns=pd.DataFrame(s_x.columns)
columns.head()
score= pd.Series([e_dataframe,columns])
score.head(5)
print(s_x.head())
x_train_chi=select_features.transform(X_train)
x_test_chi=select_features.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train_chi,y_train)
y_pred_chi = lr.predict(x_test_chi)
print(accuracy_score(y_test,y_pred_chi))
print(confusion_matrix(y_test,y_pred_chi))
# Accuracy by applying best feature selection technique i.e. 0.8401
# Accuracy based on minmax and label encoding technique.- 0.8356
#  Accuracy when used standard scaler tecnique - 0.8367