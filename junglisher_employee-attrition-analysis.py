import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score
cols = pd.read_excel(r'../input/employees-attrition-analysis/data_dictionary.xlsx')
cols
df = pd.read_csv(r'../input/employees-attrition-analysis/whole data.csv')
df
df.shape
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df = df.dropna()
df.isnull().sum()
df.describe()
df.info()
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

for i in df.columns:
    if isinstance(df[i][0],str):
        df[i] = encoder.fit_transform(df[i])
df
df.Attrition.value_counts()
X = df.drop(['Attrition'], axis=1)
y =df.Attrition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state = 4589)
from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
x_train = Scaler_X.fit_transform(x_train)
x_test = Scaler_X.transform(x_test)
lr.fit(x_train, y_train)
lr.score(x_train, y_train)
pred = lr.predict(x_test)
accuracy_score(y_test, pred)
f1_score(y_test, pred)
recall_score(y_test,pred)
df.reset_index(inplace=True)
li = list(df[df.Attrition == 0].sample(n=2910).index)
df = df.drop(df.index[li])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state = 489)
from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
x_train = Scaler_X.fit_transform(x_train)
x_test = Scaler_X.transform(x_test)
lr.fit(x_train, y_train)
lr.score(x_train, y_train)
y_pred = lr.predict(x_test)
print(metrics.confusion_matrix(y_test, y_pred))
lr.score(x_test, y_test)
accuracy_score(y_test, y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)

from imblearn.over_sampling import SMOTE
Scaler_X = StandardScaler()
scaled_X = Scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, sampling_strategy='auto')
X_train, y_train = sm.fit_sample(X_train, y_train)
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
smote_pred = lr.predict(X_test)
accuracy_score(y_test, smote_pred)
f1_score(y_test, smote_pred)
recall_score(y_test, smote_pred)
feature_names = X.columns.values
summary_table = pd.DataFrame(columns = ['Feature_names'], data = feature_names)
summary_table['coeff']= np.transpose(lr.coef_)
summary_table

summary_table.index = summary_table.index +1
summary_table.iloc[0]= ['Intercept', lr.intercept_[0]]

summary_table.sort_index()
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(40,40))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
print(df.Over18.value_counts())
print(df.StandardHours.value_counts())
print(df.EmployeeCount.value_counts())
#Dropping them as they are not relevant
df.drop(['StandardHours','EmployeeCount','EmployeeID','Over18'], inplace=True, axis=1)
X = df.drop(['Attrition'], axis=1)
y =df.Attrition
#Calculating VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif=add_constant(X)

pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)  
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X.drop(['JobInvolvement','Age','BusinessTravel','PerformanceRating','YearsAtCompany','DistanceFromHome', 'StockOptionLevel'], inplace=True, axis=1)
X.drop(['Education','Gender','JobRole','Department'],inplace =True, axis =1)
Scaler_X = StandardScaler()
scaled_X = Scaler_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, sampling_strategy='auto')
X_train, y_train = sm.fit_sample(X_train, y_train)
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = lr.predict(X_test)

# Checking accuracy
accuracy_score(y_test, smote_pred) 
f1_score(y_test, smote_pred)
recall_score(y_test, smote_pred)
## recall and F1 increased significantly
feature_names = X.columns.values
summary_table = pd.DataFrame(columns = ['Feature_names'], data = feature_names)
summary_table['coeff']= np.transpose(lr.coef_)
summary_table

summary_table.index = summary_table.index +1
summary_table.iloc[0]= ['Intercept', lr.intercept_[0]]




summary_table.sort_index()