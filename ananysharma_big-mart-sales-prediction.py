import pandas as pd
import numpy as np

train = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
test = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")
test1 = test.copy()
train.shape,test.shape
train.head()
test["Outlet_Size"].unique()
train.nunique()
test.nunique()
train.isna().sum()
map1 = {"Small":1,"Medium":2,"High":3}
train["Outlet_Size"] = train["Outlet_Size"].map(map1)
train["Item_Weight"] = train["Item_Weight"].fillna(train.Item_Weight.mean())
train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].median())
train.isna().sum()
map1 = {"Small":1,"Medium":2,"High":3}
test["Outlet_Size"] = test["Outlet_Size"].map(map1)
test["Item_Weight"] = test["Item_Weight"].fillna(test.Item_Weight.mean())
test["Outlet_Size"] = test["Outlet_Size"].fillna(test["Outlet_Size"].median())
train.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,10)
plt.hist(train["Item_Outlet_Sales"],bins = 100)
# plt.show()

# plt.rcParams['figure.figsize'] = (10,10)
plt.hist(train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()
plt.rcParams['figure.figsize'] = (5,5)
plt.hist(train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()
sns.countplot(train["Outlet_Location_Type"],palette = 'dark')
plt.show()
sns.countplot(train["Outlet_Size"],palette = 'dark')
plt.show()
sns.countplot(train["Outlet_Type"],palette = 'RdYlGn')
plt.xticks(rotation = 90)
plt.show()
sns.violinplot(x=train["Outlet_Size"],y=train["Item_Outlet_Sales"],hue = train["Outlet_Size"],palette = "Reds")
plt.legend()
plt.show()

train.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
test.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X = pd.get_dummies(train[feat])
train = pd.concat([train,X],axis=1)
train.head()
feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X1 = pd.get_dummies(test[feat])
test = pd.concat([test,X1],axis=1)
train.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)
test.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)
X_train = train.drop(labels = ["Item_Outlet_Sales"],axis=1)
y_train = train["Item_Outlet_Sales"]
X_train.shape,y_train.shape
train.head()
y_train.head()
from sklearn import preprocessing

x = X_train.values #returns a numpy array
test_s = test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
x_scaled_test = min_max_scaler.fit_transform(test_s)
df_train = pd.DataFrame(x_scaled_train)
df_test = pd.DataFrame(x_scaled_test)
df_train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.4)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
preds = model.predict(X_test)

plt.scatter(y_test, preds)
plt.show()
sns.distplot((y_test-preds),bins=50)
plt.show()

from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, preds))
print('MSE:', metrics.mean_squared_error(y_test, preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
predictions = model.predict(df_test)
final = pd.DataFrame({"Item_Identifier":test1["Item_Identifier"],"Outlet_Identifier":test1["Outlet_Identifier"],"Item_Outlet_Sales":abs(predictions)})
final.head()
final.to_csv('Submiss1.csv',index=False,header=True)


