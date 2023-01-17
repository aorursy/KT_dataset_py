import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data=pd.read_csv("googleplaystore.csv")
data.head(5)
data.shape
data.describe()
data.boxplot()
plt.show()
data.hist()
plt.show()
data.info()
data.isnull().sum()
data[data.Rating>5]
data.drop([10472],inplace=True)
data[10470:10475]
data.boxplot()
plt.show()
data.hist()
plt.show()
threshold= len(data)*0.1
threshold
data.dropna(thresh=threshold,axis=1,inplace=True)
print(data.isnull().sum())
def impute_median(series):
    return series.fillna(series.median())
data["Rating"]= data["Rating"].transform(impute_median)
data.isnull().sum()
# Now imputing the categorical values
data["Type"].fillna(str(data["Type"].mode().values[0]),inplace=True)
data["Current Ver"].fillna(str(data["Current Ver"].mode().values[0]), inplace=True)
data["Android Ver"].fillna(str(data["Android Ver"].mode().values[0]),inplace=True)
data.isnull().sum()
# Let's convert Price, Reviews and Installs into numerical values
data["Price"]= data["Price"].apply(lambda x:str(x).replace("$","") if "$" in str(x) else str(x))
data["Price"]= data["Price"].apply(lambda x: float(x))
data["Reviews"]= pd.to_numeric(data["Reviews"], errors="coerce")

data["Installs"]= data["Installs"].apply(lambda x: str(x).replace("+","") if "+" in str(x) else str(x))
data["Installs"]= data["Installs"].apply(lambda x: str(x).replace(",","") if "," in str(x) else str(x))
data["Installs"]=data["Installs"].apply(lambda x: float(x))
data["Rating"]= pd.to_numeric(data["Rating"],errors="coerce")
sns.distplot(data["Rating"],kde=True);
data.loc[data["Reviews"].idxmax()]
data.loc[data["Rating"].idxmax()]
data.loc[data["Installs"].idxmax()]
data.iloc[0]['App']

data["Category"].value_counts()[:10].sort_values(ascending=True).plot(kind="barh")
plt.show()
sns.boxplot(x="Content Rating",y="Rating",hue="Type",data=data)
plt.show()
sns.lmplot("Reviews","Rating",data=data,hue="Type",fit_reg=False,palette="Paired",scatter_kws={"marker":"D","s":100})
plt.show()
labels=["Free","Paid"]
d=[data["Type"].value_counts()[0],data["Type"].value_counts()[1]]
fig1,ax1=plt.subplots()
ax1.pie(d,labels=labels,shadow=True)
ax1.axis("equal")
plt.show()
X=data[["Reviews","Price"]]
y=data.Rating

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42)
# Lets bring the dataset features into same scale
scaler=StandardScaler()
X= scaler.fit_transform(X)
from sklearn.linear_model import LinearRegression
lin_r= LinearRegression()
model= lin_r.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
rating= model.predict(np.array([[1000,3]]))
print("Predicted rating is:",rating)
y=model.intercept_ +(1000*model.coef_[0]+2*model.coef_[1])
print("Rating is:",y)
pred= model.predict(X_test)
pred
from sklearn.metrics import mean_squared_error
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,pred))))