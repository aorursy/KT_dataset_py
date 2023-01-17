#Importing the essential libraries 
import pandas as pd
import numpy as np

#Importing the graph plotting libraries  
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
pwd
#Loading the data into data frame

df = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
df.head()
df.shape
df.columns
df['online_order'].unique()
df['book_table'].unique()
len(df['location'].unique())
len(df['rest_type'].unique())
df['approx_cost(for two people)'].unique()
len(df['approx_cost(for two people)'].unique())
df['listed_in(type)'].unique()
len(df['listed_in(type)'].unique())
df['listed_in(city)'].unique()
len(df['listed_in(city)'].unique())
df.info()
#Droping the features that are not required to build our model

df1 = df.drop(['url','address','phone','location','dish_liked','cuisines','reviews_list','menu_item'],axis = 'columns')
df1.head()
#Checking the Null Values

df1.isnull().sum()
#Dropping the Null Values

df2 = df1.dropna() 
df2.isnull().sum()
#Assiging lables to the categorical variable to Online Order attribute 

def conv(conv):
    if conv == 'No':
        return 0
    else:
        return 1

df2['online_order'] = df2['online_order'].map(conv)
df2.head()
#Ploting the online orders attributes value count

plt.figure(figsize=(10,5))
df['online_order'].value_counts().plot.bar()
plt.title('Online orders', fontsize = 20)
plt.ylabel('Frequency',size = 15)
#Assiging lables to the categorical variables to Table Booking attribute

def conv(conv):
    if conv == 'No':
        return 0
    else:
        return 1

df2['book_table'] = df2['book_table'].map(conv)
df2.head()
#Ploting the booking table attributes value count

plt.figure(figsize=(10,5))
df['book_table'].value_counts().plot.bar()
plt.title('Booking Table', fontsize = 20,pad=15)
plt.ylabel('Frequency', fontsize = 15)
df2['rate'].unique()
#Replacing the NaN values 

df2['rate'] = df2['rate'].replace('NEW',np.NaN)
df2['rate'] = df2['rate'].replace('-',np.NaN)
from scipy.stats import norm
from scipy import stats
#Converting rating values into the float values from string 

df2.rate=df2.rate.astype(str)
df2.rate=df2.rate.apply(lambda x : x.replace('/5','')).astype(float)
df2.head()
df2['rate'] = df2['rate'].fillna(df2['rate'].mean())
df2.isnull().sum()
#histogram of Price

plt.figure(figsize=(10,5))
sns.distplot(df2['rate'],color='b');
plt.title("Rating Distrubition")
plt.show()
#Renaming the Columns Names

df2=df2.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                         'listed_in(city)':'Location_in_City'})
df2.cost.dtype
#Converting the cost value to float from string

df2.cost=df2.cost.astype(str)
df2.cost=df2.cost.apply(lambda y : y.replace(',','')).astype(float)
df2.head()
#Applying One hot encoding on the Rest Type attribute 

dummiesresttype = pd.get_dummies(df2.rest_type)
dummiesresttype.head(3)
#Applying One hot encoding on the Type attribute

dummiestype = pd.get_dummies(df2.type)
dummiestype.head(3)
#Applying One hot encoding on the Location in city attribute

dummiescity = pd.get_dummies(df2.Location_in_City)
dummiescity.head(3)
#Concating them into one data frame

df3 = pd.concat([df2,dummiesresttype,dummiestype,dummiescity],axis="columns")
df3.head()
dfname = df3["name"]
dfname.head()
dflocation = df3["Location_in_City"]
dflocation.head()
#Dropping the One unwanted columns

df4 = df3.drop(["name","rest_type","type","Location_in_City"],axis = "columns")
df4.head()
df4.head()
corr = df4.corr()
corr
#Plotting correlation matrix without dummies 

corr = df2.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
X = df4.drop(['rate'],axis='columns')
X.head(10)
X.shape
y = df4.rate
y.head(10)
len(y)
#Spliting data into test and train

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_jobs=-1)
knn.fit(X_train,y_train)

Y_knn_pred = knn.predict(X_test)

Y_knn_x_pred = knn.predict(X_train)
r2 = r2_score(y_test,Y_knn_pred)
print('R-Square Score: ',r2*100)
print("RMSE value of Training dataset:" + np.sqrt(metrics.mean_squared_error(y_train,Y_knn_x_pred)).astype(str))
print("RMSE value of testing dataset:" + np.sqrt(metrics.mean_squared_error(y_test,Y_knn_pred)).astype(str))
acc = knn.score(X_train,y_train)
print('Accuracy: ',acc*100)
from sklearn.model_selection import cross_val_score
cv_res = cross_val_score(knn, X_train, y_train, cv=4, scoring="r2")
print(cv_res.mean())
cv_res
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

Y_lr_pred = lr.predict(X_test)

Y_lr_x_pred = knn.predict(X_train)
r2 = r2_score(y_test,Y_lr_pred)
print('R-Square Score: ',r2*100)
print("RMSE value of Training dataset:" + np.sqrt(metrics.mean_squared_error(y_train,Y_lr_x_pred)).astype(str))
print("RMSE value of testing dataset:" + np.sqrt(metrics.mean_squared_error(y_test,Y_lr_pred)).astype(str))
acc = lr.score(X_train,y_train)
print('Accuracy: ',acc*100)
from sklearn.model_selection import cross_val_score
cv_res = cross_val_score(lr, X_train, y_train, cv=4, scoring="r2")
print(cv_res.mean())
#plotting the KNN values predicated Rating

plt.figure(figsize=(12,7))
# preds_rf = knn.predict(X_test)
plt.scatter(y_test,X_test.iloc[:,2],color="blue")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Rating',size = 15)
plt.ylabel('Frequency',size = 15)
plt.scatter(Y_knn_pred,X_test.iloc[:,2],color="yellow")
type(Y_knn_pred)
dfrating = pd.Series(Y_knn_pred)
dfrating.head()
dfreco = pd.concat([dfname,dfrating,dflocation],axis="columns")
dfreco.head()
dfreco = dfreco.rename(columns={0:'Predicated_Rating'})
dfreco.head()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import cufflinks as cf
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
cf.go_offline()
rest = pd.pivot_table(data = dfreco, index = 'name', values = 'Predicated_Rating' , aggfunc = np.sum).reset_index()
rest = rest.sort_values(by = 'Predicated_Rating', ascending = False).reset_index(drop=True)
rest.head(5).iplot(kind = 'pie', labels= 'name', values= 'Predicated_Rating', title = 'Top Resturent with most Ratings' )

rat = dfreco[['Location_in_City','Predicated_Rating']]
rat = rat.groupby('Location_in_City').sum().sort_values('Predicated_Rating',ascending = False)
rat.head(10).iplot(kind = 'bar', title = ' Top 10 loaction with most number of Rating ')
