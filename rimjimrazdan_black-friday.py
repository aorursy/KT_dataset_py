import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
test = pd.read_csv("../input/black-friday/test.csv")
sales = pd.read_csv("../input/black-friday/train.csv")
sales.head()
test.head()
#for submission

submission = pd.DataFrame()
submission['Purchase'] = []
submission['User_ID'] = test['User_ID']
submission['Product_ID'] = test['Product_ID']
sales.shape
sales.info()
sales.describe()
sales.User_ID.nunique()
sales.Product_ID.nunique()
sales.Gender.value_counts(normalize=True)*100
sales.Age.value_counts()
sales.Occupation.nunique()
sales.City_Category.value_counts()
sales.Stay_In_Current_City_Years.value_counts()
sales.Marital_Status.value_counts()
# lets combine the data for data prep

test['Purchase']=np.nan
sales['data']='train'
test['data']='test'
test=test[sales.columns]
combined=pd.concat([sales,test],axis=0)
combined.head()
sales.isna().sum().sort_values(ascending=False)
#percent of missing data relevant to all data
percent = (sales.isnull().sum()/sales.isnull().count()).sort_values(ascending=False)
percent[[0,1]]
combined.drop('Product_Category_3',axis=1,inplace=True)
combined.Product_Category_2.value_counts()
#imputed missing values with random values in the same probability distribution as given feature already had

vc = combined.Product_Category_2.value_counts(normalize = True)
miss = combined.Product_Category_2.isna()
combined.loc[miss, 'Product_Category_2'] = np.random.choice(vc.index, size = miss.sum(), p = vc.values)
combined.Product_Category_2.value_counts()
combined.isna().sum()
#using the train data part from combined dataset for eda

sales_1 = combined[combined['data']=='train']
sns.countplot(sales_1['Gender'])
plt.show()
sns.countplot(sales_1['Age'])
plt.show()
sns.countplot(sales_1['Occupation'])
plt.show()
sns.countplot(sales_1['City_Category'])
plt.show()
sns.countplot(sales_1['Stay_In_Current_City_Years'])
plt.show()
sns.countplot(sales_1['Marital_Status'])
plt.show()
# Avearge amount spend by different age groups

data = sales_1.groupby('Age')['Purchase'].mean()
plt.plot(data.index,data.values,marker='o',color='g')
plt.xlabel('Age group');
plt.ylabel('Average_Purchase amount in $');
plt.title('Age group vs average amount spent');
plt.show()
# Avearge amount spend based on the time of stay in the current city

data = sales_1.groupby('Stay_In_Current_City_Years')['Purchase'].mean()
plt.plot(data.index,data.values,marker='o',color='y')
plt.xlabel('Stay_In_Current_City_Years');
plt.ylabel('Average_Purchase amount in $');
plt.title('Stay_In_Current_City_Years vs average amount spent');
plt.show()
# Avearge purchase based on Marital_Status

data = sales_1.groupby('Marital_Status')['Purchase'].mean()
plt.bar(data.index,data.values)
plt.xlabel('Marital_Status');
plt.ylabel('Average_Purchase amount in $');
plt.title('Avearge purchase based on Marital_Status');
plt.show()
# Top 10 products which made the highest sales

data = sales_1.groupby("Product_ID").sum()['Purchase']

plt.figure(figsize=(10,5))
data.sort_values(ascending=False)[0:10].plot(kind='bar')
plt.xticks(rotation=90)
plt.xlabel('Product ID')
plt.ylabel('Total amount purchased in Million $')
plt.title('Top 10 Products with highest sales')
plt.show()
#comparing based on Marital_Status and Gender

sns.countplot(x='Marital_Status',data=sales_1,hue='Gender')
plt.title('Comparing based on Marital_Status and Gender')
plt.show()
a =pd.crosstab(sales_1['Age'],sales_1['Product_ID'])
a.idxmax(axis=1)
#Occupations and City Category

plt.figure(figsize=(15,5))
sns.countplot(x='Occupation',data=sales_1,hue='City_Category')
plt.title('Comparing Occupations and City Category')
plt.show()
#the purchase habits of different genders across the different city categories.

g = sns.FacetGrid(sales_1,col="City_Category")
g.map(sns.barplot, "Gender", "Purchase")
plt.show()
# for datapreprocessing again working with the combined dataset
combined.head()
# User_ID data preprocess. e.g. 1000002 -> 2

combined['User_ID'] = combined['User_ID'] - 1000000

# Product_ID preprocess e.g. P00069042 -> 69042

combined['Product_ID'] = combined['Product_ID'].str.replace('P00', '')

#object to int
combined['Product_ID'] = pd.to_numeric(combined['Product_ID'],errors='coerce')
combined.info()
combined.Product_Category_2 = combined.Product_Category_2.astype('int64')
# features with datatype object

cat_cols = combined.select_dtypes(['object']).columns
cat_cols
# 4+ to 4
combined['Stay_In_Current_City_Years'] =np.where(combined['Stay_In_Current_City_Years'].str[:2]=="4+",4,combined['Stay_In_Current_City_Years'])

#object to int
combined['Stay_In_Current_City_Years'] = pd.to_numeric(combined['Stay_In_Current_City_Years'],errors='coerce')
combined['Gender'] = combined['Gender'].map({'F':0, 'M':1}).astype(int)
# Modify age column

combined['Age'] = combined['Age'].map({'0-17': 9,
                               '18-25': 22,
                               '26-35': 31,
                               '36-45': 42,
                               '46-50': 48,
                               '51-55': 53,
                               '55+': 60})
combined['Age'].value_counts()
combined = pd.get_dummies(combined,columns=['City_Category'],drop_first = True)
combined.head()
combined.info()
combined.head()
#splitting the data back into train and test as it was already provided

sales = combined[combined['data']=='train']
del sales['data']
test_input = combined[combined['data']=='test']
test_input.drop(['Purchase','data'],axis=1,inplace=True)

del combined
#Heatmap to show the correlation between various variables of the train data set

plt.figure(figsize=(12, 5))
cor = sales.corr()
ax = sns.heatmap(cor,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
#splitting the data into X and y
X = sales.drop('Purchase',axis=1)
y = sales['Purchase']

#train test split for model building
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
#Linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train) # training the algorithm

# Getting the coefficients and intercept

print('coefficients:\n', lr.coef_)
print('\n intercept:', lr.intercept_)
#Predicting on the test data

y_pred = lr.predict(X_test)

from sklearn import metrics

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# Ridge Regression

from sklearn.linear_model import Ridge

RR = Ridge(alpha=0.05,normalize=True)
RR.fit(X_train, y_train)

y_pred = RR.predict(X_test)

print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# Decision Tree Model

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

DT.fit(X_train, y_train)

y_pred = DT.predict(X_test)

print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Decision Tree 2

DT2 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)

DT2.fit(X_train, y_train)

y_pred = DT2.predict(X_test)

print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Fitting the model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 3,max_depth=10,n_estimators=25)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# another random forest

from sklearn.ensemble import RandomForestRegressor

rf3 = RandomForestRegressor(random_state=3,max_depth=10,min_samples_split=500,oob_score=True)


rf3.fit(X_train,y_train)

y_pred = rf3.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# random forest 4

rf4 = RandomForestRegressor(n_estimators=30,random_state=3,max_depth=15,min_samples_split=100,oob_score=True)


rf4.fit(X_train,y_train)

y_pred = rf4.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Fitting the model
from sklearn.ensemble import ExtraTreesRegressor

rf = ExtraTreesRegressor()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#XGBoost Model1
from xgboost import XGBRegressor


xgb1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb1.fit(X_train,y_train)

y_pred = xgb1.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
## XGBoost2
from xgboost import XGBRegressor

xgb2 = XGBRegressor(n_estimators=500,max_depth=10,learning_rate=0.05)

xgb2.fit(X_train,y_train)

y_pred = xgb2.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
## XGBoost3

xgb3 = XGBRegressor(n_estimators=6,max_depth=500)

xgb3.fit(X_train,y_train)

y_pred = xgb3.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#XGBoost4

xgb4 = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb4.fit(X_train,y_train)

y_pred = xgb4.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#XGBoost5
from xgboost import XGBRegressor

xgb5 = XGBRegressor(n_estimators=450,max_depth=8,learning_rate=0.076)

xgb5.fit(X_train,y_train)

y_pred = xgb5.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#XGBoost6
from xgboost import XGBRegressor

xgb6 = XGBRegressor(n_estimators=470,max_depth=9,learning_rate=0.06)

xgb6.fit(X_train,y_train)

y_pred = xgb6.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from lightgbm import LGBMRegressor

lgbm1 = LGBMRegressor(n_estimators=500,max_depth=10,learning_rate=0.05)

lgbm1.fit(X_train,y_train)

y_pred = lgbm1.predict(X_test)

print('r2_score:', metrics.r2_score(y_test,y_pred)) 
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.head()
# Feature Importance

imp = pd.DataFrame(xgb2.feature_importances_,index=X.columns,columns=['importance'])
imp.sort_values(by='importance',ascending=False)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(df1.Predicted,df1.Actual)
plt.plot(y_pred,y_pred,'r')
plt.xlabel('y predicted')
plt.ylabel('y actual')
plt.show()
