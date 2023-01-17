# importing necessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# load the data
data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
data.info()
data.dtypes
data.duplicated().sum()                        #checking duplicates in data
data.columns
data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})      #Rename column
data.head(3)
data.isnull().sum()                                                          #total null
pd.DataFrame(round(data.isnull().sum()/data.shape[0] * 100,3), columns = ["Missing"])  # null in percentage
data.dropna(how='any',inplace = True)
pd.DataFrame(round(data.isnull().sum()/data.shape[0] * 100,3), columns = ["Missing"])  # null in percentage
data['rate'].unique()
data['rate']=data['rate'].apply(lambda x: x.replace('/5','').strip())                             #replacing commas  
data=data.loc[data['rate']!='NEW']
data['cost'].unique()                                                                           # getting unique values
data['cost']=data['cost'].apply(lambda x: x.replace(',','.').strip())                             #replacing commas  
data['cost']=data['cost'].astype('float')                                                        #converting to float
data['rate']=data['rate'].astype('float')                                                        #converting to float 
data['votes']=data['votes'].astype('int')                                                        #converting to integer
# Encode the input Variables
def Encode(data):
    for column in data.columns[~data.columns.isin(['rate', 'cost', 'votes'])]:
        data[column] = data[column].factorize()[0]        
    return data
zomato_data = Encode(data.copy())
zomato_data.corr()
#Get Correlation between different variables
corr = zomato_data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
zomato_data.columns
from sklearn.model_selection import train_test_split
x = zomato_data.iloc[:,[2,3,5,6,7,8,9,11]]
y = zomato_data['rate']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)
x_train.head()
y_train.head()
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test.head(1))
print(y_pred)
y_predict=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_pred=DTree.predict(x_test.head(1))
print(y_pred)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#Preparing Random Forest REgression
from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=500,random_state=42,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_pred=RForest.predict(x_test.head(1))
print('test value:',x_test['rate'][0])
print('predicted value:',y_pred)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ETree=ExtraTreesRegressor(n_estimators = 100)
ETree.fit(x_train,y_train)
y_pred=ETree.predict(x_test.head(1))
print(y_pred)
y_predict=ETree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#importing Random Forest Classifier from sklearn.ensemble
from sklearn.model_selection import GridSearchCV #GridSearchCV is for parameter tuning
from sklearn.ensemble import RandomForestRegressor
cls=RandomForestRegressor()
n_estimators=[25,50,75,100,125,150,175,200] #number of decision trees in the forest, default = 100
criterion=['mse'] #criteria for choosing nodes default = 'gini'
max_depth=[3,5,10] #maximum number of nodes in a tree default = None (it will go till all possible nodes)
parameters={'n_estimators': n_estimators,'criterion':criterion,'max_depth':max_depth} #this will undergo 8*2*3 = 48 iterations
RFC_cls = GridSearchCV(cls, parameters)
RFC_cls.fit(x_train,y_train)
y_pred=RFC_cls.predict(x_test.head(1))
print('test value:',x_test['rate'][0])
print('predicted value:',y_pred)
y_predict=RFC_cls.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
