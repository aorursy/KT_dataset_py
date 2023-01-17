import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
pd.set_option('display.max_columns',None)
data = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv',na_values=['?'])
data
data.info()
nulls = data.columns[data.isnull().all()]
nulls
(data == data.iloc[0]).all()
# categorical features
cat_features = [feat for feat in data.columns if data[feat].dtype == 'object']
cat_features
#discrete features
discrete_features = [feat for feat in data.columns if len(data[feat].unique())<10 and feat not in cat_features]
discrete_features
#continuous features
con_features = [feat for feat in data.columns if feat not in discrete_features and feat not in cat_features]
con_features
#filling categorical features with first most frequently occuring value
for feat in cat_features:
    data[feat].fillna(data[feat].mode()[0],inplace=True)
data.info()
for feat in con_features:
    plt.hist(data[feat])
    plt.title(feat)
    plt.show()
#filling continuous features with mean the the column 
for feat in con_features:
    data[feat].fillna(data[feat].mean(),inplace=True)
data.info()
for feat in cat_features:
    print(feat,'\n',data[feat].unique(),'\n')
data['make'] = data['make'].map(data['make'].value_counts().to_dict())
data['fuel-system'] = data['fuel-system'].map(data['fuel-system'].value_counts().to_dict())
data.head()
label = LabelEncoder()
data[['num-of-doors','num-of-cylinders']] = data[['num-of-doors','num-of-cylinders']].apply(label.fit_transform)
data
data = pd.get_dummies(data,drop_first=True)
data
# changing the distributon of continuous features to log normal distribution
for feat in con_features:
    data[feat] = np.log(data[feat]+1)
sns.heatmap(data.corr())
label = data['price']
data.drop('price',inplace=True,axis=1)
colnames=data.columns
scaler=StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.2)

clf=svm.SVR()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print('svm', acc)

clf=LinearRegression()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print('linear regression', acc)

clf=Ridge()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print('ridge regression', acc)




X_train = pd.DataFrame(X_train)
X_train.columns = colnames
X_test = pd.DataFrame(X_test)
X_test.columns = colnames
feature_sel_model = SelectFromModel(Lasso(alpha=0.006, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)
feature_sel_model.get_support()
# this is how we can make a list of the selected features
selected_feat = X_train.columns[feature_sel_model.get_support()]

X_train = X_train[selected_feat]
X_test = X_test[selected_feat]
print(X_train.shape)
X_train.head()
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
clf=svm.SVR()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print(acc)
clf=LinearRegression()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print(acc)
clf=Ridge()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print(acc)