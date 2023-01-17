import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor


labelencoder = LabelEncoder()
%matplotlib inline

from sklearn.datasets import load_boston
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_data.head()
cols = train_data.columns[:40] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(train_data[cols].isnull(), cmap=sns.color_palette(colours))
cols = train_data.columns[40:80] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(train_data[cols].isnull(), cmap=sns.color_palette(colours))
cols = train_data.columns[:50]
plt.figure(figsize = (16,5))
sns.heatmap(train_data[cols].corr(), annot = True, cmap= 'coolwarm', fmt='.2g')
data = [train_data['Id'],train_data['SalePrice'], train_data['LotArea'], train_data['LotFrontage'],train_data['SaleType'],train_data['Condition2'], train_data['TotalBsmtSF'], train_data['1stFlrSF']]
headers = ["Id","SalePrice","LotArea","LotFrontage","SaleType","Condition", "TotalBasementSF","1stFlrSF"]
train = pd.concat(data, axis=1, keys=headers)
train.head()
train = dummyEncode(train)
train.head()
cols = test_data.columns[0:40] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(test_data[cols].isnull(), cmap=sns.color_palette(colours))
data = [train_data['Id'],test_data['LotArea'], test_data['LotFrontage'],test_data['SaleType'],test_data['Condition2'], test_data['TotalBsmtSF'], test_data['1stFlrSF']]
headers = ["Id","LotArea","LotFrontage","SaleType","Condition", "TotalBasementSF","1stFlrSF"]
test = pd.concat(data, axis=1, keys=headers)
test.head()
test = dummyEncode(train)
test.head()
med_train = train['LotFrontage'].median()
print(med_train)
train['LotFrontage'] = train['LotFrontage'].fillna(med_train)

train['PrecFrontage'] = train['LotFrontage']/train['LotArea'] * 100

med_test = test['LotFrontage'].median()
print(med_test)
test['LotFrontage'] = test['LotFrontage'].fillna(med_test)

test['PrecFrontage'] = test['LotFrontage'] / test['LotArea'] * 100

train.head()
palette = sns.color_palette("bright", 8)
sns.relplot(x="SaleType", y="LotArea", hue="Condition", data=train, legend = 'full', palette=palette)


ax1 = train.plot.scatter(x='LotArea',
                      y='SalePrice',
                      c='DarkBlue')

#palette = sns.color_palette("bright",7)
sns.relplot(x="TotalBasementSF", y="1stFlrSF", hue="SalePrice", data=train)
dataset = pd.get_dummies(train, columns = ["TotalBasementSF", "1stFlrSF", "LotArea", "SalePrice"])

#train_final = dataset[:len(train_data)]
#test_final = dataset[len(test_data):]

y=train['SalePrice']
X=train.drop('SalePrice', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 473, random_state = 2)
model_1 = RandomForestClassifier(n_estimators=100)
model_1.fit(X_train, y_train)

predict1 = model_1.predict(X_val)
acuracy1 = accuracy_score(predict1, y_val)
print('Accuracy: ', acuracy1)
#model_2 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
#model_2.fit(X_train, y_train)

#predict2 = model_2.predict(X_val)
#acuracy2 = accuracy_score(predict2, y_val)
#print('Accuracy: ', acuracy2)
model_3 = LogisticRegression(random_state=0)
model_3.fit(X_train, y_train)

predict3 = model_3.predict(X_val)
acuracy3 = accuracy_score(predict3, y_val)
print('Accuracy: ', acuracy3)
model_4 =  DecisionTreeClassifier()
model_4.fit(X_train, y_train)

predict4 = model_4.predict(X_val)
acuracy4 = accuracy_score(predict4, y_val)
print('Accuracy: ', acuracy4)
df = pd.DataFrame({'Random Forest': acuracy1, 'Logistic': acuracy3, ' Decision Tree': acuracy4} , index=[0])
df.rename(index={0:'Accuracy'}, inplace=True)
df
y_train = train['SalePrice']
X_train = train[['Id','LotArea', 'TotalBasementSF', '1stFlrSF']]

X_test = test[['Id','LotArea', 'TotalBasementSF', '1stFlrSF']]
selected_columns = X_train[['Id','LotArea',  'TotalBasementSF', '1stFlrSF']]
df1 = selected_columns.copy()

y_train = y_train.reindex(X_test.index)
X_train = X_train.reindex(X_test.index)
df1 = X_test.reindex(X_test.index)

final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(df1, y_train)

newId = test.Id +1460

final_predictions = final_model.predict(X_test)
output = pd.DataFrame({'Id': newId, 'SalePrice': final_predictions})
final_accuracy = accuracy_score(final_predictions, y_train)
print('Accuracy: ', final_accuracy)

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
output