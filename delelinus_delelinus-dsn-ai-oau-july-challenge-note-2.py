%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
file_path='train.csv'
df=pd.read_csv(file_path)
df.head()
path='test.csv'
df2=pd.read_csv(path)
df2.head()
# df2.isnull().sum()
df.describe(include='all')
# group 'Product_Identifier', 'Product_Weight' by 'Product_Identifier' so as to replace nan with the mean
#of the corresponding 'Product_Identifier'
df_gptest = df[['Product_Identifier', 'Product_Weight']]
grouped_test1 = df_gptest.groupby(['Product_Identifier'],as_index=False).mean() 
# df.isnull().sum()

# df2.isnull().sum()

#check the mode of the supermarket size values
# df['Supermarket _Size'].value_counts().idxmax()

# df2['Supermarket _Size'].value_counts().idxmax()

#replace the missing 'Supermarket_Size' values by the most frequent 
df["Supermarket _Size"].replace(np.nan, "Medium", inplace=True)
df2["Supermarket _Size"].replace(np.nan, "Medium", inplace=True)
df.Product_Weight=df.sort_values(['Product_Identifier','Product_Weight']).Product_Weight.ffill( )
df2.Product_Weight=df.sort_values(['Product_Identifier','Product_Weight']).Product_Weight.ffill( )
df.sort_values(['Product_Identifier','Supermarket_Identifier']).head()
df.dtypes
# grouped_1=df[['Product_Type','Supermarket_Identifier','Product_Supermarket_Sales']]
# grpd_1=grouped_1.groupby(['Product_Type','Supermarket_Identifier'],as_index=False).mean()
# grpd_1=grpd_1.sort_values(['Product_Type','Product_Supermarket_Sales'], ascending=False)
# grpd_1
# grouped_11=df[['Product_Type','Product_Supermarket_Sales']]
# grpd_11=grouped_11.groupby(['Product_Type'],as_index=False).sum()
# grpd_11=grpd_11.sort_values(['Product_Supermarket_Sales'], ascending=False)
# grpd_11
# grouped_pivot = grpd_1.pivot(index='Product_Type',columns='Supermarket_Identifier')
# grouped_pivot.head()
# fig, ax = plt.subplots()
# im = ax.pcolor(grouped_pivot, cmap='RdBu')

# #label names
# row_labels = grouped_pivot.columns.levels[1]
# col_labels = grouped_pivot.index

# #move ticks and labels to the center
# ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# #insert labels
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(col_labels, minor=False)

# #rotate label if too long
# plt.xticks(rotation=90)

# fig.colorbar(im)
# plt.show()
grouped_2=df[['Supermarket_Type', 'Product_Supermarket_Sales']]
grpd_2=grouped_2.groupby(['Supermarket_Type'],as_index=False).mean()
grpd_2.sort_values('Product_Supermarket_Sales',ascending=False)
df['Supermarket_Type'].replace('Supermarket Type3',4, inplace=True)
df['Supermarket_Type'].replace('Supermarket Type1',3, inplace=True)
df['Supermarket_Type'].replace('Supermarket Type2',2, inplace=True)
df['Supermarket_Type'].replace('Grocery Store',1, inplace=True)
# df[['Supermarket_Type']]

df2['Supermarket_Type'].replace('Supermarket Type3',4, inplace=True)
df2['Supermarket_Type'].replace('Supermarket Type1',3, inplace=True)
df2['Supermarket_Type'].replace('Supermarket Type2',2, inplace=True)
df2['Supermarket_Type'].replace('Grocery Store',1, inplace=True)
# df2[['Supermarket_Type']]
# sns.boxplot(x="Supermarket_Type", y="Product_Supermarket_Sales", data=df)
# grpd_2=grouped_2.groupby(['Supermarket_Type'],as_index=False)
# f_val, p_val = stats.f_oneway(grpd_2.get_group('Supermarket Type1')['Product_Supermarket_Sales'], grpd_2.get_group('Supermarket Type2')['Product_Supermarket_Sales'], grpd_2.get_group('Supermarket Type3')['Product_Supermarket_Sales'], grpd_2.get_group('Grocery Store')['Product_Supermarket_Sales'])  
# print( "ANOVA results: F=", f_val, ", P =", p_val)   
# df['Supermarket_Type'].value_counts()
grouped_3=df[['Supermarket_Location_Type', 'Product_Supermarket_Sales']]
grpd_3=grouped_3.groupby(['Supermarket_Location_Type'],as_index=False).mean()
grpd_3
df['Supermarket_Location_Type'].replace('Cluster 2',3, inplace=True)
df['Supermarket_Location_Type'].replace('Cluster 3',2, inplace=True)
df['Supermarket_Location_Type'].replace('Cluster 1',1, inplace=True)
# df['Supermarket_Location_Type'].value_counts()

df2['Supermarket_Location_Type'].replace('Cluster 2',3, inplace=True)
df2['Supermarket_Location_Type'].replace('Cluster 3',2, inplace=True)
df2['Supermarket_Location_Type'].replace('Cluster 1',1, inplace=True)
# df2['Supermarket_Location_Type'].value_counts()
# df['Supermarket_Location_Type'].value_counts()
sns.boxplot(x="Supermarket_Location_Type", y="Product_Supermarket_Sales", data=df)
# grpd_3=grouped_3.groupby(['Supermarket_Location_Type'],as_index=False)
# f_val, p_val = stats.f_oneway(grpd_3.get_group('Cluster 1')['Product_Supermarket_Sales'], grpd_3.get_group('Cluster 2')['Product_Supermarket_Sales'], grpd_3.get_group('Cluster 3')['Product_Supermarket_Sales'])  
# print( "ANOVA results: F=", f_val, ", P =", p_val)   
df['Product_Fat_Content'].value_counts()
grouped_4=df[['Product_Fat_Content', 'Product_Supermarket_Sales']]
grpd_4=grouped_4.groupby(['Product_Fat_Content'],as_index=False).mean()
grpd_4
df['Product_Fat_Content'].replace('Low Fat',3, inplace=True)
df['Product_Fat_Content'].replace('Normal Fat',2, inplace=True)
df['Product_Fat_Content'].replace('Ultra Low fat',1, inplace=True)
# df['Product_Fat_Content'].value_counts()
df2['Product_Fat_Content'].replace('Low Fat',3, inplace=True)
df2['Product_Fat_Content'].replace('Normal Fat',2, inplace=True)
df2['Product_Fat_Content'].replace('Ultra Low fat',1, inplace=True)
# df2['Product_Fat_Content'].value_counts()
# sns.boxplot(x='Product_Fat_Content', y='Product_Supermarket_Sales', data=df)
# grpd_4=grouped_4.groupby(['Product_Fat_Content'],as_index=False)
# f_val, p_val = stats.f_oneway(grpd_4.get_group('Low Fat')['Product_Supermarket_Sales'], grpd_4.get_group('Normal Fat')['Product_Supermarket_Sales'], grpd_4.get_group('Ultra Low fat')['Product_Supermarket_Sales'])  
# print( "ANOVA results: F=", f_val, ", P =", p_val)   
df['Supermarket _Size'].value_counts()
grouped_5=df[['Supermarket _Size', 'Product_Supermarket_Sales']]
grpd_5=grouped_5.groupby(['Supermarket _Size'],as_index=False).mean()
grpd_5
df['Supermarket _Size'].replace('Medium',3, inplace=True)
df['Supermarket _Size'].replace('Small',2, inplace=True)
df['Supermarket _Size'].replace('High',1, inplace=True)

df2['Supermarket _Size'].replace('Medium',3, inplace=True)
df2['Supermarket _Size'].replace('Small',2, inplace=True)
df2['Supermarket _Size'].replace('High',1, inplace=True)
sns.boxplot(x='Supermarket _Size', y='Product_Supermarket_Sales', data=df)
# grpd_5=grouped_5.groupby(['Supermarket _Size'],as_index=False)
# f_val, p_val = stats.f_oneway(grpd_5.get_group('Medium')['Product_Supermarket_Sales'], grpd_5.get_group('Small')['Product_Supermarket_Sales'],grpd_5.get_group('High')['Product_Supermarket_Sales'])  
# print( "ANOVA results: F=", f_val, ", P =", p_val) 
df['Product_Type'].value_counts()
# rdf= [x for x in df.Product_Type.value_counts().sort_values(ascending=False).head(11).index]
# for label in rdf:
#     df['Product_Type'+ label]=np.where(df['Product_Type']==label,1,0)
# for label in rdf:
#     df2['Product_Type'+ label]=np.where(df2['Product_Type']==label,1,0)
type_count= df['Product_Type'].value_counts().sort_values( ascending=False)
type_count
# grouped_11=df[['Product_Type','Product_Supermarket_Sales']]
# # grpd_11=grouped_11.groupby(['Product_Type'],as_index=False).sum()
# # grpd_11=grpd_11.sort_values(['Product_Supermarket_Sales'], ascending=False)
# # grpd_11
type_count.index
ptype_encode = {}
ptype_encode_values = range(16,0,-1)
for i,k in zip(type_count.index,ptype_encode_values):
    ptype_encode[i]=k
ptype_encode
df['Product_Type'] = df['Product_Type'].map(ptype_encode)
df
df2['Product_Type'] = df2['Product_Type'].map(ptype_encode)
df2
grouped_7=df[['Supermarket_Opening_Year','Supermarket_Identifier','Product_Supermarket_Sales']]
grpd_7=grouped_7.groupby(['Supermarket_Opening_Year','Supermarket_Identifier'],as_index=False).mean()
grpd_7.sort_values(['Product_Supermarket_Sales'], ascending=False)
# Engine size as potential predictor variable of price
sns.regplot(x="Supermarket_Opening_Year", y="Product_Supermarket_Sales", data=grpd_7)
plt.ylim(0,)
grpd_7.corr()
df.corr()
df.columns
# df.head()
# df2.head()
# No need for normalizing for this the Tree based algos though

# df['Product_Price']=df['Product_Price']/df['Product_Price'].max()
# df2['Product_Price']=df2['Product_Price']/df2['Product_Price'].max()

# df['Product_Weight']=df['Product_Weight']/df['Product_Weight'].max()
# df2['Product_Weight']=df2['Product_Weight']/df2['Product_Weight'].max()
# df.head()
X=df.drop(['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier','Product_Type','Product_Weight',
           'Supermarket_Opening_Year','Product_Fat_Content','Product_Supermarket_Sales'], axis=1)
X.head()
y=df.Product_Supermarket_Sales
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
reg = GradientBoostingRegressor()#**params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_train, reg.predict(X_train))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print(reg.score(X_train,y_train))
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print(reg.score(X_test,y_test))

# X2=df2.drop(['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier','Product_Fat_Content','Supermarket_Opening_Year'], axis=1)
# X2=pd.get_dummies(X2)
# X2.head()
# test_pred = reg.predict(X2) #predict on the test set for submission
# df3= {'Product_Supermarket_Identifier': df2['Product_Supermarket_Identifier'], 'Product_Supermarket_Sales': test_pred}
# sub = pd.DataFrame(data=df3)
# sub = sub[['Product_Supermarket_Identifier', 'Product_Supermarket_Sales']]
# sub.shape
# # sub.to_csv('submission.csv', index = False)
# subxamp=pd.read_csv('sample_submission.csv')
# subxamp.head()
from xgboost import XGBRegressor
parame = {"n_jobs":-1,'n_estimators':127,'learning_rate':0.08,
                    'max_depth':3,'subsample':0.9,'random_state':1,
                    'colsample_bylevel':0.9,'min_child_weight':2,
                    'reg_alpha':1
        }
# n_jobs=-1,n_estimators=127,learning_rate=0.08,
#                     max_depth=3,subsample=0.9,random_state =1,
#                     colsample_bylevel=0.9,min_child_weight=2,
#                     reg_alpha=1

# clf1 = XGBRegressor(n_estimators=64,learning_rate=0.08,max_depth=4,
#                      subsample=0.7, min_child_weight=2,reg_alpha=1)#**parame)

clf1 = XGBRegressor()
clf1.fit(X_train, y_train)
#very latest best
clf1 = XGBRegressor(n_estimators=62,learning_rate=0.07,max_depth=5,
                      subsample=0.7, min_child_weight=2,reg_alpha=1)
clf1.fit(X_train, y_train)
clf1 = XGBRegressor(n_estimators=62,learning_rate=0.07,max_depth=5,
                      subsample=0.7, min_child_weight=2,reg_alpha=0.8)
clf1.fit(X_train, y_train)
mse = mean_squared_error(y_train, clf1.predict(X_train))
print("The mean squared error (MSE) on train set: {:.4f}".format(mse))
print(clf1.score(X_train,y_train))

mse = mean_squared_error(y_test, clf1.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print(clf1.score(X_test,y_test))
from catboost import CatBoostRegressor
ctb = CatBoostRegressor(depth=6, learning_rate=0.019, n_estimators=270)
ctb.fit(X_train,y_train)
coeff_score = ctb.score(X_test,y_test)
coeff_score
ctb.score(X_train,y_train)
X2=df2.drop(['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier','Product_Type','Product_Weight',
           'Supermarket_Opening_Year','Product_Fat_Content'], axis=1)
X2.head()
test_pred = ctb.predict(X2) #predict on the test set for submission
df3= {'Product_Supermarket_Identifier': df2['Product_Supermarket_Identifier'], 'Product_Supermarket_Sales': test_pred}
sub = pd.DataFrame(data=df3)
sub = sub[['Product_Supermarket_Identifier', 'Product_Supermarket_Sales']]

sub.to_csv('submission_recent2.csv', index = False)
sub.head()
