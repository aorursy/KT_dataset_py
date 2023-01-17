%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
file_path='../input/dsn-ai-oau-july-challenge/train.csv'
df=pd.read_csv(file_path)
df.head()
path='../input/dsn-ai-oau-july-challenge/test.csv'
df2=pd.read_csv(path)
df2.head()
# df2.isnull().sum()
df.describe(include='all')
df.isnull().sum()
df2.isnull().sum()
#check the mode of the supermarket size values
df['Supermarket _Size'].value_counts().idxmax()
df2['Supermarket _Size'].value_counts().idxmax()
#replace the missing 'Supermarket_Size' values by the most frequent 
df["Supermarket _Size"].replace(np.nan, "Medium", inplace=True)
df2["Supermarket _Size"].replace(np.nan, "Medium", inplace=True)
df.isnull().sum()
df2.isnull().sum()
df.Product_Weight=df.sort_values(['Product_Identifier','Product_Weight']).Product_Weight.ffill( )

df2.Product_Weight=df.sort_values(['Product_Identifier','Product_Weight']).Product_Weight.ffill( )
df.isnull().sum()
df.isnull().sum()
# group 'Product_Identifier', 'Product_Weight' by 'Product_Identifier' so as to replace nan with the mean
#of the corresponding 'Product_Identifier'
df_gptest = df[['Product_Identifier', 'Product_Weight']]
grouped_test1 = df_gptest.groupby(['Product_Identifier'],as_index=False).mean() 
df.sort_values(['Product_Identifier','Supermarket_Identifier']).head()
df.dtypes
grouped_1=df[['Product_Type','Supermarket_Identifier','Product_Supermarket_Sales']]
grpd_1=grouped_1.groupby(['Product_Type','Supermarket_Identifier'],as_index=False).mean()
grpd_1=grpd_1.sort_values(['Product_Type','Product_Supermarket_Sales'], ascending=False)
grpd_1.head()
grouped_pivot = grpd_1.pivot(index='Product_Type',columns='Supermarket_Identifier')
grouped_pivot
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
grouped_2=df[['Supermarket_Type', 'Product_Supermarket_Sales']]
grpd_2=grouped_2.groupby(['Supermarket_Type'],as_index=False).mean()
grpd_2
sns.boxplot(x="Supermarket_Type", y="Product_Supermarket_Sales", data=df)
grpd_2=grouped_2.groupby(['Supermarket_Type'],as_index=False)
f_val, p_val = stats.f_oneway(grpd_2.get_group('Supermarket Type1')['Product_Supermarket_Sales'], grpd_2.get_group('Supermarket Type2')['Product_Supermarket_Sales'], grpd_2.get_group('Supermarket Type3')['Product_Supermarket_Sales'], grpd_2.get_group('Grocery Store')['Product_Supermarket_Sales'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)   
df['Supermarket_Type'].value_counts()
grouped_3=df[['Supermarket_Location_Type', 'Product_Supermarket_Sales']]
grpd_3=grouped_3.groupby(['Supermarket_Location_Type'],as_index=False).mean()
grpd_3
df['Supermarket_Location_Type'].value_counts()
sns.boxplot(x="Supermarket_Location_Type", y="Product_Supermarket_Sales", data=df)
grpd_3=grouped_3.groupby(['Supermarket_Location_Type'],as_index=False)
f_val, p_val = stats.f_oneway(grpd_3.get_group('Cluster 1')['Product_Supermarket_Sales'], grpd_3.get_group('Cluster 2')['Product_Supermarket_Sales'], grpd_3.get_group('Cluster 3')['Product_Supermarket_Sales'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)   
df['Product_Fat_Content'].value_counts()
grouped_4=df[['Product_Fat_Content', 'Product_Supermarket_Sales']]
grpd_4=grouped_4.groupby(['Product_Fat_Content'],as_index=False).mean()
grpd_4
sns.boxplot(x='Product_Fat_Content', y='Product_Supermarket_Sales', data=df)
grpd_4=grouped_4.groupby(['Product_Fat_Content'],as_index=False)
f_val, p_val = stats.f_oneway(grpd_4.get_group('Low Fat')['Product_Supermarket_Sales'], grpd_4.get_group('Normal Fat')['Product_Supermarket_Sales'], grpd_4.get_group('Ultra Low fat')['Product_Supermarket_Sales'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)   
df['Supermarket _Size'].value_counts()
grouped_5=df[['Supermarket _Size', 'Product_Supermarket_Sales']]
grpd_5=grouped_5.groupby(['Supermarket _Size'],as_index=False).mean()
grpd_5
sns.boxplot(x='Supermarket _Size', y='Product_Supermarket_Sales', data=df)
grpd_5=grouped_5.groupby(['Supermarket _Size'],as_index=False)
f_val, p_val = stats.f_oneway(grpd_5.get_group('Medium')['Product_Supermarket_Sales'], grpd_5.get_group('Small')['Product_Supermarket_Sales'],grpd_5.get_group('High')['Product_Supermarket_Sales'])  
print( "ANOVA results: F=", f_val, ", P =", p_val) 
df['Product_Type'].value_counts()
grouped_6=df[['Product_Type','Product_Supermarket_Sales']]
grpd_6=grouped_6.groupby(['Product_Type'],as_index=False).mean()
grpd_6
sns.boxplot(x='Product_Type', y='Product_Supermarket_Sales', data=grpd_1)
sns.plt.xticks(rotation=90)
list(df['Product_Type'].unique())
# grpd_6=grouped_6.groupby(['Supermarket _Size'],as_index=False)
# for 
# f_val, p_val = stats.f_oneway(grpd_6.get_group('Medium')['Product_Supermarket_Sales'], grpd_6.get_group('Small')['Product_Supermarket_Sales'],grpd_5.get_group('High')['Product_Supermarket_Sales'])  
# print( "ANOVA results: F=", f_val, ", P =", p_val) 
grouped_7=df[['Supermarket_Opening_Year','Supermarket_Identifier','Product_Supermarket_Sales']]
grpd_7=grouped_7.groupby(['Supermarket_Opening_Year','Supermarket_Identifier'],as_index=False).mean()
grpd_7.sort_values(['Product_Supermarket_Sales'], ascending=False)
sns.regplot(x="Supermarket_Opening_Year", y="Product_Supermarket_Sales", data=grpd_7)
plt.ylim(0,)
grpd_7.corr()
df.columns
# df.sort_values(['Product_Type','Product_Supermarket_Identifier','Product_Supermarket_Sales']).head(10)
df.head()
# df['Product_Price']=df['Product_Price']/df['Product_Price'].max()
# df2['Product_Price']=df2['Product_Price']/df2['Product_Price'].max()
df['Product_Weight']=df['Product_Weight']/df['Product_Weight'].max()
df2['Product_Weight']=df2['Product_Weight']/df2['Product_Weight'].max()
df.head()
# dummy_variable_1 = pd.get_dummies(df["Supermarket_Type"])
# dummy_variable_1.rename(columns={'Grocery Store':'Supermarket Type Grocery Store'}, inplace=True)
# dummy_variable_1.head()
X=df.drop(['Product_Identifier','Supermarket_Identifier','Product_Supermarket_Identifier','Product_Fat_Content','Supermarket_Opening_Year','Product_Supermarket_Sales'], axis=1)
X=pd.get_dummies(X)
X.head()
X.head()
y=df.Product_Supermarket_Sales
y.head()
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in naira)')
    plt.ylabel('Proportion of Products')

    plt.show()
    plt.close()
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
priii()
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV
# params = {'n_estimators': 500,
#           'max_depth': 4,
#           'min_samples_split': 5,
#           'learning_rate': 0.01,
#           'loss': 'ls'}
# params = {'n_estimators': 600,
#           'max_depth': 4,
#           'min_samples_split': 5,
#           'learning_rate': 0.01,
#           'loss': 'ls'}
# reg = GradientBoostingRegressor(**params)
# reg.fit(X_train, y_train)

# mse = mean_squared_error(y_train, reg.predict(X_train))
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# mse = mean_squared_error(y_test, reg.predict(X_test))
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# print ("R-squared for Train: %.2f" %reg.score(X_train, y_train))
# print ("R-squared for Test: %.2f"%reg.score(X_test, y_test))
# !pip install xgboost
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
from sklearn.linear_model import LinearRegression
lre=LinearRegression()

lre.fit(X_train,y_train)
lre.score(X_train,y_train)
lre.score(X_test,y_test)
from sklearn.metrics import mean_squared_error
yhat_train=lre.predict(X_train)
yhat_train[:5]
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(y_train, yhat_train))
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(y_test, lre.predict(X_test)))
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
yhat_test=lre.predict(X_test)
yhat_test[:5]
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_test, yhat_test, "Actual Values (Train)", "Predicted Values (Train)", Title)
