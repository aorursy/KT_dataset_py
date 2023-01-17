import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv", index_col=0)
print(dataset.shape)
print(dataset.head())
print(dataset.info())
dataset['CarName'] = dataset['CarName'].str.split(' ',expand =True)[0]
print(dataset['CarName'].unique())
dataset['CarName'] = dataset['CarName'].replace({'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                                                 'vokswagen': 'volkswagen', 'vw': 'volkswagen', 'Nissan': 'nissan'})
print(dataset['CarName'].unique())
print(dataset.duplicated().sum())
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.kdeplot(dataset['price'],  color='blue', shade=True)
plt.xlabel("price")

plt.subplot(1,2,2)
sns.boxplot(dataset['price'], palette="Set3")

plt.show()
plt.figure(figsize=(16,5))
sorted = dataset.groupby(['CarName'])['price'].median().sort_values()
sns.boxplot(x=dataset['CarName'], y=dataset['price'], order = list(sorted.index))
plt.title("Car Name vs Price")
plt.xticks(rotation=90)
plt.show()
sns.pairplot(dataset, diag_kind="kde", vars=['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'price'])
plt.show()
sns.pairplot(dataset, diag_kind="kde", vars=['boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price'])
plt.show()
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
categorical_data = dataset[categorical_columns]
plt.figure(figsize=(20,15))
for index, item in enumerate(categorical_columns, 1):
    plt.subplot(3,3,index)
    sns.barplot(x = item, y = 'price', data = dataset)
plt.show()  
from scipy import stats
numerical_columns = dataset.select_dtypes(exclude='object').columns
for i in (list(numerical_columns)):
    pearson_coef, p_value = stats.pearsonr(dataset[i], dataset['price'])
    print(i.capitalize(), "Pearson Correlation:", pearson_coef, "P-value:", p_value)
    print("The correlation is not significant:", p_value>0.05)
    print()
dataset.drop(['symboling', 'carheight', 'stroke', 'compressionratio', 'peakrpm', 'doornumber'], axis=1, inplace=True)
print(dataset.shape)
data_new = dataset.copy()
t_price = data_new.groupby(['CarName'])['price'].mean()
data_new = data_new.merge(t_price.reset_index(), how='left', on='CarName')
bins = [0,10000,20000,40000]
label =['Budget_Friendly','Medium_Range','Expensive_Cars']
dataset['Cars_Category'] = pd.cut(data_new['price_y'], bins, right=False, labels=label)
dataset.drop("CarName", axis=1, inplace=True)
dataset.head()
column = ['fueltype','aspiration','carbody', 'drivewheel', 'enginelocation', 'enginetype','cylindernumber', 'fuelsystem', 'Cars_Category']
dummies = pd.get_dummies(dataset[column], drop_first = True)
dataset = pd.concat([dataset, dummies], axis = 1)
dataset.drop(column, axis = 1, inplace = True)
print(dataset.shape)
from sklearn.preprocessing import QuantileTransformer
transform =  QuantileTransformer(n_quantiles=205)
columns = ['wheelbase', 'carlength', 'carwidth', 'curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']
dataset[columns] = transform.fit_transform(dataset[columns]) 
print(dataset.columns.values)
plt.figure(figsize = (40, 40))
sns.heatmap(dataset.corr(method ='pearson'), cmap='PuBu', annot=True, linewidths=.5, annot_kws={'size':8})
plt.show()
print(dataset.corr(method ='pearson').unstack().sort_values().drop_duplicates())
data = list(dataset.columns)
for i in data:
    pearson_coef, p_value = stats.pearsonr(dataset[i], dataset['price'])
    print(i.capitalize(), "Pearson Correlation:", pearson_coef, "P-value:", p_value)
    print("The correlation is not significant:", p_value>0.05)
    if p_value>0.05:
        dataset.drop(i, axis=1, inplace=True)
    print()
print(dataset.shape)
X = dataset.drop('price', axis=1)
y = dataset['price']
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["variables"] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
for index,column in enumerate(X.columns):
    print(index, column, vif['vif'][index])
    if vif['vif'][index]>10:
        vif = vif.drop([index], axis=0)
print(vif)
print(list(vif['variables']))
columns = list(vif['variables'])
data = dataset [columns]
data = pd.concat([data, dataset['price']], axis=1)
from sklearn.model_selection import train_test_split
X = data.drop('price', axis=1)
y = data ['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()
lr.fit(X_train,y_train)
pred_test = lr.predict(X_test)
pred_train = lr.predict(X_train)
print("R Squared Value of Train Data: {}".format(r2_score(y_train, pred_train)))
print("R Squared Value of Test Data: {}".format(r2_score(y_test, pred_test)))
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot((y_train - pred_train))
plt.title('Train Data Residual Analysis', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)

plt.subplot(1,2,2)
sns.distplot((y_test - pred_test))
plt.title('Test Data Residual Analysis', fontsize = 20)              
plt.xlabel('Errors', fontsize = 18)

plt.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(lr)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
plt.show()