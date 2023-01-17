import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/BlackFriday.csv')
plt.style.use("fivethirtyeight")
plt.figure(figsize=(20,10))
df.head()
df.info()
df.describe()
# Functions
def plt_plot(group, column, plot):
    ax = plt.figure(figsize = (12, 6))
    df.groupby(group)[column].sum().sort_values().plot(plot)
    
def sns_plot(column):
    sns.countplot(df[column])
    
def sort_descending_df_head(group, column, count_head):
    group.sort_values(column, ascending=False, inplace=True)
    return group.head(count_head)
# Total number of unique users.
df["User_ID"].nunique()
# Total amount of money raised.
df['Purchase'].sum()
product_df = df.copy()
# Total amount of unique products
print("Product amount: ", len(product_df["Product_ID"].value_counts()))
product_age_df = product_df[['Product_ID', 'Age', 'Purchase']].groupby(['Age', 'Product_ID']).count()
product_age_df = product_age_df.sort_values(['Purchase'], ascending=False).reset_index()
product_age_df.head()
age_count = {'0-17': 0, '18-25': 0, '26-35': 0, '36-45': 0, '46-50': 0, '51-55': 0, '55+': 0}
rows_to_drop = []

for i in range(len(product_age_df)):
    age_range = product_age_df['Age'][i]
    if age_count[age_range] == 0:
        age_count[age_range] = 1
    else:
        rows_to_drop.append(i)
product_age_df = product_age_df.drop(product_age_df.index[rows_to_drop])
product_age_df
product_purchase_mean = product_df[['Product_ID', 'Purchase']].groupby("Product_ID").mean().reset_index()
product_purchase_sum = product_df[['Product_ID', 'Purchase']].groupby("Product_ID").sum().reset_index()
product_purchase_count = product_df[['Product_ID', 'Purchase']].groupby("Product_ID").count().reset_index()
sort_descending_df_head(product_purchase_mean, "Purchase", 10)
sort_descending_df_head(product_purchase_sum, "Purchase", 10)
sort_descending_df_head(product_purchase_count, "Purchase", 10)
# Total amount of unique men and women that bought a product. 
gender_df = df.copy()
gender_unique = gender_df.groupby(['User_ID','Gender'])['Gender'].count().groupby(['Gender']).count()
gender_unique
# Total amount of money collected by each gender.
plt.figure(figsize  = (20,10))
gender_purchase_total = gender_df[['Gender', 'Purchase']].groupby("Gender").sum()
gender_purchase_total.plot.bar()
sns.barplot("Gender", "Purchase", data = gender_df)
plt.show()
plt.figure(figsize = (20,8))
sns_plot("Gender")
# Cantidad de compras realizadas por Hombres (M) y Mujeres (F)
# Total amount of purchases made by men (M) and women (F)
df['Gender'].value_counts()
# Average money spent per gender
gender_df = df.copy()
gender_purchase_merge = gender_df[['Gender', 'Purchase']].groupby("Gender").mean()
gender_purchase_merge.plot.bar(figsize=(15,8))
plt.figure(figsize = (20,8))
sns.barplot("Gender", "Purchase", data = gender_df)
plt.show()
gender_purchase_merge
age_data_frame = df.copy()
# Number of purchase records by age.
plt.figure(figsize = (20,8))
sns.countplot(age_data_frame["Age"])
# Quantity of purchase records by age separated by gender.
plt.figure(figsize = (20,8))
sns.countplot(age_data_frame["Age"], hue=age_data_frame["Gender"])
# Average purchase by age and gender.
plt.figure(figsize = (20,8))
sns.barplot(x="Age", y="Purchase", hue="Gender", data=age_data_frame)
# The total sum of purchases in dollars by age range.
age_data_frame.pivot_table('Purchase', ['Age'], aggfunc=np.sum)
# Unique number of users by age range.
age_data_frame = age_data_frame.groupby('Age')['User_ID'].nunique()
age_data_frame
occupation_df = df.copy()
# Number of users per occupation.
users_in_occupation = occupation_df.groupby('Occupation')['User_ID'].nunique()
users_in_occupation.sort_values(ascending=False)
# The amount of purchases, the total dollars collected and the average purchase per occupation.
occupation_purchases = occupation_df.groupby('Occupation')['Purchase'].agg(['sum', 'count', 'mean'])
occupation_purchases.sort_values('sum', ascending=False)
# Number of users who bought by occupation and city.
users_in_occupation = occupation_df.groupby(['Occupation', 'City_Category'])['User_ID'].nunique()
users_in_occupation
# Number of records per occupation
df["Occupation"].value_counts()
occupation_df.drop(['User_ID', 'Marital_Status', 'Product_Category_2', 'Product_Category_3', 'Stay_In_Current_City_Years', 'Product_ID', 'City_Category'], axis=1, inplace=True)
occupation_df = occupation_df.groupby(["Occupation", "Product_Category_1"]).sum().reset_index()
occupation_df
# Amount of money spent per category, per occupation.
plt.figure(figsize = (20,8))
sns.catplot(x='Product_Category_1', y='Purchase', col="Occupation", col_wrap=3, data=occupation_df, kind="bar")
city_df = df.copy()
# Quantity of purchases made and amount of money collected by city.
city_purchase = city_df.groupby('City_Category')['Purchase'].agg(['sum', 'count'])
city_purchase
city_user = city_df.groupby('City_Category')['User_ID'].nunique()
city_user
# Total number of purchases by age and city
fig1, ax1 = plt.subplots(figsize = (20, 7))
sns.countplot(df['City_Category'], hue = df["Age"])
plt.figure(figsize = (20,8))
sns.boxplot('Age','Purchase', data = df)
plt.show()
stay_years_df = df.copy()
years_user = stay_years_df.groupby(['City_Category', 'Stay_In_Current_City_Years'])['User_ID'].nunique()
years_user
df["Stay_In_Current_City_Years"].value_counts()
df["Marital_Status"].value_counts()
marital_status_df = df.copy()
marital_gender_df = marital_status_df.groupby(['Gender', 'Marital_Status']).size()
marital_gender_df
marital = marital_status_df.groupby(['Marital_Status', 'Gender']).sum()
marital.drop(['User_ID', 'Occupation', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis=1)
product_category_df = df.copy()
product_category_df['Product_Category_2'].fillna(-1, inplace=True)
product_category_df['Product_Category_3'].fillna(-1, inplace=True)
product_category_df["Product_Category_2"] = product_category_df["Product_Category_2"].astype(np.int64)
product_category_df["Product_Category_3"] = product_category_df["Product_Category_3"].astype(np.int64)
product_category_df.drop(['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Stay_In_Current_City_Years'], axis=1, inplace=True)
product_category_df.sort_values(["Product_Category_1", "Product_Category_2", "Product_Category_3"], inplace=True)
product_category_df.head(15000)
# Amount of money spent for category 1
plt.figure(figsize = (20,8))
plt_plot("Product_Category_1", "Purchase", "bar")
# Amount of money spent for category 2
plt_plot("Product_Category_2", "Purchase", "bar")
# Amount of money spent for category 3
plt_plot("Product_Category_3", "Purchase", "bar")
product_category_1_df = product_category_df.drop(["Product_Category_2", "Product_Category_3"], axis=1)
product_category_2_df = product_category_df.drop(["Product_Category_1", "Product_Category_3"], axis=1)
product_category_3_df = product_category_df.drop(["Product_Category_1", "Product_Category_2"], axis=1)

product_category_2_df = product_category_2_df[product_category_2_df["Product_Category_2"] != -1]
product_category_3_df = product_category_3_df[product_category_3_df["Product_Category_3"] != -1]
category_sum = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0}
category_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0}

def sum_category_purchases(dataframe, category_dict):
    for i in range(len(dataframe)):
        category_number = dataframe.iloc[i][0]
        category_dict[category_number] = category_dict.get(category_number) + dataframe.iloc[i][1]
# Quantity of products purchased by category.
count_product_category_1_df = product_category_1_df.groupby("Product_Category_1").count().reset_index()
count_product_category_2_df = product_category_2_df.groupby("Product_Category_2").count().reset_index()
count_product_category_3_df = product_category_3_df.groupby("Product_Category_3").count().reset_index()

sum_category_purchases(count_product_category_1_df, category_count)
sum_category_purchases(count_product_category_2_df, category_count)
sum_category_purchases(count_product_category_3_df, category_count)
    
category_count
# Amount of money spent per category.
sum_product_category_1_df = product_category_1_df.groupby("Product_Category_1").sum().reset_index()
sum_product_category_2_df = product_category_2_df.groupby("Product_Category_2").sum().reset_index()
sum_product_category_3_df = product_category_3_df.groupby("Product_Category_3").sum().reset_index()

sum_category_purchases(sum_product_category_1_df, category_sum)
sum_category_purchases(sum_product_category_2_df, category_sum)
sum_category_purchases(sum_product_category_3_df, category_sum)
    
category_sum
plt.figure(figsize = (20,10))
sns.heatmap(df.isnull(), cbar=True)
df.isnull().sum()
df_pred = pd.read_csv('../input/BlackFriday.csv')
# Transform categorical to numerical data
cols = ['User_ID','Product_ID']
df_pred.drop(cols, inplace = True, axis =1)
df_pred.head()
df_pred['Age'] = df_pred['Age'].map({'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6})
df_pred['Gender'] = df_pred['Gender'].map({'M': 0,'F': 1})
df_pred['City_Category'] = df_pred['City_Category'].map({'A': 0,'B': 1,'C': 2})
df_pred['Stay_In_Current_City_Years'] = df_pred['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4})
prod_cat_1_2 = df_pred[['Product_Category_1', 'Product_Category_2']]
prod_cat_1_2 = prod_cat_1_2.dropna()
prod_cat_1_2.head()
df_pred = df_pred.dropna()
df_pred["Product_Category_2"] = df_pred["Product_Category_2"].astype(np.int64)
df_pred["Product_Category_3"] = df_pred["Product_Category_3"].astype(np.int64)
df_pred.head()
corrmat = df_pred.corr()
fig,ax = plt.subplots(figsize = (20,10))
sns.heatmap(corrmat, vmax=.8, square=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
x = prod_cat_1_2['Product_Category_1'].values
y = prod_cat_1_2['Product_Category_2'].values

x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("Accuracy:", reg.score(X_test, y_test))
age_aux = pd.read_csv('../input/BlackFriday.csv')
age_aux['Age'] = age_aux['Age'].map({'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6})

x = age_aux['Age'].values
y = age_aux['Marital_Status'].values

x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# K-Nearest Neighbor
knn_df = df_pred[['Occupation', 'Gender', 'Purchase']]
knn_df.head()
knn_df.info()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Best K
neighbors = np.arange(1, 40)

train_accuracy = []
test_accuracy = []

for i, k in enumerate (neighbors):
    clf = KNeighborsClassifier(n_neighbors = k)
    
    scaler = StandardScaler()  
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)  
    x_test = scaler.transform(X_test)
    
    clf.fit(x_train,y_train)
    train_accuracy.append(clf.score(x_train,y_train))
    test_accuracy.append(clf.score(x_test,y_test))

# Plot
plt.figure(figsize=(13, 8))

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.xlabel('Neighbors')
plt.ylabel('Accuracy')

plt.xticks(neighbors)

plt.show()

print('Best Accuracy is {} with K = {}'.format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))
clf = KNeighborsClassifier(n_neighbors = 37)

# X = Occupation, Purchase; Y = Gender
x, y = knn_df.loc[:,knn_df.columns != 'Gender'], knn_df.loc[:,'Gender']

# Create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

scaler = StandardScaler()  
scaler.fit(x_train)

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print('Accuracy: ', clf.score(x_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(x_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)
print('Accuracy: ', clf.score(x_test, y_test))
