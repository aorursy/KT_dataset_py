import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
filename = "/kaggle/input/eval-lab-1-f464-v2/train.csv"
# headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

#          "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

#          "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

#          "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filename) #, names = headers)
df.head(n=10)
df.info()
missing_values = df.isnull().sum()

missing_values[missing_values>0]

# replace "?" with NaN

df.replace("?", np.nan, inplace = True)

#df.head(5)
df.isnull().any().any()
missing_count = df.isnull().sum()

missing_count[missing_count > 0]
# Calculate mean for column normalized-losses

# df["normalized-losses"].mean()
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
df.head()
#numerical_features = [""]

#df[numerical_features] = df[numerical_features].astype("float")
df.dtypes
# simply drop whole row with NaN in "price" column

# df.dropna(subset=["price"], 0axis=0, inplace=True)



# reset index, because we droped two rows

# df.reset_index(drop=True, inplace=True)
#avg = df["normalized-losses"].mean()

#print("Average of normalized-losses:", avg_norm_loss)
# df["normalized-losses"].fillna(value=avg_norm_loss, inplace=True)

# OR

# df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df.fillna(value=df.mean(),inplace=True)

df.isnull().any().any()
# df['num-of-doors'].value_counts()
# df['type'].value_counts()
#replace the missing 'num-of-doors' values by the most frequent 

# df["num-of-doors"].replace(np.nan, "four", inplace=True)
df.head()
df.isnull().any().any()
df.describe()
df.describe(include='object')
# df['drive-wheels'].unique()
# df_group = df[['drive-wheels','body-style','price']]
# Use groupby to calculate average price for each category of drive-wheels

# grouped_test1 = df_group.groupby(['drive-wheels'],as_index=False).mean()

# grouped_test1
# List the data types for each column

print(df.dtypes)
# Engine size as potential predictor variable of price

sns.boxplot(x="type", y="rating", data=df)
#df[["engine-size", "price"]].corr()
#sns.regplot(x="highway-mpg", y="price", data=df)
#df[['highway-mpg', 'price']].corr()
#sns.regplot(x="", y="rating", data=df)

# import matplotlib.pyplot as plt

# plt.plot(df2['rating'])

# plt.ylabel('some numbers')

# plt.show()
# df[['peak-rpm','price']].corr()
# Find the correlation between x="stroke", y="price"

# df[["stroke","price"]].corr()
# Given the correlation results between "price" and "stroke" do you expect a linear relationship?

# Verify your results using the function "regplot()".

sns.regplot(x="id", y="feature4", data=df)
# sns.boxplot(x="type", y="rating", data=df)
# sns.boxplot(x="", y="price", data=df)
# drive-wheels

# sns.boxplot(x="drive-wheels", y="price", data=df)
df.corr()
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X = df[["id","feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]].copy()

y = df[['rating']].copy()
# y.head()

X
# numerical_features = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]

# categorical_features = ["type"]
# from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler()

# X_scaled = scaler.fit_transform(X[numerical_features])



# X_scaled
# X_encoded = pd.get_dummies(X[categorical_features])

# X_encoded.head()
# X_new = np.concatenate([X_scaled,X_encoded.values],axis=1)

# # X_new
# from sklearn.model_selection import train_test_split



# # X_train.head()

# a_train = X.iloc[:,0:13]

# a_train.head()



# b_train = y.iloc[:,0:1]

# b_train.head()

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=42)



from sklearn.linear_model import LinearRegression



reg_lr = LinearRegression().fit(X_train,y_train)



# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# scaler.fit(X_train)



# X_train = scaler.transform(X_train)

# X_test = scaler.transform(X_test)



# # from sklearn.neighbors import KNeighborsClassifier



# # classifier = KNeighborsClassifier(n_neighbors = 5)#,leaf_size=1,algorithm="auto",n_jobs=-1,p =30, weights="distance",metric="euclidean")

# # classifier.fit(X_train, y_train)



# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors=50)

# classifier.fit(X_train, np.ravel(y_train,order='C'))




y_pred = reg_lr.predict(X_val)

y_pred



# from sklearn.metrics import classification_report, confusion_matrix

# print(confusion_matrix(y_test, y_pred))

# print(classification_report(y_test, y_pred))
#root mean square error

from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(y_pred,y_val))

print(rmse)
df2 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

# df2.head()

# freq = df2['feature11'].isnull().sum()

# freq

# df2.isnull().any().any()

df2.fillna(value=df2.mean(),inplace=True)

df2.isnull().any().any()
X_t = df2[["id","feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]].copy()

y_t_pred = reg_lr.predict(X_t)

y_t_pred
df2.info()

df2['rating'] = y_t_pred
df2.info()
final_pred = df2[['id','rating']].copy()

final_pred['rating'] = final_pred['rating'].round(decimals=0)

final_pred['rating'] = final_pred['rating'].astype("int")

final_pred
final_pred.to_csv("eval_1_v2_pred2.csv",index=False,encoding='utf-8')