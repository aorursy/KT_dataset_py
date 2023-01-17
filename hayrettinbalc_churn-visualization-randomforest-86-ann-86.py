import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df
df.info()
fig = px.box(df, y="Age")
fig.show()
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True)
age_labels = ['18-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
Age_group = pd.cut(df['Age'], range(10, 101, 10), right=False, labels=age_labels)
df.groupby(Age_group)['EstimatedSalary'].mean().plot(kind='bar',stacked=True)
plt.title("Estimated Salary Distribution by Age Groups",fontsize=14)
plt.ylabel('Estimated Salary')
plt.xlabel('Age Group');
df.groupby(Age_group)['Exited'].mean().plot(kind='bar',stacked=True)
plt.title("Distribution of Age Groups",fontsize=14)
plt.ylabel('Credit Score')
plt.xlabel('Age Group')
plt.figure(figsize=(20,20))
sns.catplot(x="Geography", y="EstimatedSalary", hue="Gender", kind="box", data=df)
plt.title("Geography VS Estimated Salary")
plt.xlabel("Geography")
plt.ylabel("Estimated Salary")
fig = px.box(df, x="Age", y="Geography", notched=True)
fig.show()

fig = px.parallel_categories(df, dimensions=['HasCrCard', 'IsActiveMember'],
                 color_continuous_scale=px.colors.sequential.Inferno,
                labels={'HasCrCard':'Credit Card Holder', 'IsActiveMember':'Activity Status'})
fig.show()
fig = px.parallel_categories(df, dimensions=['HasCrCard', 'Gender','IsActiveMember'],
                 color_continuous_scale=px.colors.sequential.Inferno,
                labels={'Gender':'Gender', 'HasCrCard':'Credit Card Holder', 'IsActiveMember':'Activity Status'})
fig.show()

fig = px.parallel_categories(df, dimensions=['IsActiveMember', 'Exited',],
                 color_continuous_scale=px.colors.sequential.Inferno,
                labels={'IsActiveMember':'Activity Status', 'Exited':'Exited Members',})
fig.show()

fig = plt.figure(figsize=(8,8))
sns.distplot(df.CreditScore, color="orange", label="CreditScore")
plt.legend();
fig = plt.figure(figsize=(8,8))
sns.distplot(df.Balance, color="red", label="Balance")
plt.legend();
fig = plt.figure(figsize=(8,8))
sns.distplot(df.EstimatedSalary, color="blue", label="Estimated Salary")
plt.legend();
df.drop('RowNumber', axis = 1, inplace = True)
df.drop('CustomerId', axis = 1, inplace = True)
df.drop('Surname', axis = 1, inplace = True)
df.Geography.unique()
df_geo = pd.get_dummies(df['Geography'], columns= df.Geography[0], dtype= 'int64')
df_gender = pd.get_dummies(df['Gender'], columns= df.Gender[0], dtype= 'int64')
df = df.join(df_geo)
df = df.join(df_gender)
df.drop('Geography', axis = 1, inplace = True)
df.drop('Gender', axis = 1, inplace = True)
df["Balance"] = df["Balance"].replace(0, np.nan)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
df.info()
imp = IterativeImputer(KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='kd_tree'))
df = imp.fit_transform(df)
df = pd.DataFrame(data=imp.transform(df), 
                             columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                                      'IsActiveMember', 'EstimatedSalary', 'Exited', 'France', 'Germany', 
                                      'Spain', 'Female', 'Male'])
fig = plt.figure(figsize=(8,8))
sns.distplot(df.Balance, color="red", label="Balance")
plt.legend();
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns = ['CreditScore', 'Balance', 'EstimatedSalary']
for col in columns:
    column = scaler.fit_transform(df[col].values.reshape(-1, 1))
    df[col] = pd.DataFrame(data=column, columns=[col])
exited_df = df['Exited']
df.drop('Exited', axis = 1, inplace = True)
df = df.join(exited_df)
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype('float')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)
len(y_train), len(y_val)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# fit the data
model.fit(X_train, y_train)

# Get predictions
y_preds = model.predict(X_val)

# Get score
accuracy_score(y_preds, y_val)
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(df)
pca.explained_variance_ratio_
plt.figure(figsize = (8,8))
plt.plot(range(1,15), pca.explained_variance_ratio_.cumsum(), marker= 'o', linestyle= '--')
plt.title('Explained Variance by Component')
plt.xlabel('Number of Component')
plt.ylabel('Cumulative Explained Variance')
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
from sklearn.cluster import KMeans
wccs = []
for i in range (1, 15):
    kmeans_pca = KMeans(n_clusters=i, init= 'k-means++', random_state = 42)
    kmeans_pca.fit(principalDf)
    wccs.append(kmeans_pca.inertia_)
plt.figure(figsize = (8,8))
plt.plot(range(1,15), wccs, marker= 'o', linestyle= '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-Means with PCA')
plt.show()
kmeans_pca = KMeans(n_clusters= 2, init = 'k-means++', random_state=42)
kmeans_pca.fit(principalDf)
principalDf['KmeansPredict'] = kmeans_pca.labels_
principalDf
x_axis = principalDf['principal component 1']
y_axis = principalDf['principal component 2']
plt.figure(figsize= (8,8))
sns.scatterplot(x_axis, y_axis, hue= principalDf['KmeansPredict'], palette=['r', 'b'])
plt.show()
x_axis = principalDf['principal component 1']
y_axis = principalDf['principal component 2']
plt.figure(figsize= (8,8))
sns.scatterplot(x_axis, y_axis, hue= df['Exited'], palette=['r', 'b'])
plt.show()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='orthogonal', activation='softplus'))
model.add(Dense(8, kernel_initializer='orthogonal', activation='softplus'))
model.add(Dense(4, kernel_initializer='orthogonal', activation='softplus'))
model.add(Dense(1, kernel_initializer='orthogonal', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=250, batch_size= 200)
y_preds = model.predict(X_val)
y_preds = y_preds > 0.5
accuracy_score(y_preds, y_val)
