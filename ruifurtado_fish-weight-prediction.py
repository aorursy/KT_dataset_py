import pandas as pd



data = pd.read_csv('../input/fish-market/Fish.csv')

data.head()
data.rename(columns={"Length1": "Vlength", "Length2": "Dlength", "Length3": "Clength"},inplace=True)

data.head()
data.shape
for col in data.columns:

    print('Column {} has {} missing values'.format(col,data[col].isnull().sum()))
for col in data.columns:

    print('Column {} type is {}'.format(col,data[col].dtype))
data['Species'].value_counts()
data.describe()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



plt.figure(figsize=(10,5))

sns.scatterplot(x="Weight", y="Height", hue="Species" ,data=data)
f,ax=plt.subplots(figsize=(10,5))

sns.heatmap(data.corr(), annot=True, cmap='YlGnBu')

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(20,10))



ax = sns.boxplot(x="Species", y="Weight", data=data, orient='v', 

    ax=axes[0, 0])

ax = sns.boxplot(x="Species", y="Height", data=data, orient='v', 

    ax=axes[0, 1])

ax = sns.boxplot(x="Species", y="Vlength", data=data, orient='v', 

    ax=axes[1, 0])

ax = sns.boxplot(x="Species", y="Width", data=data, orient='v', 

    ax=axes[1, 1])

ax = sns.boxplot(x="Species", y="Dlength", data=data, orient='v', 

    ax=axes[2, 0])

ax = sns.boxplot(x="Species", y="Clength", data=data, orient='v', 

    ax=axes[2, 1])
data.columns[1:]
outliers_index=[]

for dim in data.columns[1:]:   # to exclude species column

    for species in ['Roach','Smelt','Pike']:

        print('Outliers for {} in {} dimension'.format(species,dim))

        col=data[data['Species']==species][dim]

        Q1 = col.quantile(0.25)

        Q3 = col.quantile(0.75)

        IQR = Q3 - Q1

        outliers = col[(col < Q1 - (1.5 * IQR)) | (col > Q3 + (1.5 * IQR))]

        for index,outlier in zip(outliers.index,outliers):

            outliers_index.append(index) if index not in outliers_index else None

            print(index,outlier) 

    print("")
data.drop(outliers_index, inplace=True)

data.shape
y = data.Weight

X = data.loc[:,data.columns != 'Weight']
X=pd.get_dummies(X, columns=["Species"])

X=X.iloc[:,:-1] # remove the last column to avoid dummy variable trap

X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77, shuffle=True)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



model = LinearRegression() # Tried to normalize the dataset but actually results do not improve that much 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))
residuals = y_test-y_pred

plt.figure(figsize=(12,5))

plt.axhline(y=0,color="black")

plt.scatter(y_pred,residuals)
fig, axes = plt.subplots(3, 2, figsize=(20,15))



y_pred=y_pred.reshape(len(y_pred),)



ax = sns.scatterplot(x=X_test['Height'], y=y_pred, ax=axes[0, 0])

ax = sns.scatterplot(x=X_test["Height"], y=y_test, ax=axes[0, 0])

ax = sns.scatterplot(x=X_test['Vlength'], y=y_pred, ax=axes[0, 1])

ax = sns.scatterplot(x=X_test["Vlength"], y=y_test, ax=axes[0, 1])

ax = sns.scatterplot(x=X_test['Dlength'], y=y_pred, ax=axes[1, 0])

ax = sns.scatterplot(x=X_test["Dlength"], y=y_test, ax=axes[1, 0])

ax = sns.scatterplot(x=X_test['Clength'], y=y_pred, ax=axes[1, 1])

ax = sns.scatterplot(x=X_test["Clength"], y=y_test, ax=axes[1, 1])

ax = sns.scatterplot(x=X_test['Width'], y=y_pred, ax=axes[2, 0])

ax = sns.scatterplot(x=X_test["Width"], y=y_test, ax=axes[2, 0])
params = pd.Series(model.coef_.flatten(), index=X_test.columns)

params
print("y = {} + {}x1 + {}x2 + {}x3 + {}x4 {}x5 + {}x6 + {}x7 {}x8 + {}x9 + {}x10".format(round(params[0]),round(params[1],2),round(params[2],2),round(params[3],2),round(params[4],2),round(params[5],2),round(params[6],2),round(params[7],2),round(params[8],2),round(params[9],2),round(params[10],2)))