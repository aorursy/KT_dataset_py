#Libraries import



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



sns.set_style("whitegrid")



from sklearn import preprocessing



from sklearn.ensemble import RandomForestRegressor
#dataset import

data_raw = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv") 
#list the data columns type

data_raw.info()
print("Rows with more than 80% of NA values ")

num_rows = data_raw.shape[0]



for i in data_raw:

    if data_raw[data_raw[i] != data_raw[i]][i].shape[0] > 0.8*num_rows:

        print(i)
#Drop those columns

data_raw = data_raw.drop(["Alley","PoolQC","Fence","MiscFeature"],axis=1)
print("Number of uniques values by columns")

for i in data_raw:

    value_count = data_raw[i].value_counts().shape[0]

    print(i, " ", value_count)
plt.figure(figsize=(8,20))

plt.suptitle("Quick observation of features", fontsize="medium")

count = 0

for x in data_raw:

    count += 1

    plt.subplot(20,4,count)

    data_raw[x].value_counts().plot(kind='bar')

    plt.axis('off')
# Label encode of categorical columns  



data_encode = data_raw.copy()



#Create label encode

le = preprocessing.LabelEncoder()

for i in data_raw:

    if data_raw[i].dtype == "object":

        

        le.fit(data_raw[i].fillna('NA').value_counts().index.unique())



        data_encode[i] = le.transform(np.array(data_raw[i].fillna('NA')))

        

test_encode = test.copy()

        

for i in test:

    if test[i].dtype == "object":

        

        le.fit(test[i].fillna('NA').value_counts().index.unique())



        test_encode[i] = le.transform(np.array(test[i].fillna('NA')))
sns.set_context("notebook", font_scale=.3)
# Matrix of correlation between features



print("Matrix of correlation between features")

corr = data_encode.corr()

#plt.figure(figsize=(12, 12))

plt.title("Matrix of correlation between features (corr > |0.1|)", fontsize="large")

sns.heatmap(corr, vmax=1, square=True)



print(corr.shape)
# Select only columns with correlation < |0.1|



correlate_columns = corr[(corr['SalePrice']  > 0.1) | (corr['SalePrice']  < -0.1)]['SalePrice'].index

corr2 = data_encode[correlate_columns].corr() 



print("Number of features: ",corr2.shape[0])

print("List of features (corr > |0.1|)\n",correlate_columns)
sns.set_context("notebook", font_scale=.6)
# Matrix of correlation between features (corr > |0.1|)



plt.title("Matrix of correlation between features (corr > |0.1|)", fontsize="large")

sns.heatmap(corr2, vmax=1, square=True)
# Drop the target column name

correlate_columns = correlate_columns.drop("SalePrice")
#Fill missing values by the mean in each features

data_encode = data_encode.fillna(data_encode.mean())

test_encode = test_encode.fillna(test_encode.mean())
regressor = RandomForestRegressor(n_estimators=100, min_samples_split=2)
regressor.fit(data_encode[correlate_columns],data_encode['SalePrice'])
regressor.score(data_encode[correlate_columns],data_encode['SalePrice'])
predict = regressor.predict(data_encode[correlate_columns])
residuals = data_encode['SalePrice']-predict
sns.set_context("notebook", font_scale=1.3)
plt.figure(figsize=(5,3))

plt.title("Residuals")

residuals.hist(bins=100)

plt.show()
plt.figure(figsize=(5,3))

plt.title("Log(Residuals)")

np.log(residuals).hist(bins=100)

plt.show()
# Comparision between prediction and train



plt.figure(figsize=(8,40))

plt.title("Quick observation of features", fontsize="large")

count = 0



sns.set_context("notebook", font_scale=.5)



for x in correlate_columns:

    count += 1

    plt.subplot(30,2,count)

    plt.scatter(data_encode[x], data_encode['SalePrice'], s=1.5, c='b',edgecolors='none')

    plt.scatter(data_encode[x], predict, s=1, c='r',edgecolors='none')



sns.set_context("notebook", font_scale=.5)
#Outliers rows

residuals[residuals < -100000]
def model_test():

    regressor = RandomForestRegressor(n_estimators=100, min_samples_split=2)

    regressor.fit(data_encode[correlate_columns],data_encode['SalePrice'])

    score = regressor.score(data_encode[correlate_columns],data_encode['SalePrice'])

    print("Model score: ",score)

    predict = regressor.predict(data_encode[correlate_columns])

    residuals = data_encode['SalePrice']-predict

    #residuals.hist(bins=100)

    np.log(residuals).hist(bins=100)

    return None
data_encode = data_encode.drop(data_encode.index[[523,1298]])
model_test()
regressor.fit(data_encode[correlate_columns],data_encode['SalePrice'])
test_encode['SalePrice'] = regressor.predict(test_encode[correlate_columns])
submission = pd.DataFrame({'Id': test_encode['Id'],

                               'SalePrice': test_encode['SalePrice']})
submission.to_csv("submission_01.csv",index=False)