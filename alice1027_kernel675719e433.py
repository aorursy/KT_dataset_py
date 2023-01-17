import warnings

warnings.simplefilter('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('/kaggle/input/boston-housing-dataset/train.csv')

#df_test = pd.read_csv('/kaggle/input/Boston_House_dataset/boston_test_data.csv')
df.head()
df.describe()
df.info()
#X = df.iloc[:,0:13]  #independent columns

#y = df.iloc[:,-1]    #target column i.e price range

#y = y.astype(int)





#model = ExtraTreesClassifier()

#model.fit(X,y)

#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

#feat_importances = pd.Series(model.feature_importances_, index=X.columns)

#feat_importances.nlargest(10).plot(kind='barh')

#plt.show()
X = df.iloc[:,0:13]  #independent columns

y = df.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,10))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Correlation with output variable

cor_target = abs(corrmat["MEDV"])

#Selecting highly correlated features

cor_target.sort_values().plot(kind='barh')

cor_target = cor_target[cor_target>0.5]

cor_target
df[cor_target.index].corr()
#The dataset is different from the one what I had downloaded, got a little bit different result.



#below result is using prrvious dataset..

#select four feature : indus,rm,ptratio,lstat

#but indus and lstat had highly corrected to each other, in linear regression, each of feature should be independent 

#variable, hence I drop indus to keep lstat with higher corrected with medv.



# Finally rm,ptratio,lstat are my feature selection
import statsmodels.api as sm



X = df[['RM','LSTAT']]

y = df["MEDV"]



# Note the difference in argument order

model = sm.OLS(y, X).fit()

predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()