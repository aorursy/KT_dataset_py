import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

mobile_data = pd.read_csv("../input/mobile-price-classification/train.csv")

X = mobile_data.iloc[:,0:20]  #independent variables
y = mobile_data.iloc[:,-1]    #target variable i.e price range
mobile_data.head()
#apply SelectKBest class to extract top 10 best features

BestFeatures = SelectKBest(score_func=chi2, k=10)
fit = BestFeatures.fit(X,y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
#concatenating two dataframes for better visualization

f_Scores = pd.concat([df_columns,df_scores],axis=1)               # feature scores
f_Scores.columns = ['Specs','Score']  
f_Scores                # Score value is directly proportional to the feature importance
print(f_Scores.nlargest(10,'Score'))       # print 10 best features in descending order
import xgboost
import matplotlib.pyplot as plt

model = xgboost.XGBClassifier()
model.fit(X,y)
print(model.feature_importances_) 
# plot the graph of feature importances for better visualization 

feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')

plt.figure(figsize=(8,6))
plt.show()
import seaborn as sns

#get correlations of each features in dataset
corrmat = mobile_data.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(mobile_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")