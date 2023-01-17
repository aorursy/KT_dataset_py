import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

data = pd.read_csv("../input/mobile-price-classification/train.csv")
X = data.iloc[:,0:20]
y = data.iloc[:,-1]
print(y)

best_features = SelectKBest(score_func=chi2,k=10)
fit = best_features.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
featureScore = pd.concat([df_columns,df_scores],axis=1)
featureScore.columns = ['Feature','Score']
featureScore
featureScore = featureScore.sort_values(by="Score",ascending=False)
featureScore = featureScore.head(n=7)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
plt.bar(featureScore["Feature"],featureScore["Score"])
plt.show()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X,y)
model.feature_importances_
featureImp = pd.DataFrame(model.feature_importances_,index=X.columns,columns=["Importances"])
featureImp
featureImp = featureImp.sort_values(by="Importances",ascending=False)
featureImp = featureImp.head(n=7)
plt.figure(figsize=(10,10))
plt.bar(featureImp.index,featureImp['Importances'])
plt.show()
data_corr = data.corr()
plt.figure(figsize =(20,20))
sns.heatmap(data_corr,annot=True)
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc = RandomForestClassifier()
scores = cross_val_score(rfc,X,y,cv=10)
scores.mean()
kBest_features = featureScore["Feature"].values
kBestScore = cross_val_score(rfc,X[kBest_features],y,cv=10)
kBestScore.mean()
imp_features = featureImp.index.values
impScore = cross_val_score(rfc,X[imp_features],y,cv=10)
impScore.mean()
