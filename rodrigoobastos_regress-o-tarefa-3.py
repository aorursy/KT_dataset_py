import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.head()
train.corr()
train.isnull().sum()
Ytrain = train["median_house_value"]
Ytrain.head()
Xtrain = train.iloc[:,1:9]
Xtrain.head()
Xtrainnf = train.iloc[:,1:9]
Xtrainnf["average_population"] = train["population"]/train["households"]
Xtrainnf["average_rooms"] = train["total_rooms"]/train["households"]
Xtrainnf["average_bedrooms"] = train["total_bedrooms"]/train["households"]
Xtrainnf.head()
trainnf = Xtrainnf.iloc[:,0:11]
trainnf["median_house_value"] = train["median_house_value"]
trainnf.head()
trainnf.corr()
Xtraincorr = Xtrainnf[["latitude","median_age","total_rooms","median_income","average_rooms"]]
Xtrainlin = (Xtrain - Xtrain.min())/(Xtrain.max() - Xtrain.min())
Xtrainlinnf = (Xtrainnf - Xtrainnf.min())/(Xtrainnf.max() - Xtrainnf.min())
Xtrainlincorr = (Xtraincorr - Xtraincorr.min())/(Xtraincorr.max() - Xtraincorr.min())
Xtrainnor = (Xtrain - Xtrain.mean())/Xtrain.std()
Xtrainnornf = (Xtrainnf - Xtrainnf.mean())/Xtrainnf.std()
Xtrainnorcorr = (Xtraincorr - Xtraincorr.mean())/Xtraincorr.std()
scores = []
linreg = LinearRegression()
Xselect = [Xtrain, Xtrainnf, Xtraincorr, Xtrainlin, Xtrainlinnf, Xtrainlincorr, Xtrainnor, Xtrainnornf, Xtrainnorcorr]
Xstring = ["Xtrain", "Xtrainnf", "Xtraincorr", "Xtrainlin", "Xtrainlinnf", "Xtrainlincorr","Xtrainnor", "Xtrainnornf", "Xtrainnorcorr"]
for Xvalue in Xselect:
    score = cross_val_score(linreg, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
    scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[scores.index(max(scores))])

scores = []
alphavec = []
for i in range (1, 101):
    alphavec.append(i*0.01)
for Xvalue in Xselect:
    for alpha in alphavec:
        ridreg = Ridge(alpha=alpha)
        score = cross_val_score(ridreg, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[scores.index(max(scores))//100], " Alpha:", alphavec[scores.index(max(scores))%100] )
scores = []
for Xvalue in Xselect:
    for alpha in alphavec:
        lasreg = Lasso(alpha=alpha)
        score = cross_val_score(lasreg, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[scores.index(max(scores))//100], " Alpha:", alphavec[scores.index(max(scores))%100] )
scores = []
for Xvalue in Xselect:
    for alpha in range (1,11):
        lasreg = Lasso(alpha=alpha)
        score = cross_val_score(lasreg, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[scores.index(max(scores))//10], " Alpha:", (1+scores.index(max(scores))%10) )
scores = []
for Xvalue in Xselect:
    for K in range (1,51):
        Knnu = KNeighborsRegressor(n_neighbors = K, weights = 'uniform')
        score = cross_val_score(Knnu, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[(scores.index(max(scores)))//50], " K:", (1+scores.index(max(scores))%50) )
scores = []
for Xvalue in Xselect:
    for K in range (1,51):
        Knnd = KNeighborsRegressor(n_neighbors = K, weights = 'distance')
        score = cross_val_score(Knnd, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[(scores.index(max(scores)))//50], " K:", (1+scores.index(max(scores))%50) )
scores = []
for Xvalue in Xselect:
    for prof in range (1,101):
        dectre = DecisionTreeRegressor(max_depth=prof)
        score = cross_val_score(dectre, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[(scores.index(max(scores)))//100], " Profundidade:", (1+scores.index(max(scores))%100) )
scores = []
for Xvalue in Xselect:
    for prof in range (1,11):
        randfor = RandomForestRegressor(n_estimators= 30, max_depth=prof)
        score = cross_val_score(randfor, Xvalue, Ytrain, scoring='neg_mean_squared_error', cv=10)
        scores.append(score.mean())
print ("MSE:", max(scores)," Xvalue:", Xstring[(scores.index(max(scores)))//10], " Profundidade:", (1+scores.index(max(scores))%10))
randfor = RandomForestRegressor(n_estimators= 30, max_depth=10)
score = cross_val_score(randfor, Xtrain, Ytrain, scoring='neg_mean_squared_error', cv=10)
score.mean()
randfor.fit(Xtrain, Ytrain)
test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testx = test.iloc[:,1:9]
testy = randfor.predict(testx)
testy
Ytabela = pd.DataFrame(test.Id)
Ytabela['median_house_value'] = testy
Ytabela.head()
Ytabela.to_csv("results.csv", index=False)