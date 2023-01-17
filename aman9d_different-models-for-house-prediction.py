import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import skew
import matplotlib
%pylab inline
pd.options.display.max_columns = 300
train = pd.read_csv("../input/train.csv")
target = train["SalePrice"]
train = train.drop("SalePrice",1) # take out the target variable
test = pd.read_csv("../input/test.csv")
combi = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition'])) # this is the combined data frame without the target variable
print(shape(train))
print(shape(test))
print(shape(combi))
combi.head()
figure(figsize(8,4))
subplot(1,2,1)
hist(target*1e-6);
xlabel("Sale Price in Mio Dollar")
subplot(1,2,2)
hist(log1p(target));
xlabel("log1p(Sale Price in Dollar)")
target1 = log1p(target)
#log transform skewed numeric features:
numeric_feats = combi.dtypes[combi.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

combi[skewed_feats] = np.log1p(combi[skewed_feats])
#Next, let's look at some rows of the data. There are a lot of categorical data and NaN values, it's a bit of a mess:
combi.head(10)
# create new features from categorical data:
combi = pd.get_dummies(combi)
# and fill missing entries with the column mean:
combi = combi.fillna(combi.mean())

# create the new train and test arrays:
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
train.isnull().sum().max()
combi.shape
model = LinearRegression()
score = mean(sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = 5)))
score1 = mean(sqrt(-cross_val_score(model, train, target1,scoring="neg_mean_squared_error", cv = 5)))
print("linear regression score: ", score)
print("linear regression score1: ", score1)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, target1, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()
#cv_ridge
model_linearRegression = LinearRegression()
cv_linearRegression = rmse_cv(model_linearRegression).mean()
cv_linearRegression = pd.Series(cv_linearRegression)
cv_linearRegression.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_linearRegression
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train, target1)
cv_lasso = [rmse_cv(model_lasso).mean()]
print(cv_ridge.min())
print(cv_linearRegression.min())
print(cv_lasso)
coef = pd.Series(model_lasso.coef_, index = train.columns)
coef = pd.Series(model_lasso.coef_, index = train.columns)
print("lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the ridge Model")
#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(train), "true":target1})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
#Feedforward Neural Nets doesn't seem to work well at all...I wonder why.
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train = StandardScaler().fit_transform(train)
X_tr, X_val, y_tr, y_val = train_test_split(train, target1,test_size=0.2, random_state = 3)
X_tr.shape
X_val.shape
model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss = "mse", optimizer = "adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
pd.Series(model.predict(X_val)[:,0]).hist()
model_tree = DecisionTreeRegressor()
model_tree.fit(X_tr, y_tr)
score = (y_val - model_tree.predict(X_val)).mean()
print(score)
print(rmse_cv(model_tree).mean())
model_random = RandomForestRegressor()
model_random.fit(X_tr, y_tr)
print(rmse_cv(model_random).mean())

from sklearn.linear_model import LogisticRegression
model_logReg = LogisticRegression()
model_logReg.fit(train,target1)

from sklearn.svm import SVR
model_svm = SVR()
model_svm.fit(X_tr,y_tr)
#print(model_svm.score(model_svm.predict(X_val),y_val))
print(rmse_cv(model_svm).mean())
