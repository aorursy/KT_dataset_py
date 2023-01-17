import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline
# Load datasets.
train = pd.read_csv("../input/TrainData_2006-2010.csv")
test = pd.read_csv("../input/TestData_2014.csv")

study_variables = ['quantidade_doacoes','quantidade_doadores','total_receita',
    'media_receita','recursos_de_outros_candidatos/comites','recursos_de_pessoas_fisicas',
    'recursos_de_pessoas_juridicas','recursos_proprios','quantidade_despesas',
    'quantidade_fornecedores','total_despesa','media_despesa','grau','estado_civil','votos']
all_data = pd.concat((train.loc[:,study_variables], test.loc[:,study_variables]))
# Plot head of train, to check values.
train.head()
# Configure canvas for the graphics.
matplotlib.rcParams['figure.figsize'] = (15.0, 6.0)

# Plot comparison between Y and log(Y + 1) distribution.
votos = pd.DataFrame({
    "votos": train["votos"], 
    "log(votos + 1)": np.log1p(train["votos"])
})
votos.hist()
# Log transform the Y target:
train["votos"] = np.log1p(train["votos"])

# Log transform Skewed(> 0.75) numeric features on the dataset:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # Compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]                      # Filter by skew > 0.75
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])             # Log transform the skewed values
# Dummies: boolean variables derived from the categorical variables
dummies_columns = ['grau','estado_civil']
all_data = pd.get_dummies(all_data, columns=dummies_columns)
# Filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
# Separating data to use with sklearn:
X_train = all_data[:train.shape[0]][all_data.columns.difference(dummies_columns)]
X_test = all_data[train.shape[0]:][all_data.columns.difference(dummies_columns)]
y = train.votos
y_test = test.votos
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.metrics import mean_squared_error

def rmse_train(model):
    y_pred = model.predict(X_train)
    return mean_squared_error(y, y_pred)

def rmse_test(model):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
ridge_alphas = [2**i for i in range(-20,7)]
model_ridge = RidgeCV(alphas = ridge_alphas, store_cv_values = True).fit(X_train, y)
ridge_rmses = np.mean(model_ridge.cv_values_, axis=0)
print("Scores: (Train -> %f, Test -> %f)" % (model_ridge.score(X_train, y), model_ridge.score(X_test, y_test)))
cv_ridge = pd.Series(ridge_rmses, index = ridge_alphas)
cv_ridge.plot(title = "Ridge CV RMSEs - Just Do It")
plt.xlabel("Alfa")
plt.ylabel("RMSE")
ridge_min_rmse = cv_ridge.min()
ridge_min_rmse
# Let's look at the residuals as well:
ridge_preds = pd.DataFrame({"y Previsto":model_ridge.predict(X_train), "y Real":y})
ridge_preds["Resíduos"] = ridge_preds["y Real"] - ridge_preds["y Previsto"]
ridge_preds.plot(x = "y Previsto", y = "Resíduos",kind = "scatter")
lasso_alphas = [2**i for i in range(-15,-7)]
model_lasso = LassoCV(alphas = lasso_alphas, max_iter=1000, cv=20).fit(X_train, y)
lasso_rmses = np.mean(model_lasso.mse_path_, axis=1)
print("Scores: (Train -> %f, Test -> %f)" % (model_lasso.score(X_train, y), model_lasso.score(X_test, y_test)))
cv_lasso = pd.Series(lasso_rmses, index = lasso_alphas)
cv_lasso.plot(title = "Lasso CV RMSEs - Just Do It")
plt.xlabel("Alfa")
plt.ylabel("RMSE")
lasso_min_rmse = cv_lasso.min()
lasso_min_rmse
# Let's look at the residuals as well:
lasso_preds = pd.DataFrame({"y Previsto":model_lasso.predict(X_train), "y Real":y})
lasso_preds["Resíduos"] = lasso_preds["y Real"] - lasso_preds["y Previsto"]
lasso_preds.plot(x = "y Previsto", y = "Resíduos",kind = "scatter")
model_linear = LinearRegression().fit(X_train, y)
print("Scores: (Train -> %f, Test -> %f)" % (model_linear.score(X_train, y), model_linear.score(X_test, y_test)))
y_linear_pred = model_linear.predict(X_train)
mean_squared_error(y, y_linear_pred)
# Let's look at the residuals as well:
linear_preds = pd.DataFrame({"y Previsto":model_linear.predict(X_train), "y Real":y})
linear_preds["Resíduos"] = linear_preds["y Real"] - linear_preds["y Previsto"]
linear_preds.plot(x = "y Previsto", y = "Resíduos",kind = "scatter")
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coeficientes no modelo Lasso")
print("Modelo Ridge:\t%f\nModelo Lasso:\t%f\nModelo Linear:\t%f" % (rmse_test(model_ridge), rmse_test(model_lasso), rmse_test(model_linear)))
