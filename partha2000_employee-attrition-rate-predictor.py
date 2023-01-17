# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_dataset_path = "/kaggle/input/Dataset/Train.csv"

test_dataset_path  = "/kaggle/input/Dataset/Test.csv"
# Data manipulation libraries

import pandas as pd

import numpy as np



# Data visualistaion libraries

import seaborn as sns

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

%matplotlib inline



# Machine learning libraries

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score



# Machine Learning Models

from sklearn.ensemble import RandomForestRegressor
# Both training and testing data

training_dataframe = pd.read_csv(train_dataset_path,index_col="Employee_ID")

testing_dataframe  = pd.read_csv(test_dataset_path,index_col ="Employee_ID")
training_dataframe.describe()
## Data Distribution

training_dataframe.hist(bins = 50, figsize = (20,15))
print(training_dataframe)
print(training_dataframe.Travel_Rate)
# All the columns

print(training_dataframe.columns)
training_dataframe.head()
training_dataframe.tail()
## Columns with missing values

cols_with_missing = [col for col in training_dataframe.columns if training_dataframe[col].isnull().any()]

print(cols_with_missing)
# Get a list of categorical columns



s = (training_dataframe.dtypes == 'object')

object_cols = list(s[s].index)

print(object_cols)
# Remove rows with missing target, separate target from predictors in training data

X_train_full = training_dataframe



X_train_full.dropna(axis=0, subset=['Attrition_rate'], inplace=True, how = "any")

y_train_full = X_train_full.Attrition_rate

# X_train_full.drop(['Attrition_rate'], axis=1, inplace=True)





X_train_full.head()
X_valid_full = testing_dataframe



labelEncoder = LabelEncoder()

for col in object_cols:

    X_train_full[col] = labelEncoder.fit_transform(X_train_full[col])

    X_valid_full[col] = labelEncoder.transform(X_valid_full[col])

    
X_train_full.head()
X_valid_full.head()
si = SimpleImputer(strategy='most_frequent')

X_train_imputed = pd.DataFrame(si.fit_transform(X_train_full))

# It is important to fit_transform the training data





# Imputation removed column names; they have to be put back

X_train_imputed.columns = X_train_full.columns

# X_train_imputed.index = X_train_full.index

print("shape of X_train_imputed = ",X_train_imputed.shape)

X_train_imputed.head()
print("shape of y_train_full = ",y_train_full.shape)
print("Shape of X_valid_full = ",X_valid_full.shape )
X_train_imputed.head()

data = X_train_imputed

ind_var = y_train_full



plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs Age")



sns.lineplot(y=data["Attrition_rate"],x=data["Age"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs time_since_promotion")

sns.lineplot(y=data["Attrition_rate"],x=data["Time_since_promotion"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs time_of_service")

sns.lineplot(y=data["Attrition_rate"],x=data["Time_of_service"])
sns.jointplot(x=data["Time_of_service"],y=data["Attrition_rate"],kind="kde")
plt.figure(figsize= (8,6))

plt.title("Attrition Rate Vs Gender")

sns.barplot(y=data["Attrition_rate"],x=data["Gender"])

plt.legend(title='Gender', loc='lower left', labels=['Male : 1', 'Female : 0'])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs eductaion_level")

sns.lineplot(y=data["Attrition_rate"],x=data["Education_Level"])
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs Work life balance")

sns.barplot(y=data["Attrition_rate"],x=data["Work_Life_balance"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs work_life_balance")

sns.lineplot(y=data["Attrition_rate"],x=data["Work_Life_balance"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs growth rate(%)")

sns.lineplot(y=data["Attrition_rate"],x=data["growth_rate"])
sns.jointplot(x=data["growth_rate"],y=data["Attrition_rate"],kind="kde")
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs Job Unit")

sns.barplot(y=data["Attrition_rate"],x=data["Unit"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs Pay scale")

sns.lineplot(y=data["Attrition_rate"],x=data["Pay_Scale"])
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs Pay_scale")

sns.barplot(y=data["Attrition_rate"],x=data["Pay_Scale"])
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs decision making skills")

sns.barplot(y=data["Attrition_rate"],x=data["Decision_skill_possess"])
plt.figure(figsize=(20,6))

plt.title("Attrition Rate Vs travel rate")

sns.lineplot(y=data["Attrition_rate"],x=data["Travel_Rate"])
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs Travel rate")

sns.barplot(y=data["Attrition_rate"],x=data["Travel_Rate"])
plt.figure(figsize= (15,10))

plt.title("Attrition Rate Vs Compensation_and_Benefits")

sns.barplot(y=data["Attrition_rate"],x=data["Compensation_and_Benefits"])
plt.title("Attrition Rate Vs Anominised variables")

sns.lineplot(y=data["Attrition_rate"],x=data["VAR1"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR2"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR3"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR4"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR5"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR6"])
sns.lineplot(y=data["Attrition_rate"],x=data["VAR7"])
corr_matrix = data.corr()

corr_matrix["Attrition_rate"].sort_values(ascending = False)
sample_data = data

plt.figure(figsize = (30,10))

corr = sample_data.corr()

ax = sns.heatmap(corr,vmin = -0.03,vmax = 0.03, center = 0,cmap=sns.diverging_palette(20, 220, n=200), square=True, linewidths = 0.5)



ax.set_xticklabels( ax.get_xticklabels(),rotation=45, horizontalalignment='right')
features_1 = ["Compensation_and_Benefits","Travel_Rate","Pay_Scale",

              "Unit","growth_rate","Education_Level","Time_of_service","Age"]



features_2 = ["Gender","Relationship_Status","Hometown","Unit",

              "Decision_skill_possess","Time_since_promotion",

              "growth_rate","Post_Level","Work_Life_balance"]     #Strong positive correaltion



features_3 = data.columns[:-1]



features_4 =["Gender","Unit","Work_Life_balance","Decision_skill_possess","Post_Level","growth_rate","Time_since_promotion","Travel_Rate",                 

"VAR4","Age","Pay_Scale", "VAR7","Time_of_service", "VAR2","Compensation_and_Benefits"]      # Strong negative and positive correaltion



features_5 =["Gender","Unit","Work_Life_balance","Decision_skill_possess","Post_Level","growth_rate","Time_since_promotion","Travel_Rate",                 

"VAR4","Age","Pay_Scale", "VAR7","Time_of_service", "VAR2","Compensation_and_Benefits","Relationship_Status"

            ,"Education_Level","VAR1"] 



X = data[features_4]

y = data["Attrition_rate"]



# Separating validation from training data

train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.7, test_size=0.3,random_state = 0)



# Model selection and Training



random_forest_model = RandomForestRegressor(random_state = 1)

random_forest_model.fit(train_X,train_y)

model_rf_preds = random_forest_model.predict(val_X)

mae_score_rf = mean_absolute_error(val_y,model_rf_preds)

rmse_rf = mean_squared_error(val_y, model_rf_preds, squared=False)

print("Mean absolute error with Random Forest = ",mae_score_rf)

print("Root mean square error with Random Forest = ",rmse_rf)

print("Final Score for comp = ",100*(1-rmse_rf))
# Model selection and training



def xgb_n_estimators_selection(x):

    xgb_model = XGBRegressor(n_estimators = x)

    xgb_model.fit(train_X,train_y)

    model_xgb_preds = xgb_model.predict(val_X)

    mae_score_xgb = mean_absolute_error(val_y,model_xgb_preds)

    return mae_score_xgb

mae_list=[]

n_min = 0;

for i in range(1,50):

    mae_list.append(xgb_n_estimators_selection(i))

    plt.plot(i,mae_list[i-1],'bo')

    

plt.title("MAE vs n_estimators fo XGBoost")

plt.xlabel("n_estimators")

plt.ylabel("MAE")
print("The best score is {val} at n_estimator of {n}".format(val=min(mae_list),n= mae_list.index(min(mae_list))))
xgbModel_updated = XGBRegressor(n_estimators = 500, learning_rate = 0.1,early_stopping_rounds = 20,

                  eval_set=[(val_X,val_y)], verbose = False)



xgbModel_updated.fit(train_X,train_y)





predictions_xgbModel_updated = xgbModel_updated.predict(val_X)

mae_xgbModel_updated = mean_absolute_error(predictions_xgbModel_updated,val_y)

print("MAE on updated XGB model wiht early stopping= ",mae_xgbModel_updated)

xgbModel_updated_rmse = mean_squared_error(predictions_xgbModel_updated,val_y,squared = False)

print("RMSE on xgbModel with early stopping= ",xgbModel_updated_rmse)

print("Final Score for comp = ",100*(1-xgbModel_updated_rmse))
from sklearn import linear_model

lr_model = linear_model.Lasso(alpha = 0.01)

lr_model.fit(train_X,train_y)

lr_model_prediction = lr_model.predict(val_X)

lr_model_mae = mean_absolute_error(lr_model_prediction,val_y)

print("MAE on Lasso regression model= ",lr_model_mae)

lr_model_rmse = mean_squared_error(lr_model_prediction,val_y,squared = False)

print("RMSE on lasso regression model= ",lr_model_rmse)

print("Final Score for comp = ",100*(1-lr_model_rmse))
from sklearn.linear_model import ElasticNet

en_model = ElasticNet(alpha = 0.01, l1_ratio = 0.8)       # l1_ratio is the ratio of ridge and lasso regression. Here 20:80

en_model.fit(train_X,train_y)

en_model_prediction = en_model.predict(val_X)

en_model_mae = mean_absolute_error(en_model_prediction,val_y)

print("MAE on Elastic Net regression model= ",en_model_mae)

en_model_rmse = mean_squared_error(en_model_prediction,val_y,squared = False)

print("RMSE on Elastic Net regression model= ",en_model_rmse)

print("Final Score for comp = ",100*(1-en_model_rmse))
# Linear Support Vector Regression



from sklearn.svm import LinearSVR

linear_svr_model = LinearSVR(epsilon = 1.5)

linear_svr_model.fit(train_X,train_y)

linear_svr_model_pred = linear_svr_model.predict(val_X)

linear_svr_model_mae = mean_absolute_error(linear_svr_model_pred,val_y)

print("MAE on Linear Support Vector regression model= ",linear_svr_model_mae)

linear_svr_model_rmse = mean_squared_error(linear_svr_model_pred,val_y,squared = False)

print("RMSE on Linear Support Vector regression model= ",linear_svr_model_rmse)

print("Final Score for comp = ",100*(1-linear_svr_model_rmse))
# SVR with poly kernel



from sklearn.svm import SVR

svr_model = SVR(kernel = "poly",epsilon = 0.1)

svr_model.fit(train_X,train_y)

svr_model_pred = svr_model.predict(val_X)

svr_model_mae = mean_absolute_error(svr_model_pred,val_y)

print("MAE on Support Vector regression model with poly kernel= ",svr_model_mae)

svr_model_rmse = mean_squared_error(svr_model_pred,val_y,squared = False)

print("RMSE on Support Vector regression model with poly kernel= ",svr_model_rmse)

print("Final Score for comp = ",100*(max(0,1-svr_model_rmse)))
from sklearn.linear_model import PoissonRegressor



pr_model = PoissonRegressor(max_iter=300)

pr_model.fit(train_X,train_y)

pr_model_pred = pr_model.predict(val_X)

pr_model_mae = mean_absolute_error(pr_model_pred,val_y)

print("MAE on Poisson regression model = ",pr_model_mae)

pr_model_rmse = mean_squared_error(pr_model_pred,val_y,squared = False)

print("RMSE on Poisson regression model = ",pr_model_rmse)

print("Final Score for comp = ",100*(max(0,1-pr_model_rmse)))


def alpha_tuning_pr(x):

    test_model  = PoissonRegressor(alpha = x,max_iter = 300)

    test_model.fit(train_X,train_y)

    test_pred = test_model.predict(val_X)

    test_rmse = mean_squared_error(pr_model_pred,val_y,squared = False)

    return test_rmse

    
rmse_list_pr= []

alpha_list_pr = []

a = 0;

for i in np.arange(1e-15,9e-15, 1e-15):

    rmse_list_pr.append(alpha_tuning_pr(i))

    alpha_list_pr.append(i)

    plt.plot(i,alpha_tuning_pr(i),'bo')

    

plt.title("RMSE vs alpha for Poisson Regression")

plt.xlabel("alpha")

plt.ylabel("RMSE")
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor



gbr_model =  HistGradientBoostingRegressor(loss="poisson",max_leaf_nodes=2)

gbr_model.fit(train_X,train_y)

gbr_model_pred = gbr_model.predict(val_X)

gbr_model_mae = mean_absolute_error(gbr_model_pred,val_y)

print("MAE on gbr_model = ",gbr_model_mae)

gbr_model_rmse = mean_squared_error(gbr_model_pred,val_y,squared = False)

print("RMSE on gbr_model = ",gbr_model_rmse)

print("Final Score for comp = ",100*(max(0,1-gbr_model_rmse)))
def maxLeafNodes_tuning_gbr(x):

    test_model  = HistGradientBoostingRegressor(loss="poisson",max_leaf_nodes=x)

    test_model.fit(train_X,train_y)

    test_pred = test_model.predict(val_X)

    test_rmse = mean_squared_error(test_pred,val_y,squared = False)

    return test_rmse
rmse_list_gbr= []

leaf_list_gbr = []

a = 0;

for i in range(10,200,10):

    rmse_list_gbr.append(maxLeafNodes_tuning_gbr(i))

    leaf_list_gbr.append(i)

    plt.plot(i,maxLeafNodes_tuning_gbr(i),'bo')

    

plt.title("RMSE vs Max Leaf nodes for GBR")

plt.xlabel("Max Leaf nodes for GBR")

plt.ylabel("RMSE")
from sklearn.linear_model import TweedieRegressor

tr_model = TweedieRegressor(power=0, alpha=1, link='log')

tr_model.fit(train_X,train_y)

tr_model_pred = tr_model.predict(val_X)

tr_model_mae = mean_absolute_error(tr_model_pred,val_y)

print("MAE on TweedieRegressor_model = ",tr_model_mae)

tr_model_rmse = mean_squared_error(tr_model_pred,val_y,squared = False)

print("RMSE on TweedieRegressor_model = ",tr_model_rmse)

print("Final Score for comp = ",100*(max(0,1-tr_model_rmse)))
from sklearn.linear_model import PassiveAggressiveRegressor

par_model = PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)

par_model.fit(train_X,train_y)

par_model_pred = par_model.predict(val_X)

par_model_mae = mean_absolute_error(par_model_pred,val_y)

print("MAE on Passive Agressive Regressor model = ",par_model_mae)

par_model_rmse = mean_squared_error(par_model_pred,val_y,squared = False)

print("RMSE on Passive Agressive Regressor model = ",par_model_rmse)

print("Final Score for comp = ",100*(max(0,1-par_model_rmse)))
from sklearn.linear_model import OrthogonalMatchingPursuit

omp_model = OrthogonalMatchingPursuit()

omp_model.fit(train_X,train_y)

omp_model_pred = omp_model.predict(val_X)

omp_model_mae = mean_absolute_error(omp_model_pred,val_y)

print("MAE on OMP model = ",omp_model_mae)

omp_model_rmse = mean_squared_error(omp_model_pred,val_y,squared = False)

print("RMSE on OMP model = ",omp_model_rmse)

print("Final Score for comp = ",100*(max(0,1-omp_model_rmse)))
from sklearn import linear_model

br_model = linear_model.BayesianRidge()

br_model.fit(train_X,train_y)

br_model_pred = br_model.predict(val_X)

br_model_mae = mean_absolute_error(br_model_pred,val_y)

print("MAE on Bayesian Ridge model = ",br_model_mae)

br_model_rmse = mean_squared_error(br_model_pred,val_y,squared = False)

print("RMSE on BR model = ",br_model_rmse)

print("Final Score for comp = ",100*(max(0,1-br_model_rmse)))
from sklearn import linear_model

ard_reg_model = linear_model.ARDRegression()

ard_reg_model.fit(train_X,train_y)

ard_reg_model_pred = ard_reg_model.predict(val_X)

ard_reg_model_mae = mean_absolute_error(ard_reg_model_pred,val_y)

print("MAE on ARD regresssor model = ",ard_reg_model_mae)

ard_reg_model_rmse = mean_squared_error(ard_reg_model_pred,val_y,squared = False)

print("RMSE on ARD Regressor model = ",ard_reg_model_rmse)

print("Final Score for comp = ",100*(max(0,1-ard_reg_model_rmse)))
# final_model =  XGBRegressor(n_estimators = 11)

final_model   =  pr_model

cvScores = -1 * cross_val_score(final_model, X, y, cv = 5, scoring = "neg_root_mean_squared_error" )

print("RMSE scores = : ", cvScores)

comp_scores = 100 * (1 - cvScores)

print("Actual score =",comp_scores)
print("Worst score = ", comp_scores.min())

print("Best score = ", comp_scores.max())

print("Average score = ",comp_scores.mean())
# final_model.fit(X,y)

pr_model.fit(X,y)

X_valid_for_submission = X_valid_full[features_4]



# X_valid_for_submission.head()



# Generating prediction on the testing data

new_X_valid = X_valid_for_submission.fillna(X_valid_for_submission.median())

full_prediction = pr_model.predict(new_X_valid)
output = pd.DataFrame({ 'Employee_ID':X_valid_full.index,

                       'Attrition_rate' : full_prediction

                      })

output.to_csv('submission21.csv', index = False)