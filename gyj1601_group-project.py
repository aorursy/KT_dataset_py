SEED                    = 12345   # global random seed for better reproducibility
GLM_SELECTION_THRESHOLD = 0.001   # threshold above which a GLM coefficient is considered "selected"
from rmltk import explain, evaluate, model                        # simple module for training, explaining, and eval
import h2o                                                        # import h2o python bindings to h2o java server
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import numpy as np                                                # array, vector, matrix calculations
import operator                                                   # for sorting dictionaries
import pandas as pd                                               # DataFrame handling
import time                                                       # for timers
from sklearn.preprocessing import OneHotEncoder                   # for one-hot encoding
import xgboost as xgb                                             #xgboost for modeling
from math import exp, expm1                                       #make exponential
import matplotlib.pyplot as plt      # plotting
pd.options.display.max_columns = 999 # enable display of all columns in notebook

# enables display of plots in notebook
%matplotlib inline

np.random.seed(SEED)                     # set random seed for better reproducibility

h2o.init(max_mem_size='24G', nthreads=4) # start h2o with plenty of memory and threads
h2o.remove_all()                         # clears h2o memory
h2o.no_progress()                        # turn off h2o progress indicators    
df_train = pd.read_csv('https://gwu-workshop-kai.s3.amazonaws.com/train.csv')
test = pd.read_csv('https://gwu-workshop-kai.s3.amazonaws.com/test.csv')
# Categorical boolean mask
mask = df_train.dtypes==object

# filter categorical columns using mask and turn it into a list
cats = df_train.columns[mask].tolist()

#filter numeric/float columns
nums = df_train.columns[~mask].tolist()

#print determine data types
print("Categorical =" , cats)
print()
print("Numeric = ", nums )
split_ratio = 0.9 # 90%/10% train/test split

# execute split
split = np.random.rand(len(df_train)) < split_ratio
train = df_train[split]
valid = df_train[~split]

# summarize split
print('Train data rows = %d, columns = %d' % (train.shape[0], train.shape[1]))
print('Validation data rows = %d, columns = %d' % (valid.shape[0], valid.shape[1]))

#summarize test set
print('Test data rows = %d, columns = %d' % (test.shape[0], test.shape[1]))
#describe categorical data
df_train[cats].describe()
#drop column
droplist = ["Utilities"]
train = train.drop(columns = droplist)
valid = valid.drop(columns = droplist)
test = test.drop(columns = droplist)
missingsum = df_train.isnull().sum()
misslist = missingsum[missingsum>0].sort_values()
misslist.to_frame(name="Missing Value Counts")
#set 15% of total rows as threshold
drop_threshold = int(len(df_train) * 0.15)

#set mask
mask = misslist < drop_threshold

#filter the variables
miss = list(misslist.index[mask])
drop = [name for name in misslist.index if name not in miss]

#remove columns stated above
train = train.drop(columns = drop)
valid = valid.drop(columns = drop)
test = test.drop(columns = drop)

#print the missing values, and the dropped values
print("Missing values are:",miss)
print()
print("Drop the following values:",drop)

#filter the categorical missing variables
miss_cats = [name for name in miss if name in cats]

#filter the numeric missing variables
miss_nums = [name for name in nums if name in miss]

print("Categorical missing variables are:", miss_cats)
print()
print("Categorical numeric variables are:", miss_nums)
#describe the data
print(train[miss_cats].describe())
print()
print(train[miss_nums].describe())
#set the variable list
var = ["Electrical","MasVnrType","MasVnrArea"]

#use loop to fill the missing value with the most frequent records
for xs in var:
    train[xs] = train[xs].fillna(train[xs].value_counts().index[0])
    valid[xs] = valid[xs].fillna(valid[xs].value_counts().index[0])
    test[xs] = test[xs].fillna(test[xs].value_counts().index[0])

#check the result
for xs in var:
    print("training", xs, "missing",train[xs].isnull().sum(),
         ", valid", xs, "missing",valid[xs].isnull().sum(),
         ", test", xs, "missing",test[xs].isnull().sum(),)
# since NA means there is no basement, we change "NA" to "None"

#set variable name list
var = ["BsmtCond","BsmtQual","BsmtFinType1","BsmtFinType2","BsmtExposure"]

#use loop to change data
for xs in var:
    train[xs] = train[xs].fillna(value = "None")
    valid[xs] = valid[xs].fillna(value = "None")
    test[xs] = test[xs].fillna(value = "None")
    
#double check result
for xs in var:
    print("training", xs, "missing",train[xs].isnull().sum(),
         ", valid", xs, "missing",valid[xs].isnull().sum(),
         ", test", xs, "missing",test[xs].isnull().sum(),)
#set variable name list
var = ["GarageCond","GarageQual","GarageFinish","GarageType"]

#fill missing variables in var with None
for xs in var:
    train[xs] = train[xs].fillna(value = "None")
    valid[xs] = valid[xs].fillna(value = "None")
    test[xs] = test[xs].fillna(value = "None")
    
#filling "GarageYrBlt" with "0"
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(value = 0)
valid['GarageYrBlt'] = valid['GarageYrBlt'].fillna(value = 0)
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(value = 0)

#double check result
var.append("GarageYrBlt")
for xs in var:
    print("training", xs, "missing",train[xs].isnull().sum(),
         ", valid", xs, "missing",valid[xs].isnull().sum(),
         ", test", xs, "missing",test[xs].isnull().sum(),)
var = list(train.columns)
var.remove("Id")
var.remove("SalePrice")
#check pearson correlation
y_name = "SalePrice"
corr = pd.DataFrame(train[var + [y_name]].corr()[y_name]).iloc[:-1]
corr.columns = ['Pearson Correlation Coefficient']
corr.sort_values(by ='Pearson Correlation Coefficient',ascending = False ).head(10)
#scatter plot of Overall
plt.scatter(y = train["SalePrice"],x= train["OverallQual"])

#find the index of outlier
print(train[["Id","OverallQual"]].loc[train["OverallQual"]<3])

#store the outlier
out_OQ = list(train.loc[train["OverallQual"]<3].index)
#scatter plot of GrLivArea
plt.scatter(y = train["SalePrice"],x= train["GrLivArea"])

#find the index of outlier
print(train[["Id","GrLivArea"]].loc[train["GrLivArea"]>4000])

#store the outlier
out_GA = list(train["Id"].loc[train["GrLivArea"]>4000].index)
#scatter plot of GarageCars
plt.scatter(y = train["SalePrice"],x= train["GarageCars"])

#find the index of outlier
print(train[["Id","SalePrice"]].loc[train["SalePrice"]>700000])

#store the outlier
out_GC = list(train["Id"].loc[train["SalePrice"]>700000].index)
#scatter plot of GarageArea
plt.scatter(y = train["SalePrice"],x= train["GarageArea"])

#find the index of outlier
print(train[["Id","SalePrice"]].loc[train["GarageArea"]>1200])

#store the outlier      
out_GA2 = list(train["Id"].loc[train["GarageArea"]>1200].index)
#scatter plot
plt.scatter(y = train["SalePrice"],x= train["TotalBsmtSF"])

#show the outlier Id
print(train[["Id","SalePrice"]].loc[train["TotalBsmtSF"]>3000])

#save the id
out_TB = list(train["Id"].loc[train["TotalBsmtSF"]>3000].index)
#scatter plot
plt.scatter(y = train["SalePrice"],x= train["1stFlrSF"])

#show the outlier Id
print(train[["Id","SalePrice"]].loc[train["1stFlrSF"]>4000])

#save the id
out_1SF = list(train["Id"].loc[train["1stFlrSF"]>4000].index)
plt.scatter(y = train["SalePrice"],x= train["YearBuilt"])

out = train[(train["YearBuilt"]<1900) & (train["SalePrice"]>400000)]
print(out[["Id","SalePrice"]])

out_YB = list(out.index)
#outliers
out = []
names = [out_OQ, out_GA,out_GA2,out_GC,out_TB,out_1SF,out_YB]
for xs in names:
    for n in xs:
        out.append(n)

#print the outlier id
print(out)

#remove all outliers
train = train.drop(out,axis = 0)

#double check the shape
train.shape
plt.boxplot(train["SalePrice"])
#check log transformation on the "SalePrice" -looks good
train["SalePrice"].apply(np.log).hist()
#valid["SalePrice"].apply(np.log).hist()

#apply log transformation
train["SalePrice"] = np.log(train["SalePrice"])
valid["SalePrice"] = np.log(valid["SalePrice"])
print(train["SalePrice"].head())
#check the boxplot
plt.boxplot(train["SalePrice"])
#remove the outlier
out = train[train["SalePrice"]<10.75].index
train.drop(out,axis=0,inplace = True)
#deep copy the data frame
train_glm = train.copy()
valid_glm = valid.copy()
test_glm = test.copy()

# Categorical boolean mask
mask = train_glm.dtypes==object

# filter categorical columns using mask and turn it into a list
cats = train_glm.columns[mask].tolist()
#one-hot encode training frame
train_glm = pd.get_dummies(train_glm)
train_SP = train_glm["SalePrice"]

#one-hot encode test frame
valid_glm = pd.get_dummies(valid_glm)
valid_SP = valid_glm["SalePrice"]


#keep only the same new columns in the encoded new frames
train_diff_cols = list(set(train_glm.columns) - set(valid_glm.columns))
valid_diff_cols = list(set(valid_glm.columns) - set(train_glm.columns))
train_glm.drop(train_diff_cols, axis=1, inplace=True)
valid_glm.drop(valid_diff_cols, axis=1, inplace=True)


#check that columns are actually the same in both frames
print(train_glm.shape)
print(valid_glm.shape)
print(all(train_glm.columns == valid_glm.columns))

#one-hot encode test frame
test_glm = pd.get_dummies(test_glm)

#keep only the same new columns in the encoded new frames
train_diff_cols = list(set(train_glm.columns)-set(test_glm.columns))
train_glm.drop(train_diff_cols, axis=1, inplace=True)
valid_glm.drop(train_diff_cols, axis=1, inplace=True)

# check that columns are actually the same in encoded train and valid frames
print(train_glm.shape)
print(valid_glm.shape)
print(all(train_glm.columns == valid_glm.columns))

#remove columns in encoded test not in encoded train and vals
train_diff_cols = list(set(test_glm.columns)-set(train_glm.columns))
test_glm.drop(train_diff_cols, axis = 1, inplace = True)

#check that columns are actually the same in all encoded frames
print(train_glm.shape)
print(valid_glm.shape)
print(test_glm.shape)
print(all(train_glm.columns == valid_glm.columns)
      and all(valid_glm.columns == test_glm.columns))

#add the SalePrice column to the train and valid
train_glm["SalePrice"] = train_SP
valid_glm["SalePrice"] = valid_SP

#check the shape
print(train_glm.shape)
print(valid_glm.shape)
#assign input and response variables
y_name = "SalePrice"
x_names = [name for name in train_glm if name not in [y_name, "Id"]]

print('y_name =', y_name)
print()
print('x_names =', x_names)
#set the hyper parameter
hyper_param = {'alpha': [0.01, 0.25, 0.5, 0.75, 0.99] }

#create model and train it
grid = H2OGridSearch(H2OGeneralizedLinearEstimator(family="gaussian",link = "log",lambda_search=True,seed=SEED),
        hyper_params=hyper_param)

grid.train(x= x_names, y=y_name,training_frame=h2o.H2OFrame(train_glm), validation_frame=h2o.H2OFrame(valid_glm))
best_glm = grid.get_grid()[0]
best_glm
#the thereshold is set to be 0.001
selected_feature = []
print('Best penalized GLM coefficients:')
for c_name, c_val in sorted(best_glm.coef().items(), key=operator.itemgetter(1)):
    if abs(c_val) > GLM_SELECTION_THRESHOLD:
        print('%s %s' % (str(c_name + ':').ljust(25), c_val))
        if c_name != "Intercept":
            selected_feature.append(c_name)
best_glm.varimp_plot()
# collect regularization paths from dict in DataFrame
reg_path_dict = best_glm.getGLMRegularizationPath(best_glm)

reg_path_frame = pd.DataFrame(columns=reg_path_dict['coefficients'][0].keys())
for i in range(0, len(reg_path_dict['coefficients'])): 
    reg_path_frame = reg_path_frame.append(reg_path_dict['coefficients'][i], 
                                           ignore_index=True)

# plot regularization paths
fig, ax_ = plt.subplots(figsize=(8, 6))
_ = reg_path_frame[selected_feature].plot(kind='line', ax=ax_, title='Penalized GLM Regularization Paths',
                                      colormap='gnuplot')
_ = ax_.set_xlabel('Iteration')
_ = ax_.set_ylabel('Coefficient Value')
_ = ax_.axhline(c='k', lw=1, xmin=0.045, xmax=0.955)
_ = plt.legend(bbox_to_anchor=(1.05, 0),
               loc=3, 
               borderaxespad=0.)
fes = ["GrLivArea","OverallQual","YearBuilt","TotalBsmtSF","OverallCond"]
best_glm.partial_plot(h2o.H2OFrame(train_glm),cols = fes)
#use valid set to make prediction
yhat = best_glm.predict(h2o.H2OFrame(valid_glm))

#merge
glm_yhat_valid = pd.concat([valid_glm.reset_index(drop=True),
                           yhat.as_data_frame()],
                           axis = 1)

#rename
glm_yhat_valid = glm_yhat_valid.rename(columns = {"predict":"p_SalePrice"})

#find percentile
glm_percentile_dict = explain.get_percentile_dict("p_SalePrice",glm_yhat_valid,"Id")

#display
glm_percentile_dict
#show the SalePrice and predicted result
glm_yhat_valid[["SalePrice","p_SalePrice"]].head(10)
#'get current axis'
ax = plt.gca()

glm_yhat_valid.plot(x = "Id",y = "SalePrice",color = "red",ax=ax)
glm_yhat_valid.plot(x = "Id",y = "p_SalePrice", ax = ax)

#plt.show()
#assign model roles
x=train_glm[x_names]
y=train_glm['SalePrice']

#set the model
model_xgb = xgb.XGBRegressor(n_estimators=1150, 
                             max_depth=3, 
                             learning_rate=0.06,
                             colsample_bytree=0.2) 

model_xgb.fit(x, y)
model_xgb
#make prediction
p_SalePrice = best_glm.predict(h2o.H2OFrame(test_glm))
p_SalePrice = p_SalePrice.as_data_frame()

#format a new data frame
Id = test_glm["Id"]
glm = pd.DataFrame(data = Id)
glm["SalePrice"] = p_SalePrice

#make exponential
glm["SalePrice"] = np.exp(glm["SalePrice"])
    
#export csv file
glm.to_csv(r"../glm_submission.csv",index = False, header = True)
#display the head
glm.head()
#make prediction
test_glm.drop(columns = "Id",inplace = True)
pred_xgb=model_xgb.predict(test_glm)
pred_xgb = np.exp(pred_xgb)

#create dataframe
xgb = pd.DataFrame(data = test["Id"])
xgb["SalePrice"] = pred_xgb
xgb.to_csv(r"../xgb_submission.csv",index = False, header = True)
xgb.head()
#calculate the average prediction result
avg_result = (xgb["SalePrice"] +glm["SalePrice"])/2
avg = pd.DataFrame(data = test["Id"])
avg["SalePrice"] = avg_result

avg.to_csv(r"../avg_submission.csv",index = False, header = True)
avg.head(10)
#gbm
pros_gbm1 = H2OGradientBoostingEstimator(nfolds=5,
                                        seed=SEED,
                                        keep_cross_validation_predictions = True)
pros_gbm1.train(x=selected_feature, y=y_name, training_frame=h2o.H2OFrame(train_glm),validation_frame=h2o.H2OFrame(valid_glm))


#get score history
sh1 = pros_gbm1.score_history()

#set variables
x_var = "number_of_trees"
y1 = "training_rmse"
y2 = "validation_rmse"

##'get current axis'
ax = plt.gca()

sh1.plot(x = x_var,y = y1,color = "red",ax=ax)
sh1.plot(x = x_var,y = y2, ax = ax)
yhat_gbm1 = pros_gbm1.predict(h2o.H2OFrame(valid_glm))
#merge
glm_yhat_valid = pd.concat([valid_glm.reset_index(drop=True),
                           yhat_gbm1.as_data_frame()],
                           axis = 1)

#rename
glm_yhat_valid = glm_yhat_valid.rename(columns = {"predict":"p_SalePrice"})

#find percentile
glm_percentile_dict = explain.get_percentile_dict("p_SalePrice",glm_yhat_valid,"Id")

#display
glm_percentile_dict

glm_yhat_valid[["SalePrice","p_SalePrice"]].head(10)

#'get current axis'
ax = plt.gca()

glm_yhat_valid.plot(x = "Id",y = "SalePrice",color = "red",ax=ax)
glm_yhat_valid.plot(x = "Id",y = "p_SalePrice", ax = ax)

#plt.show()
pros_gbm2 = H2OGradientBoostingEstimator(nfolds=5,
                                        seed=SEED,
                                        keep_cross_validation_predictions = True)
pros_gbm2.train(x=x_names, y=y_name, training_frame=h2o.H2OFrame(train_glm),validation_frame=h2o.H2OFrame(valid_glm))

#get score history
sh2 = pros_gbm2.score_history()

##'get current axis'
ax = plt.gca()

sh2.plot(x = x_var,y = y1,color = "red",ax=ax)
sh2.plot(x = x_var,y = y2, ax = ax)

yhat_gbm2 = pros_gbm2.predict(h2o.H2OFrame(valid_glm))
#merge
glm_yhat_valid2 = pd.concat([valid_glm.reset_index(drop=True),
                           yhat_gbm2.as_data_frame()],
                           axis = 1)

#rename
glm_yhat_valid2 = glm_yhat_valid.rename(columns = {"predict":"p_SalePrice"})

#find percentile
glm_percentile_dict2 = explain.get_percentile_dict("p_SalePrice",glm_yhat_valid,"Id")

#display
glm_percentile_dict2

#'get current axis'
ax = plt.gca()

glm_yhat_valid2.plot(x = "Id",y = "SalePrice",color = "red",ax=ax)
glm_yhat_valid2.plot(x = "Id",y = "p_SalePrice", ax = ax)

#plt.show()

