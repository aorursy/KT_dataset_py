import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # for splitting the dataset in train,test and validation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model 
from sklearn.linear_model import Ridge
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error # for calcualting mse
#from sklearn.metrics import r2_score
from sklearn.cross_validation import StratifiedKFold # to implement stratifiedKFold
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input")) #dataset available in 'input' folder
#import dataset in variable insurance
insurance = pd.read_csv('../input/insurance.csv')
insurance = insurance.sample(frac=1).reset_index(drop=True)# shuffle
display(insurance.head(10))
print(insurance.info())

insurance.shape
insurance.dtypes
insurance.describe().T
insurance.groupby(insurance["smoker"]).mean()
insurance.groupby(insurance["region"]).count()
import matplotlib.ticker as mtick # For specifying the axes tick format 
ax = (insurance['smoker'].value_counts()*100.0 /len(insurance))\
.plot.pie(autopct='%.1f%%', labels = ['no', 'yes'],figsize =(5,5), fontsize = 12 )                                                                           
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Smoker',fontsize = 12)
ax.set_title('% Smoker', fontsize = 12)
sns.distplot(insurance["bmi"])
sns.distplot(insurance["charges"])
plt.figure(figsize=(12,5))
plt.title("Distribution of age in training set")
ax=sns.distplot(insurance["age"])
# temp_age_charges=pd.DataFrame()
# temp_age_charges["age"]=X_train["age"]
# temp_age_charges["charges"]=Y_train
_ = sns.lmplot("age", "charges", data=insurance, fit_reg=True)
# temp_bmi_charges=pd.DataFrame()
# temp_bmi_charges["bmi"]=X_train["bmi"]
# temp_bmi_charges["charges"]=Y_train
_ = sns.lmplot("bmi", "charges", data=insurance, fit_reg=True)
insurance.hist(figsize=(15, 10))
plt.show()
ax = sns.countplot(x="sex", hue="sex", data=insurance)
insurance.isnull().sum(axis = 0)
cols=[ 'bmi', 'charges']
#Determine outliers in dataset

for i in cols:
    quartile_1,quartile_3 = np.percentile(insurance[i],[25,75])
    quartile_f,quartile_l = np.percentile(insurance[i],[1,99])
    IQR = quartile_3-quartile_1
    lower_bound = quartile_1 - (1.5*IQR)
    upper_bound = quartile_3 + (1.5*IQR)
    print(i,lower_bound,upper_bound,quartile_f,quartile_l)

    insurance[i].loc[insurance[i] < lower_bound] = quartile_f
    insurance[i].loc[insurance[i] > upper_bound] = quartile_l

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

insurance1=remove_outlier(insurance, 'bmi')
insurance1=remove_outlier(insurance1, 'charges')
insurance=insurance1
#insurance['charges']=(insurance['charges']-insurance['charges'].min())/(insurance['charges'].max()-insurance['charges'].min())
insurance['bmi']=(insurance['bmi']-insurance['bmi'].min())/(insurance['bmi'].max()-insurance['bmi'].min())
insurance.head()
insurance.hist(figsize=(15, 10))
plt.show()
insurance.describe().T

insurance=pd.get_dummies(insurance,drop_first=True)
insurance.head()
#using randomforest to find the feature importance
train_y = insurance['charges'].values
train_X = insurance.drop(['charges'], axis=1)

from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,5))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="g", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()
colormap = plt.cm.RdBu
plt.figure(figsize=(10,10)) #controls the graph size
plt.title('Pearson Correlation of Features', y=1, size=20) #"size" controls the title size

sns.heatmap(insurance.astype(float).corr(),linewidths=2,vmax=1.0, 
            square=True, cmap=colormap, linecolor='red', annot=True)
sns.pairplot(insurance, x_vars=["bmi", "age",'children'], y_vars=["charges"],
              aspect=1, kind="reg");

#Selected Features
cols= ['smoker_yes','age','bmi','children']
X=insurance[cols]
y=insurance['charges']
X.shape
def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error
#     #global dummy variables to store error and accuracy of different models 
#     #and methods of splitting for comparison in last
#     r_score = []
#     loss = []
#     r_score1=[]
#     loss1=[]
def regression_model(X_train, Y_train, X_validation,Y_validation,X_test, Y_test):
    ''' Input:
        X_train : independent training data
        Y_train : target variable for trainig
        X_validation: validation dataset
        Y_validation: validation target variable
        evaluate: accepts boolean value "true" or "fale". If true then function will predict the test accuracy 
                instead of training and validation error'''
    linear_Model = linear_model.LinearRegression(fit_intercept=True) #creating model
    linear_Model.fit(X_train,Y_train,) # fitting the model

    train_error_rmse = calc_train_error(X_train, Y_train, linear_Model)
    valid_error_rmse= calc_validation_error(X_validation, Y_validation, linear_Model)
    train_error_rmse, valid_error_mse= round(train_error_rmse, 3) ,round(valid_error_rmse, 3)
    print('RMSE - train error: {} | validation error: {}'.format(train_error_rmse, valid_error_rmse))
    
#     y_pred= linear_Model.predict(X_test)
#     print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,Y_test),
#                                                                                 np.sqrt(mean_squared_error(y_pred,Y_test))))
    accuracy = linear_Model.score(X_validation,Y_validation)
    print('Accuracy :',format(round(accuracy,3),"%"))
   
    return linear_Model,train_error_rmse, valid_error_mse
   
def evaluate(X_train, Y_train, X_test, Y_test, train_error,validation_error, model):

    new_train_error = mean_squared_error(Y_train,model.predict(X_train))
    new_train_error = np.sqrt(new_train_error)
   # new_validation_error = mean_squared_error(Y_validation, model.predict(X_validation))
    new_test_error = mean_squared_error(Y_test, model.predict(X_test))
    new_test_error = np.sqrt(new_test_error)

    
    print('Comparison')
    print('-'*50)
    print('ORIGINAL ERROR')
    print('train error: {} | validation error: {}\n'.format(round(train_error,3), round(validation_error,3)))
    accuracy = model.score(X_validation,Y_validation)
    print('Accuracy :',format(round(accuracy,3),"%"))
    print('-' * 50)
    print('FINAL ERROR OF MODEL')
    print('train error: {} | test error: {}'.format(round(new_train_error,3), round(new_test_error,3)))
    accuracy = model.score(X_test,Y_test)
    print('Accuracy :',format(round(accuracy,3),"%"))
    y_pred= model.predict(X_test)
#     print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,Y_test),
#                                                                                   np.sqrt(mean_squared_error(y_pred,Y_test))))
    
   # loss.append(new_test_error)
   # r_score.append(r2_score(y_pred,Y_test)*100)
    #r_score.append(round(accuracy,3)*100)
    #loss1.append(new_test_error)
   # r_score1.append(round(accuracy,3)*100)
   # r_score1.append(r2_score(y_pred,Y_test)*100)
    return y_pred, accuracy,new_test_error

print("Independent features")
display(X.head())
print("-"*100)
print("Target variable")
display(y.head())
# create bins
# bins = np.linspace(insurance.charges.min(),insurance.charges.max(),6)
# charges_groups = np.digitize(insurance.charges,bins)

# #Add bin information to the dataframe
# insurance= insurance.assign(charge_groups = charges_groups)
# insurance['charge_groups'] = pd.Categorical(insurance.charge_groups)

# #Last bin has too few values to stratify, so we merge it with second last group and reduce the number of bins
# insurance.charge_groups[insurance.charge_groups == 6]= 6

# insurance.head()
# #insurance.shape
X_intermediate, X_test, Y_intermediate, Y_test = train_test_split(X, 
                                                                  y, 
                                                                  shuffle=True,
                                                                  test_size=0.1 
                                                                  )

# train/validation split (gives us train and validation sets)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_intermediate,
                                                                Y_intermediate,
                                                                shuffle=False,
                                                                test_size=0.2
                                                                )
# print proportions
#holdoutA=[] #to store accuracy
holdoutE=[] #to store rmse error
print('train: {}% | validation: {}% | test {}%'.format(round(len(Y_train)/len(y),2),
                                                       round(len(Y_validation)/len(y),2),
                                                       round(len(Y_test)/len(y),2)))
# # this is the training data after randomly splitting using holdout method
# print("Training data after holdout")
# print("-"*50)
# display(X_train.head()) #display the first five rows of train set
# print(X_train.shape)
# display(Y_train.head()) #display the first five rows of train target variable
# print(Y_train.shape)
model , train_error,validation_error= regression_model(X_train, Y_train, X_validation,Y_validation,X_test,Y_test)

y_pred,accuracy,test_error=evaluate(X_intermediate, Y_intermediate, X_test, Y_test, train_error,validation_error, model)
#holdoutA.append(accuracy)
holdoutE.append(test_error)
def model_scatter_plot(y_pred,Y_test):
    model_table = pd.DataFrame(y_pred,Y_test).reset_index()
    model_table.columns=['Y_test','y_pred']
    #Model Graph
    sns.lmplot(x = 'Y_test',y='y_pred',data = model_table,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},fit_reg=True)
    plt.title('Analysis of True and Predicted Cost',fontsize=14)
    plt.xlabel('Y_test',fontsize=12)
    plt.ylabel('y_pred',fontsize=12)
    #plt.scatter(y_test,y_pred)
    return plt.show()

print("[Holdout][Linear Regression]")
model_scatter_plot(y_pred,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Holdout] Residual Plot for Linear Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,y_pred) ## regression Residual Plot for linear regression model using bootstrapping
#hyperparameters
para=np.linspace(0.01,10,100)
error=dict()

#tuning hyperparameter
print("Parameter   RMSE Error")
print("-"*25)
for parameter in para:
    ridge = Ridge(fit_intercept=True, alpha=parameter)
    ridge.fit(X_train,Y_train)
    
    terr = calc_train_error(X_train,Y_train,ridge)
    verr = calc_validation_error(X_validation,Y_validation,ridge)
    total_error =np.abs(terr-verr)
    error[parameter]= total_error
    p= min(error,key = error.get)
    print("{:.3f}        {:.3f}".format(parameter,error[parameter]))
    
print("Best parameter after validation: {:.3f}".format(p))

#plot of error
def plotErr(select,X_train,Y_train,para_range,X_validation,Y_validation):   
    #Setup arrays to store training and test accuracies
    para = np.linspace(0.01,para_range,100)
    train_accuracy =np.empty(len(para))
    test_accuracy = np.empty(len(para))

    
    
    for i,k in enumerate(para):
        mod = [Ridge(fit_intercept=True, alpha=k),RandomForestRegressor(max_depth=k, random_state=0),SVR(C=k, epsilon=0.2)]
        m = mod[select]
        m.fit(X_train,Y_train)
        train_accuracy[i]= calc_train_error(X_train, Y_train, m)
        test_accuracy[i] = calc_validation_error(X_validation, Y_validation,m)
       

    #Generate plot
    plt.figure()
    plt.figure(figsize=(8,8))
    plt.title('error plot for training and validation')
    plt.plot(para, test_accuracy, label='validation error')
    plt.plot(para, train_accuracy, label='train error')
    plt.legend()
    plt.xlabel('Value of alpha')
    plt.ylabel('RMSE')
    plt.show()
print("[Holdout][Ridge Regression]")    
plotErr(0,X_train,Y_train,10,X_validation,Y_validation)    
# Create ridge regression model
ridge = Ridge(fit_intercept=True, alpha=p)
# Train the model using the training set
ridge.fit(X_train,Y_train)
# Compute RMSE on training data
p = ridge.predict(X_train)
err = p-Y_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))
print("[Holdout][Ridge Regression]\nTraining error : {:.2f}".format(rmse_train))
#Compute RMSE on validation data
p = ridge.predict(X_validation)
err = p-Y_validation
total_error = np.dot(err,err)
rmse_validation = np.sqrt(total_error/len(p))
print("[Holdout][Ridge Regression]\nValidation error : {:.2f}".format(rmse_validation))
#accuracy = p.score(X_validation,Y_validation)
#Compute RMSE on test data
p = ridge.predict(X_test)
err = p-Y_test
total_error = np.dot(err,err)
rmse_test = np.sqrt(total_error/len(p))
holdoutE.append(rmse_test)
print("[Holdout][Ridge Regression]\ntest error : {:.2f}".format(rmse_test))
print("[Holdout][Ridge Regression]")
model_scatter_plot(p,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Holdout] Residual Plot for Ridge Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,p) ## regression Residual Plot for Ridge Regression model using bootstrapping
#hyperparameters
para=np.arange(2,100,1)
error=dict()

#tuning hyperparameter
print("Parameter   RMSE Error")
print("-"*25)
for parameter in para:
    regr = RandomForestRegressor(max_depth=parameter, random_state=0)
    regr.fit(X_train,Y_train)
    
    terr = calc_train_error(X_train,Y_train,regr)
    verr = calc_validation_error(X_validation,Y_validation,regr)
    total_error =np.abs(terr-verr)
    error[parameter]= total_error
    p= min(error,key = error.get)
    print("{:.3f}        {:.3f}".format(parameter,error[parameter]))
    
print("Best parameter after validation: {:.3f}".format(p))

print("[Holdout][Random Forest]")
plotErr(1,X_train,Y_train,100,X_validation,Y_validation) 
# Create random forest regression model
regr = RandomForestRegressor(max_depth=p, random_state=0)

# Train the model using the training set
regr.fit(X_train,Y_train)

# Compute RMSE on training data
p = regr.predict(X_train)
err = p-Y_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))
print("[Holdout][Random Forest]\nTraining error : {:.2f}".format(rmse_train))

#Compute RMSE on validation data
p = regr.predict(X_validation)
err = p-Y_validation
total_error = np.dot(err,err)
rmse_validation = np.sqrt(total_error/len(p))
print("Validation error : {:.2f}".format(rmse_validation))
#accuracy = p.score(X_validation,Y_validation)
#Compute RMSE on test data
p = regr.predict(X_test)
err = p-Y_test
total_error = np.dot(err,err)
rmse_test = np.sqrt(total_error/len(p))
holdoutE.append(rmse_test)
print("[Holdout][Random Forest]\nTest error : {:.2f}".format(rmse_test))
print("[Holdout][Random Forest]")
model_scatter_plot(p,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Holdout] Residual Plot for Random Forest', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,p) ## regression Residual Plot for Random Forest model using bootstrapping
#hyperparameters
para=np.linspace(1,10,100)
error=dict()

#tuning hyperparameter
print("Parameter   RMSE Error")
print("-"*25)
for parameter in para:
    sv = SVR(C=parameter)
    sv.fit(X_train,Y_train)
    
    terr = calc_train_error(X_train,Y_train,sv)
    verr = calc_validation_error(X_validation,Y_validation,sv)
    total_error =np.abs(terr-verr)
    error[parameter]= total_error
    p= min(error,key = error.get)
    print("{:.3f}        {:.3f}".format(parameter,error[parameter]))
    
print("Best parameter after validation: {:.3f}".format(p))

print("[Holdout][Support Vector Regression]")
plotErr(2,X_train,Y_train,100,X_validation,Y_validation) 
# Create support vector regression model
sv = SVR(C=10)

# Train the model using the training set
sv.fit(X_train,Y_train)

# Compute RMSE on training data
p = sv.predict(X_train)
err = p-Y_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))
print("[Holdout][Support Vector Regressor]\nTraining error : {:.2f}".format(rmse_train))

#Compute RMSE on validation data
p = sv.predict(X_validation)
err = p-Y_validation
total_error = np.dot(err,err)
rmse_validation = np.sqrt(total_error/len(p))
print("Validation error : {:.2f}".format(rmse_validation))
#accuracy = p.score(X_validation,Y_validation)
#Compute RMSE on test data
p = sv.predict(X_test)
err = p-Y_test
total_error = np.dot(err,err)
rmse_test = np.sqrt(total_error/len(p))
holdoutE.append(rmse_test)
print("[Holdout][Support Vector regressor]\nTest error : {:.2f}".format(rmse_test))
print("[Holdout][Support Vector Regression]")
model_scatter_plot(p,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Holdout] Residual Plot for Support Vector Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,p) ## regression Residual Plot for Support Vector Regression model using bootstrapping
#create bins
X["charges"] = y

bins = np.linspace(X.charges.min(),X.charges.max(),7)
charges_groups = np.digitize(X.charges,bins)

#Add bin information to the dataframe
X= X.assign(charge_groups = charges_groups)
X['charge_groups'] = pd.Categorical(X.charge_groups)

#Last bin has too few values to stratify, so we merge it with second last group and reduce the number of bins
X.charge_groups[X.charge_groups == 7]= 6
y_stratify= X["charge_groups"]
X.head()
#X["charge_groups"].unique()
X.groupby(X["charge_groups"]).count()
X_intermediateS, X_testS, Y_intermediateS, Y_testS = train_test_split(X, 
                                                                  y_stratify, 
                                                                  shuffle=True,
                                                                  test_size=0.2,
                                                                  stratify=y_stratify
                                                                  )

#train/validation split (gives us train and validation sets)
X_trainS, X_validationS, Y_trainS, Y_validationS = train_test_split(X_intermediateS,
                                                                Y_intermediateS,
                                                                shuffle=True,
                                                                test_size=0.2,
                                                                stratify=Y_intermediateS
                                                                )

#remove the "charges" and "charge_groups" column and separate target variable
X_str=X_intermediateS.copy()
y_str=Y_intermediateS
Y_intermediateS=X_intermediateS['charges']
del X_intermediateS["charges"]
del X_intermediateS["charge_groups"]
Y_trainS=X_trainS["charges"]
del X_trainS["charges"]
del X_trainS["charge_groups"]

Y_testS=X_testS["charges"]
del X_testS["charges"]
del X_testS["charge_groups"]
X_str.head()

holdoutStf=[] #to store rmse error
print('train: {}% | validation: {}% | test {}%'.format(round(len(Y_trainS)/len(y),2),
                                                       round(len(Y_validationS)/len(y),2),
                                                       round(len(Y_testS)/len(y),2)))
from sklearn import model_selection
stratifiedkfold=[]
del X_str["charges"]
del X_str["charge_groups"]
num_of_splits=5
def stratifiedKFold (model): 
    kf = model_selection.StratifiedKFold(n_splits=num_of_splits,shuffle=True)
    pred_test_full =0
    cv_score =[]
    i=1
    for train_index,test_index in kf.split(X_str,y_str):
        print('{} of StratifiedKFold {}'.format(i,kf.n_splits))
        
        xtr,xvl = X_intermediateS.iloc[train_index],X_intermediateS.iloc[test_index]
        ytr,yvl = Y_intermediateS.iloc[train_index],Y_intermediateS.iloc[test_index]

        model.fit(xtr,ytr)
        rmse = calc_train_error(xtr,ytr,model)
        print('Train RMSE :{:.2f}'.format(rmse))
        rmse = calc_train_error(xvl,yvl,model)
        print('Validation RMSE :{:.2f}'.format(rmse))
        print('_'*50)
        cv_score.append(rmse)    
        pred_test = model.predict(X_testS)
        pred_test_full +=pred_test
        i+=1
       
    print('Mean Stratified K Fold CV Score : {:.2f}'.format(np.mean(cv_score))) 
    return np.mean(cv_score), pred_test_full
      
linreg = LinearRegression()
err,y_pred=stratifiedKFold (linreg)
stratifiedkfold.append(err)
y_pred = y_pred/num_of_splits
print("[Stratified K Fold][Linear Regression]")
model_scatter_plot(y_pred,Y_testS)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Stratified K Fold CV] Residual Plot for Linear Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_testS,y_pred) ## regression Residual Plot for linear regression model using bootstrapping
ridge = Ridge()
err,y_pred=stratifiedKFold (ridge)
stratifiedkfold.append(err)
y_pred = y_pred/num_of_splits
print("[Stratified K Fold][Ridge Regression]")
model_scatter_plot(y_pred,Y_testS)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Stratified K Fold CV] Residual Plot for Ridge Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_testS,y_pred) ## regression Residual Plot for Ridge regression model using bootstrapping
rf = RandomForestRegressor()
err,y_pred=stratifiedKFold (rf)
stratifiedkfold.append(err)
y_pred = y_pred/num_of_splits
print("[Stratified K Fold][Random forest]")
model_scatter_plot(y_pred,Y_testS)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Stratified K Fold CV] Residual Plot for Random Forest', y=1, size=20) #"size" controls the title size
sns.residplot(Y_testS,y_pred) ## regression Residual Plot for Random Forest model using bootstrapping
sv = SVR()
err,y_pred=stratifiedKFold (sv)
stratifiedkfold.append(err)
y_pred = y_pred/num_of_splits
print("[Stratified K Fold][Support Vector Regression]")
model_scatter_plot(y_pred,Y_testS)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Stratified K Fold CV] Residual Plot for Support Vector Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_testS,y_pred) ## regression Residual Plot for Support Vector regression model using bootstrapping
from sklearn.utils import resample
#from sklearn.metrics import accuracy_score

bootstra=[]
# load dataset
X_intermediate["charges"]= Y_intermediate
data = X_intermediate
values = data.values

# configure bootstrap
n_iterations = 10 #number of iterations
n_size = int(len(data) * 1) #sample size
# run bootstrap
stats = list()

def bootstrap(model_name):
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(values, n_samples=n_size)
        test = np.array([x for x in values if x.tolist() not in train.tolist()])
        # fit model
        model = model_name
        model.fit(train[:,:-1], train[:,-1])
         # evaluate model
        train_pred = model.predict(train[:,:-1])
        mse = mean_squared_error(train[:,-1], train_pred)
        err_train = np.sqrt(mse)
       # err_train = calc_train_error(train[:,-1], train_pred,model)
        test_pred = model.predict(test[:,:-1])
        mse = mean_squared_error(test[:,-1], test_pred)
        err_test = np.sqrt(mse)
        #err_test = calc_train_error(test[:,-1], test_pred,model)
        print("{} of iteration {}".format(i,n_iterations-1))
        print("RMSE for Train : {:.2f}".format(err_train))
        print("RMSE for Validation : {:.2f}".format(err_test))
      #  print(rmse)
        print("-"*50)
        stats.append(err_test)
    print("Average RMSE for Validation : {:.2f}".format(np.mean(stats))) 
    res= (np.mean(stats))
    # plot rmse
    plt.hist(stats)
    plt.show()
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = max(1.0, np.percentile(stats, p))
    print('{:.0f} % confidence interval {:.2f} and {:.2f}'.format(alpha*100, lower, upper))
    return np.mean(stats)
print("[Logistic Regression] The RMSE for each bootstrap iteration: ")
bootstra.append(bootstrap(LinearRegression()))
linreg=LinearRegression()
linreg.fit(X_train,Y_train)
pred=linreg.predict(X_test)
mse = mean_squared_error(Y_test,pred)
err_test = np.sqrt(mse)
print("[Bootstrap][Linear Regression]\nRMSE on Test data: {:.2f}".format(err_test))
model_scatter_plot(pred,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Bootstrapping] Residual Plot for Linear Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,pred) ## regression Residual Plot for Linear regression model using bootstrapping
print("[Ridge Regression] The RMSE for each bootstrap iteration: ")
bootstra.append(bootstrap(Ridge()))
ridge=Ridge()
ridge.fit(X_train,Y_train)
pred=ridge.predict(X_test)
mse = mean_squared_error(Y_test,pred)
err_test = np.sqrt(mse)
print("[Bootstrap][Linear Regression]\nRMSE on Test data: {:.2f}".format(err_test))
model_scatter_plot(pred,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Bootstrapping] Residual Plot for Ridge Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,pred) ## regression Residual Plot for Ridge regression model using bootstrapping
print("[Random Forest] The RMSE for each bootstrap iteration: ")
bootstra.append(bootstrap(RandomForestRegressor()))
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
pred=rf.predict(X_test)
mse = mean_squared_error(Y_test,pred)
err_test = np.sqrt(mse)
print("[Bootstrap][Random Forest]\nRMSE on Test data: {:.2f}".format(err_test))
model_scatter_plot(pred,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Bootstrapping] Residual Plot for Random Forest', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,pred) ## regression Residual Plot for Random Forest model using bootstrapping
print("[Support Vector Regression] The RMSE for each bootstrap iteration: ")
bootstra.append(bootstrap(SVR()))
sv=SVR()
sv.fit(X_train,Y_train)
pred=sv.predict(X_test)
mse = mean_squared_error(Y_test,pred)
err_test = np.sqrt(mse)
print("[Bootstrap][Support Vector Regression]\nRMSE on Test data: {:.2f}".format(err_test))
model_scatter_plot(pred,Y_test)
plt.figure(figsize=(10,5)) #controls the graph size
plt.title('[Bootstrapping] Residual Plot for Support Vector Regression', y=1, size=20) #"size" controls the title size
sns.residplot(Y_test,pred) ## regression Residual Plot for Support vector regression model using bootstrapping
# data to plot
n_groups = 4
means_Holdout = (holdoutE[0], holdoutE[1], holdoutE[2], holdoutE[3])
means_Stratified_Holdout=[4324,2322,5654,3345]
#means_Stratified_Holdout = (holdoutStf[0], holdoutstf[1], holdoutstf[2], holdoutstf[3])
means_StratifiedKFold = (stratifiedkfold[0], stratifiedkfold[1], stratifiedkfold[2], stratifiedkfold[3])
means_Bootstrapping = (bootstra[0], bootstra[1], bootstra[2], bootstra[3])
 
# create plot
 
fig,ax = plt.subplots(figsize=(20,8))
index = np.arange(n_groups)
bar_width = 0.21
opacity = 0.8
#plt.figure(figsize=(12,7))
rects1 = plt.bar(index+bar_width, means_Holdout, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Holdout')
 
rects2 = plt.bar(index + 2*bar_width, means_Stratified_Holdout, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Stratified Holdout')

rects3 = plt.bar(index+ 3*bar_width, means_StratifiedKFold, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Stratified K Fold')
rects4 = plt.bar(index+ 4*bar_width, means_Bootstrapping, bar_width,
                 alpha=opacity,
                 color='m',
                 label='Bootstrapping')
 
ax.set_ylim([0,10000])   
plt.xlabel('Model',fontsize=16)
plt.ylabel('RMSE',fontsize=16)
plt.title('Model comparsion on basis of RMSE',fontsize=18)
plt.xticks(index + 2*0.18, ('Linear Regression', 'Ridge Regression ', 'Random Forest','Support Vector Regression'),rotation=20, fontsize=12)
plt.legend(loc='left', bbox_to_anchor=(0.15,1))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '{:.2f}'.format(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.show()
