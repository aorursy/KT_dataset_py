import numpy as np
import pandas as pd  # To read data
import matplotlib.pyplot as plt  # To visualize
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
data = pd.read_csv('../input/insurance.csv')
# show the first 10 lines
data.head(10)
data.describe()
categorical_summaries = [data[c].value_counts() for c in data.columns if data[c].dtype == 'object']

for i in categorical_summaries:
    display(pd.DataFrame(i))
def plot_numeric_distribution(colnumber, plot_type='histogram', data=data):
    """
    Function for plotting histogram of the column number
    corresponding to the numerical variable selected, 
    or scatter plot of final target var against numerical variable
    """
    
    # if scatter, plot scatter plot with final variable as y axis
    if plot_type=='scatter':
        # x label for the plot
        plt.xlabel(data.columns[colnumber])
        plt.ylabel(data.columns[-1])
        plt.title(data.columns[-1] + ' distribution depending on ' + data.columns[colnumber])
        plt.scatter(data.iloc[:,colnumber], data.iloc[:,-1],marker='x', alpha=0.5)
       
    # else if boxplot, plot boxplots with final variable as y axis
    elif plot_type=='boxplot':
        data2=data
        data2[data2.columns[colnumber]]=data2[data2.columns[colnumber]].astype('category',copy=False)
        sns.set(style="ticks", color_codes=True)
        sns.catplot(x=data2.columns[colnumber], y=data2.columns[-1], kind='box',data=data2);
        plt.title(data2.columns[-1] + ' distribution depending on ' + data2.columns[colnumber])        
        
    # otherwise plot histogram of the column 
    elif plot_type=='histogram':
        # x label for the plot
        plt.xlabel(data.columns[colnumber])
        plt.ylabel('Frequency')
        plt.title(data.columns[colnumber] + ' distribution')
        data.iloc[:,colnumber].plot.hist()

plot_numeric_distribution(-1)
plot_numeric_distribution(0,'boxplot')
plot_numeric_distribution(0, 'scatter')
plot_numeric_distribution(2)
plot_numeric_distribution(2, 'scatter')
plot_numeric_distribution(3)
plot_numeric_distribution(3, 'scatter')
plot_numeric_distribution(3, 'boxplot')
plot_numeric_distribution(-1)
# specify which features you want to transform to be more Gaussian
non_normal_features = [0,3,6]

# load the powertransformer tool
power_transformer = preprocessing.PowerTransformer(standardize=False)

# fit the model and immediately use it to transform your data
data.iloc[:,non_normal_features] = power_transformer.fit_transform(data.iloc[:,non_normal_features])
plot_numeric_distribution(6)
data.describe()
scaler = preprocessing.StandardScaler()
data[data.columns[data.dtypes!='object']] = scaler.fit_transform(data[data.columns[data.dtypes!='object']])
data.head()
def categorical_distribution (colnumber,plot_type):
    """
    Function for plotting histogram of the column number
    corresponding to the categorical variable selected against the final target, 
    or boxplot of final target var against categorical variable.
    """    
    
    if plot_type=='histogram':
        # x, y and title labels
        plt.xlabel(data.columns[colnumber])
        plt.ylabel('Frequency')
        plt.title(data.columns[colnumber] + ' distribution')
        data.iloc[:,colnumber].value_counts().plot(kind='bar')
        
    elif plot_type=='boxplot':
        # setting type of plot
        sns.set(style="ticks", color_codes=True)
        # setting what values we plot
        sns.catplot(x=data.columns[colnumber], y=data.columns[-1], kind='box',data=data);
        #title
        plt.title(data.columns[-1] + ' distribution depending on ' + data.columns[colnumber])        
categorical_distribution(1,'histogram')
categorical_distribution(1,'boxplot')
categorical_distribution(4,'histogram')
categorical_distribution(4,'boxplot')
categorical_distribution(5,'histogram')
categorical_distribution(5,'boxplot')
# 1,4 and 5 are the categorical variables index
dummy= pd.get_dummies(data.iloc[:,[1,4,5]])
dummy.head()
# we select the numerical variables (0, 2 and 3 are the index)
num_data=data.iloc[:,[0,2,3]]

# we select the output variable
y=data.iloc[:,-1]

# we concatenate the numerical variables, to the dummy variables and the ouput variable 
data=pd.concat([num_data,dummy,y],axis=1)

# first 10 lines of the final dataset
data.head(10)
# values visualization
data.corr()
# graphical visualization
f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16, y=1.11); 
data=data.drop(['sex_male','smoker_no'], axis=1)
data.head()
# create training and testing variables. test_size describes what proportion 
# of the initial set we want for our test set
# you can select the proportion you want
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
print("R2_score="+str(round(regressor.score(X_train, y_train), 4)))
# To retrieve the intercept:
print("intercept = " + str(round(regressor.intercept_, 4)))
# For retrieving the slope:
for i in range(0,len(regressor.coef_)):
    print(data.columns[i]+ " coefficient = " + str(round(regressor.coef_[i], 4)))
# prediction
y_pred = regressor.predict(X_test)

# comparison of the predictions to the actual values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(10)
# 3 accuracy scores
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 4))  
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("R2_score="+str(round(ridge.score(X_train, y_train), 4)))
print("intercept = " + str(round(ridge.intercept_, 4)))

# print the coefficients
for i in range(0,len(regressor.coef_)):
    print(data.columns[i]+ " coefficient = " + str(round(ridge.coef_[i], 4)))
# predictions
y_pred = ridge.predict(X_test)

# comparison of the predictions to the actual values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(10)
# 3 accuracy scores
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 4))  
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("R2_score="+str(round(lasso.score(X_train, y_train), 4)))
# plot the intercept
print("intercept = "+ str(round(lasso.intercept_, 4)))
# plot the slope:
for i in range(0,len(regressor.coef_)):
    print(data.columns[i]+ " coefficient = " + str(round(lasso.coef_[i], 4)))
#predictions
y_pred = lasso.predict(X_test)

# comparison of the predictions to the actual values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(10)
# 3 accuracy scores
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 4))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))
def GridSearch (input_train, output_train, input_test, output_test, function, alpha):
    """
    Function taking: - training and test data
                     - ML model (function)
                     - alpha values to be tested
    and plotting the grid search values, the best model and the accuracy of the model on the test  
    """   
    # we set the alpha parameters we want to try
    params = {'alpha':alpha, 
            'fit_intercept':[False,True]}
    
    # we do the gridseach, it will select the parameter giving the best result
    GS_models = GridSearchCV(function,
                            param_grid=params,
                            scoring='explained_variance',cv=5).fit(input_train, output_train)
    
    # we stock the scores
    scores = GS_models.cv_results_.get('mean_test_score')
    
    # we initialize the vectors stocking the values for the plot
    scores_true=[]
    scores_false=[]
    alphas_true=[]
    alphas_false=[]
    
    # we loop all the the hyperparameter combinations
    for i in range(0,len(scores)) :
        GS_models.cv_results_['params'][i].get('alpha')
        
        # if intercept true we stock the values in the true vectors
        if GS_models.cv_results_['params'][i].get('fit_intercept'):
            scores_true.append(scores[i])
            alphas_true.append(GS_models.cv_results_['params'][i].get('alpha'))
        else:
            scores_false.append(scores[i])
            alphas_false.append(GS_models.cv_results_['params'][i].get('alpha'))
    # plot         
    plt.plot(alphas_true, scores_true, 'bx',label='Intercept true')
    plt.plot(alphas_false, scores_false, 'r*',label='Intercept false')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('R2')
    plt.show()
    
    print("The best parameters are the ones maximizing the R2_score")
    print("Best model description:")
    # best parameter
    print(    GS_models.best_params_)
    # best score
    print("   R2_score= " + str(GS_models.best_score_))
    
    # 3 accuracy scores
    print('Accuracy results on the test set')
    print('   Mean Absolute Error=', round(metrics.mean_absolute_error(y_test, y_pred), 4))
    print('   Mean Squared Error=',  round(metrics.mean_squared_error(y_test, y_pred), 4))  
    print('   Root Mean Squared Error=',  round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4))
    return GS_models
alpha=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lasso_GS_models=GridSearch (X_train, y_train, X_test, y_test, linear_model.Lasso(), alpha)
# predictions
y_pred = lasso_GS_models.predict(X_test)

# comparison of the predictions to the actual values
print("Predictions comparison:")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head()
alpha=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ridge_GS_models=GridSearch (X_train, y_train, X_test, y_test, linear_model.Ridge(), alpha)
# predictions
y_pred = ridge_GS_models.predict(X_test)

# comparison of the predictions to the actual values
print("Predictions comparison:")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head()
def RandomSearch(input_train, output_train, input_test, output_test, function):
    """
    Function taking: - training and test data
                     - ML model (function)
    and plotting the random search values, the best model and the accuracy of the model on the test  
    """   
    params_grid= {'alpha': sp_rand() , 'fit_intercept':[False,True]}
    
    # create and fit a ridge regression model, testing random alpha values
    rsearch = RandomizedSearchCV(estimator=function, 
                                   param_distributions=params_grid, 
                                   scoring='explained_variance',
                                   n_iter=100, cv=5)
    rsearch.fit(input_train, output_train)
    
    # get scores
    scores=np.array(rsearch.cv_results_.get('mean_test_score'))

    # get parameters
    params=rsearch.cv_results_.get('params')

    # alpha parameters tried
    alpha_params=np.array([params[i].get('alpha') for i in range(0,len(params))])

    # intercept parameters tried
    intercept_params=np.array([params[i].get('fit_intercept') for i in range(0,len(params))])

    # get scores where intercept is True
    with_int=np.where(intercept_params == True)[0]
    # get scores where intercept is False
    no_int=np.where(intercept_params == False)[0]

    # plotting scores depending on alpha parameter where intercept is True
    plt.plot(alpha_params[with_int], scores[with_int],'x',label='fit intercept: ' + 'True')
    # plotting scores depending on alpha parameter where intercept is False
    plt.plot(alpha_params[no_int], scores[no_int],'x',label='fit intercept: ' + 'False')

    # setting legends
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('R2')
    plt.show()
    
    # summarize the results of the random parameter search
    print("The best parameters are the ones maximizing the R2_score")
    print("Best model description")
    print("   R2_score= " + str(rsearch.best_score_))
    print(   rsearch.best_params_)
    
    # predictions
    print("Predictions accuracy:")
    print('   Mean Absolute Error= ', metrics.mean_absolute_error(output_test, y_pred))  
    print('   Mean Squared Error= ', metrics.mean_squared_error(output_test, y_pred))  
    print('   Root Mean Squared Error= ', np.sqrt(metrics.mean_squared_error(output_test, y_pred)))
    
    return rsearch
lasso_rsearch=RandomSearch (X_train, y_train, X_test, y_test, linear_model.Lasso())
# predictions
y_pred = lasso_rsearch.predict(X_test)

# comparison of the predictions to the actual values
print("Predictions comparison:")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head()
ridge_rsearch=RandomSearch (X_train, y_train, X_test, y_test, linear_model.Ridge())
# predictions
y_pred = ridge_rsearch.predict(X_test)

# comparison of the predictions to the actual values
print("Predictions comparison:")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head()
df_sum = pd.DataFrame(columns = ['Model','R2_score','Mean Absolute error','Mean Squared Error'])
df_sum.loc[0]=['Linear regression',regressor.score(X_train, y_train),metrics.mean_absolute_error(y_test, regressor.predict(X_test)),metrics.mean_squared_error(y_test, regressor.predict(X_test))]
df_sum.loc[1]=['Lasso regression',lasso.score(X_train, y_train),metrics.mean_absolute_error(y_test, lasso.predict(X_test)), metrics.mean_squared_error(y_test, lasso.predict(X_test))]
df_sum.loc[2]=['Ridge regression',ridge.score(X_train, y_train),metrics.mean_absolute_error(y_test, ridge.predict(X_test)),metrics.mean_squared_error(y_test, ridge.predict(X_test))]
df_sum.loc[3]=['Lasso regression grid search',lasso_GS_models.score(X_train, y_train),metrics.mean_absolute_error(y_test, lasso_GS_models.predict(X_test)),metrics.mean_squared_error(y_test, lasso_GS_models.predict(X_test))]
df_sum.loc[4]=['Ridge regression grid search',ridge_GS_models.score(X_train, y_train),metrics.mean_absolute_error(y_test, ridge_GS_models.predict(X_test)),metrics.mean_squared_error(y_test, ridge_GS_models.predict(X_test))]
df_sum.loc[5]=['Lasso regression random search',lasso_rsearch.score(X_train, y_train),metrics.mean_absolute_error(y_test, lasso_rsearch.predict(X_test)),metrics.mean_squared_error(y_test, lasso_rsearch.predict(X_test))]
df_sum.loc[6]=['Ridge regression random search',ridge_rsearch.score(X_train, y_train),metrics.mean_absolute_error(y_test, ridge_rsearch.predict(X_test)),metrics.mean_squared_error(y_test, ridge_rsearch.predict(X_test))]
df_sum.loc[7]=['Baseline model',0,metrics.mean_absolute_error(y_test,np.full(np.shape(y_test), np.mean(y_train))), metrics.mean_squared_error(y_test,np.full(np.shape(y_test), np.mean(y_train)))]
round(df_sum, 4)