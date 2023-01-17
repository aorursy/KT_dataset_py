import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline  

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor



from sklearn import model_selection #import cross_val_score, StratifiedKFold

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz

from sklearn import metrics  # mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score

from sklearn.feature_selection import SelectFromModel, RFECV

from sklearn.metrics import max_error

from sklearn.decomposition import PCA

from sklearn import preprocessing
#dataset column names:



col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23']

#load training data



df_train_raw = pd.read_csv('../input/PM_train.txt', sep = ' ', header=None)

df_train_raw.head()


#assign column names



df_train_raw.columns = col_names

df_train_raw.head()

df_train_raw.id.unique()
#drop extra space columnn



df_train_raw=df_train_raw.drop(columns=['s22','s23'])

df_train_raw.head()
# get some stat



df_train_raw.describe()
# check the data types



df_train_raw.dtypes
df_train_raw.isnull().sum()
#load test data



df_test_raw = pd.read_csv('../input/PM_test.txt', sep = ' ', header=None)

df_test_raw.head()
# #drop extra space columnn

# df_test_raw.drop([26,27], axis=1, inplace='True')



#assign column names

df_test_raw.columns = col_names

df_test_raw.head()

#drop extra space columnn



df_test_raw=df_test_raw.drop(columns=['s22','s23'])
# get some stat on test data



df_test_raw.describe()
df_test_raw.isnull().sum()
# Load the truth data - actual 'ttf' for test data



df_truth = pd.read_csv('../input/PM_truth.txt', sep = ' ', header=None)

df_truth.head()
#drop extra empty column in the truth data and rename remaining 'ttf'



# df_truth.drop([1], axis=1, inplace='True')

df_truth.columns = ['ttf','1']

df_truth=df_truth.drop(columns=['1'])

df_truth.head()
#get some stat on truth data



df_truth.describe()


def add_features(df_in, rolling_win_size):

    

    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.

    

    Args:

            df_in (dataframe)     : The input dataframe to be proccessed (training or test) 

            rolling_win_size (int): The window size, number of cycles for applying the rolling function

        

    Reurns:

            dataframe: contains the input dataframe with additional rolling mean and std for each sensor

    

    """

    

    sensor_cols = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

    

    sensor_av_cols = [nm.replace('s', 'av') for nm in sensor_cols]

    sensor_sd_cols = [nm.replace('s', 'sd') for nm in sensor_cols]

    

    df_out = pd.DataFrame()

    

    ws = rolling_win_size

    

    #calculate rolling stats for each engine id

    

    for m_id in pd.unique(df_in.id):

    

        # get a subset for each engine sensors

        df_engine = df_in[df_in['id'] == m_id]

        df_sub = df_engine[sensor_cols]



    

        # get rolling mean for the subset

        av = df_sub.rolling(ws, min_periods=1).mean()

        av.columns = sensor_av_cols

    

        # get the rolling standard deviation for the subset

        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)

        sd.columns = sensor_sd_cols

    

        # combine the two new subset dataframes columns to the engine subset

        new_ftrs = pd.concat([df_engine,av,sd], axis=1)

    

        # add the new features rows to the output dataframe

        df_out = pd.concat([df_out,new_ftrs])

        

    return df_out


def prepare_train_data (df_in, period):

    

    """Add regression and classification labels to the training data.



        Regression label: ttf (time-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine

        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)

        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

        

      Args:

          df_in (dataframe): The input training data

          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

          

      Returns:

          dataframe: The input dataframe with regression and classification labels added

          

    """

    

    #create regression label

    

    #make a dataframe to hold the last cycle for each enginge in the dataset

    df_max_cycle = pd.DataFrame(df_in.groupby('id')['cycle'].max())

    df_max_cycle.reset_index(level=0 , inplace=True)

    df_max_cycle.columns = ['id', 'last_cycle']



    #add time-to-failure ttf as a new column - regression label

    df_in = pd.merge(df_in, df_max_cycle, on='id')

    df_in['ttf'] = df_in['last_cycle'] - df_in['cycle']

    df_in.drop(['last_cycle'], axis=1 , inplace=True)

    

#     #create binary classification label

#     df_in['label_bnc'] = df_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    

#     #create multi-class classification label

#     df_in['label_mcc'] = df_in['ttf'].apply(lambda x: 2 if x <= period/2 else 1 if x <= period else 0)

    

    return df_in

    
df_max_cycle = pd.DataFrame(df_train_raw.groupby('id')['cycle'].max())

df_max_cycle.reset_index(level=0 , inplace=True)

df_max_cycle.columns = ['id', 'last_cycle']

df_max_cycle.describe()
df_max_cycle
df_max_cycle.describe()
# https://github.com/Samimust/predictive-maintenance

def prepare_test_data(df_test_in, df_truth_in, period):

    

    """Add regression and classification labels to the test data.



        Regression label: ttf (time-to-failure) = extract the last cycle for each enginge and then merge the record with the truth data

        Binary classification label: label_bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)

        Multi-class classification label: label_mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2

        

      Args:

          df_in (dataframe): The input training data

          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels

          

      Returns:

          dataframe: The input dataframe with regression and classification labels added

    



    

    """

    

    df_tst_last_cycle = pd.DataFrame(df_test_in.groupby('id')['cycle'].max())

    

    df_tst_last_cycle.reset_index(level=0, inplace=True)

    df_tst_last_cycle.columns = ['id', 'last_cycle']

#     , inplace=True

    df_test_in = pd.merge(df_test_in, df_tst_last_cycle, on='id')





    df_test_in = df_test_in[df_test_in['cycle'] == df_test_in['last_cycle']]



    df_test_in.drop(['last_cycle'], axis=1, inplace=True)

    

    df_test_in.reset_index(drop=True, inplace=True)

    

    df_test_in = pd.concat([df_test_in, df_truth], axis=1)

    

#     #create binary classification label

#     df_test_in['label_bnc'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)

    

#     #create multi-class classification label

#     df_test_in['label_mcc'] = df_test_in['ttf'].apply(lambda x: 2 if x <= period/2 else 1 if x <= period else 0)



    return df_test_in
# add extracted features to training data



df_train_fx = add_features(df_train_raw, 5)

df_train_fx.head()
#add labels to training data using period of 30 cycles for classification



df_train = prepare_train_data (df_train_fx, 30)

df_train.head()
df_train.describe()
x=df_train[df_train.ttf==30]

y=x[x.id<10]

y
df_train.dtypes
# save the training data to csv file for later use



df_train.to_csv('train.csv', index=False)
# add extracted features to test data



df_test_fx = add_features(df_test_raw, 5)

df_test_fx.head()
df_test_fx['id'].value_counts()
# df_test_fx=df_test_fx[df_test_fx.id==1]

# df_test_fx=df_test_fx[df_test_fx.cycle==31]

# df_test_fx
#add labels to test data using period of 30 cycles for classification

df_test = prepare_test_data(df_test_fx, df_truth, 30)

df_test.head()
df_test.dtypes
# save the test data to csv file for later use



df_test.to_csv('test.csv', index=False)
df_tr_lbl = pd.read_csv('train.csv')

df_tr_lbl.head(10)

df_tr_lbl.info()
#exclude enging id and cycle number from the input features:



featurs = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
#plot and compare the standard deviation of input features:



df_tr_lbl[featurs].std().plot(kind='bar', figsize=(10,6), title="Features Standard Deviation")
df_tr_lbl[featurs].std().sort_values(ascending=False)
# get ordered list features correlation with regression label ttf

df_tr_lbl[featurs].corrwith(df_tr_lbl.ttf).sort_values(ascending=False)
df_tr_lbl[df_tr_lbl.s5==14.62]["s5"].value_counts()
df_tr_lbl["s5"].std()
# list of features having low or no correlation with regression label ttf and very low or no variance

# These features will be target for removal in feature selection

low_cor_featrs = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting2', 'setting1']

df_tr_lbl[low_cor_featrs].describe()
# list of features having high correlation with regression label ttf



correl_featurs = ['s12', 's7', 's21', 's20', 's6', 's14', 's9', 's13', 's8', 's3', 's17', 's2', 's15', 's4', 's11']



df_tr_lbl[correl_featurs].describe()
# add the regression label 'ttf' to the list of high corr features 



correl_featurs_lbl = correl_featurs + ['ttf']

correl_featurs_lbl
df_tr_lbl[correl_featurs_lbl].corr()
# # most correlated features

# corrmat = df_tr_lbl.corr()

# top_corr_features = corrmat.index[abs(corrmat["ttf"])>0.6]

# plt.figure(figsize=(20,10))

# g = sns.heatmap(df_tr_lbl[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plot a heatmap to display +ve and -ve correlation among features and regression label:



import seaborn as sns

cm = np.corrcoef(df_tr_lbl[correl_featurs_lbl].values.T)

sns.set(font_scale=1.0)

fig = plt.figure(figsize=(10, 8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=correl_featurs_lbl, xticklabels=correl_featurs_lbl)

plt.title('Features Correlation Heatmap')

plt.show()
def explore_col(s):

    

    """Plot 4 main graphs for a single feature.

    

        plot1: histogram 

        plot2: boxplot 

        plot3: line plot (time series over cycle)

        plot4: scatter plot vs. regression label ttf

        

    Args:

        s (str): The column name of the feature to be plotted.

        e (int): The number of random engines to be plotted for plot 3. Range from 1 -100, 0:all engines, >100: all engines.



    Returns:

        plots

    

    """

    

    fig = plt.figure(figsize=(10, 8))





    sub1 = fig.add_subplot(221) 

    sub1.set_title(s +' histogram') 

    sub1.hist(df_tr_lbl[s])



    sub2 = fig.add_subplot(222)

    sub2.set_title(s +' boxplot')

    sub2.boxplot(df_tr_lbl[s])

    

    sub3 = fig.add_subplot(224)

    sub3.set_title("scatter: "+ s + " /ttf (regr label)")

    sub3.set_xlabel('ttf')

    sub3.scatter(df_tr_lbl['ttf'],df_tr_lbl[s])

    plt.tight_layout()

    plt.show()
explore_col("s12")
explore_col("av11")
df_test = pd.read_csv('test.csv')

df_test
dt=df_test["cycle"]+df_test["ttf"]

dt.describe()
df_test.describe()
#Prepare data for regression model



# original features

features_orig = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']



# original + extracted fetures

features_adxf = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9', 'av10', 'av11', 'av12', 'av13', 'av14', 'av15', 'av16', 'av17', 'av18', 'av19', 'av20', 'av21', 'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9', 'sd10', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd16', 'sd17', 'sd18', 'sd19', 'sd20', 'sd21']



# features with low or no correlation with regression label

features_lowcr = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting1', 'setting2']



# features that have correlation with regression label

features_corrl = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20','s21']

# features_impor = ['s4', 's7', 's9', 's11', 's12']

# a variable to hold the set of features to experiment with



features = features_corrl

X_train = df_train[features]

y_train = df_train['ttf']



X_test = df_test[features]

y_test = df_test['ttf']

# from sklearn.preprocessing import MinMaxScaler

# min_max_scaler = preprocessing.MinMaxScaler()

# X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train),columns=features_corrl)

# X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test),columns=features_corrl )
def get_regression_metrics(model, actual, predicted):

    

    """Calculate main regression metrics.

    

    Args:

        model (str): The model name identifier

        actual (series): Contains the test label values

        predicted (series): Contains the predicted values

        

    Returns:

        dataframe: The combined metrics in single dataframe

    

    

    """

    x= np.mean(np.abs((actual - predicted) / actual)) * 100

    regr_metrics = {

                        'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,

                        'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),

                        'R^2' : metrics.r2_score(actual, predicted),

                        'Explained Variance' : metrics.explained_variance_score(actual, predicted),

                        'Max Error' : metrics.max_error(actual, predicted),

                        'Mean absolute percentage error': x

               

                   }



    #return reg_metrics

    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')

    df_regr_metrics.columns = [model]

    return df_regr_metrics
def plot_features_weights(model, weights, feature_names, weights_type='c'):

    

    """Plot regression coefficients weights or feature importance.

    

    Args:

        model (str): The model name identifier

        weights (array): Contains the regression coefficients weights or feature importance

        feature_names (list): Contains the corresponding features names

        weights_type (str): 'c' for 'coefficients weights', otherwise is 'feature importance'

        

    Returns:

        plot of either regression coefficients weights or feature importance

        

    

    """

    (px, py) = (8, 10) if len(weights) > 30 else (8, 5)

    W = pd.DataFrame({'Weights':weights}, feature_names)

    W.sort_values(by='Weights', ascending=True).plot(kind='barh', color='r', figsize=(px,py))

    label = ' Coefficients' if weights_type =='c' else ' Features Importance'

    plt.xlabel(model + label)

    plt.gca().legend_ = None
def plot_residual(model, y_train, y_train_pred, y_test, y_test_pred):

    

    """Print the regression residuals.

    

    Args:

        model (str): The model name identifier

        y_train (series): The training labels

        y_train_pred (series): Predictions on training data

        y_test (series): The test labels

        y_test_pred (series): Predictions on test data

        

    Returns:

        Plot of regression residuals

    

    """

    

    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')

    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')

    plt.xlabel('Predicted Values')

    plt.ylabel('Residuals')

    plt.legend(loc='upper left')

    plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)

    plt.title(model + ' Residuals')

    plt.show()

    
#try linear regression



linreg = linear_model.LinearRegression()

linreg.fit(X_train, y_train)



y_test_predict = linreg.predict(X_test)

y_train_predict = linreg.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



linreg_metrics = get_regression_metrics('Linear Regression', y_test, y_test_predict)

linreg_metrics

plot_features_weights('Linear Regression', linreg.coef_, X_train.columns, 'c')
plot_residual('Linear Regression', y_train_predict, y_train, y_test_predict, y_test)
#try LASSO



lasso = linear_model.Lasso(alpha=0.001)

lasso.fit(X_train, y_train)



y_test_predict = lasso.predict(X_test)

y_train_predict = lasso.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



lasso_metrics = get_regression_metrics('LASSO', y_test, y_test_predict)



lasso_metrics
plot_features_weights('LASSO', lasso.coef_, X_train.columns, 'c')
plot_residual('LASSO', y_train_predict, y_train, y_test_predict, y_test)
#try ridge



rdg = linear_model.Ridge(alpha = 0.01)

rdg.fit(X_train, y_train)



y_test_predict = rdg.predict(X_test)

y_train_predict = rdg.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



rdg_metrics = get_regression_metrics('Ridge Regression', y_test, y_test_predict)

rdg_metrics
plot_features_weights('Ridge Regression', rdg.coef_, X_train.columns, 'c')
plot_residual('Ridge Regression', y_train_predict, y_train, y_test_predict, y_test)
#try Polynomial Regression



from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2)



X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.fit_transform(X_test)





polyreg = linear_model.LinearRegression()

polyreg.fit(X_train_poly, y_train)



y_test_predict = polyreg.predict(X_test_poly)

y_train_predict = polyreg.predict(X_train_poly)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



polyreg_metrics = get_regression_metrics('Polynomial Regression', y_test, y_test_predict)

polyreg_metrics
plot_residual('Polynomial Regression', y_train_predict, y_train, y_test_predict, y_test)
#try Decision Tree regressor



#dtrg = DecisionTreeRegressor(max_depth=8, max_features=5, random_state=123) # selected features

dtrg = DecisionTreeRegressor(max_depth=7, random_state=123)

dtrg.fit(X_train, y_train)



y_test_predict = dtrg.predict(X_test)

y_train_predict = dtrg.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



dtrg_metrics = get_regression_metrics('Decision Tree Regression', y_test, y_test_predict)
plot_features_weights('Decision Tree Regressor', dtrg.feature_importances_, X_train.columns, 't' )
plot_residual('Decision Tree Regression', y_train_predict, y_train, y_test_predict, y_test)
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.model_selection import GridSearchCV

# param_grid = {

#     'bootstrap': [True],

#     'max_depth': [4, 5, 6, 10],

#     'max_features': [2, 3],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [100, 200, 300, 1000]

# }

# # Create a based model

# rf = RandomForestRegressor()

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 2)

# grid_search.fit(X_train, y_train)

# grid_search.best_params_
#try Random Forest



#rf = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=4, n_jobs=-1, random_state=1) # selected features

rf1 = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1) # original features

#rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=7, n_jobs=-1, random_state=1) # orig + extrcted 



rf1.fit(X_train, y_train)



y_test_predict = rf1.predict(X_test)

y_train_predict = rf1.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



rf1_metrics = get_regression_metrics('Random Forest Regression', y_test, y_test_predict)

rf1_metrics
rf2 = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=10, n_jobs=-1, random_state=1) # original features

#rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=7, n_jobs=-1, random_state=1) # orig + extrcted 



rf2.fit(X_train, y_train)



y_test_predict = rf2.predict(X_test)

y_train_predict = rf2.predict(X_train)



print('R^2 training: %.3f, R^2 test: %.3f' % (

      (metrics.r2_score(y_train, y_train_predict)), 

      (metrics.r2_score(y_test, y_test_predict))))



rf2_metrics = get_regression_metrics('Random Forest Regression', y_test, y_test_predict)

rf2_metrics
plot_residual('Random Forest Regression', y_train_predict, y_train, y_test_predict, y_test)
# try recursive feature elimination



kfold = model_selection.KFold(n_splits=5, random_state=10)



dtrg = DecisionTreeRegressor(max_depth=7)



rfecv = RFECV(estimator=dtrg, step=1, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)

rfecv.fit(X_train, y_train)



print("Optimal number of features : %d" % rfecv.n_features_)



sel_features = [f for f,s in zip(X_train.columns, rfecv.support_) if s]

print('The selected features are: {}'.format(sel_features))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected (RFE)")

plt.ylabel("Cross validation score (mse)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# view predictions vs actual



rf_pred_dict = {

                'Actual' : y_test,

                'Prediction' : y_test_predict

            }

    

rf_pred = pd.DataFrame.from_dict(rf_pred_dict)

abs(rf_pred.Actual-rf_pred.Prediction).hist(bins=10)
#regression metrics comparison before feature engineering



reg_metrics_bfe = pd.concat([linreg_metrics, lasso_metrics, rdg_metrics, dtrg_metrics, polyreg_metrics, rf1_metrics,rf2_metrics], axis=1)

reg_metrics_bfe
# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import Lasso



# alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]



# lasso = Lasso()



# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



# ridge_regressor = GridSearchCV(lasso, parameters,scoring='neg_mean_squared_error', cv=5)



# ridge_regressor.fit(X_train, y_train)