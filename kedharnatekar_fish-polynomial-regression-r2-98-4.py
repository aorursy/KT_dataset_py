# Importing necessary models 



import pandas as pd

import matplotlib as mp

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from scipy.stats import pearsonr

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score 

from sklearn.metrics import r2_score

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
def fish_data_import():

    """

    Function useful for importing a file and converting it to a dataframe

    """

    datafile = pd.read_csv('/kaggle/input/fish-market/Fish.csv')

    return datafile
# Importing the dataset file

fish_df = fish_data_import()
fish_df.info()
fish_df
fish_df.describe()
# Defining a function for Horizonal bar plot 



def plot_counts_bar(data,column,fig_size=(16,9),col='blue',col_annot='grey',water_m=False,water_text='KedNat'):

    """

    Function plot_counts_bar plots a horizontal bar graph for Value counts for a given Dataframe Attribute.

    This is much useful in analysis phase in Datascience Projects where data counts for a particular attributes needs to be visualized.

    Mandatory inputs to this function. 

        1. 'data' where dataframe is given as input 

        2. 'column' where column name is given as input for which we need the value counts.

    Optional inputs to this function:

        1. 'fig_size' which represent the figure size for this plot. Default input is (16,9)

        2. 'col' which represents the color of the bar plot. Default input is 'blue'

        3. 'col_annot' which represents the color of annotations. Default input is 'grey'

        4. 'water_m' which represents if we need a watermark text. Default input is boolean as False

        5. 'water_text' which inputs a string variable used for watermark. Default is KedNat

    """

    

    # Figure Size 

    fig, ax = plt.subplots(figsize =fig_size) 



    # Defining the dataframe for value counts

    df = data[column].value_counts().to_frame()

    df.reset_index(inplace=True)

    df.set_axis([column ,'Counts'], axis=1, inplace=True)

    X_data = df[column]

    y_data = df['Counts']



    # Horizontal Bar Plot 

    ax.barh(X_data, y_data , color=col) 



    # Remove axes splines 

    for s in ['top', 'bottom', 'left', 'right']: 

        ax.spines[s].set_visible(False)



    # Remove x, y Ticks 

    ax.xaxis.set_ticks_position('none') 

    ax.yaxis.set_ticks_position('none') 



    # Add padding between axes and labels 

    ax.xaxis.set_tick_params(pad = 5) 

    ax.yaxis.set_tick_params(pad = 10) 



    # Show top values 

    ax.invert_yaxis()

    

    # Add annotation to bars 

    for i in ax.patches: 

        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()), 2)),fontsize = 10, fontweight ='bold',color =col_annot) 



    # Add Plot Title 

    title = 'Counts of each '+column

    ax.set_title(title, loc ='left', fontweight="bold" , fontsize=16) 

    

    # Add Text watermark 

    if water_m == True:

        fig.text(0.9, 0.15, water_text, fontsize = 12, color ='grey', ha ='right', va ='bottom', alpha = 0.7) 



    ax.get_xaxis().set_visible(False)



    # Show Plot 

    plt.show() 
# Plotting the species counts on entire dataset

plot_counts_bar(fish_df,'Species',(10,6),col='green',col_annot='blue')
# Defining a function for Stratified split on a given column



def stratified_split(data,column,testsize=0.2):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data[column]):    

        strat_train_set = fish_df.loc[train_index]    

        strat_test_set = fish_df.loc[test_index]

        return(strat_train_set,strat_test_set)
# Splitting into train and test dataset on basis of Stratified split column of Species

train_set,test_set = stratified_split(fish_df,'Species')
# Counts of species on test set

plot_counts_bar(test_set,'Species',(10,6),col='green',col_annot='blue')
# fish_df will now be Training set

fish_df = train_set.copy()
fish_df.info()
# Defining a function for Heatmap on a given data

def heat_map(data,fig_size=(8,8)):



    fig, ax = plt.subplots(figsize=fig_size)

    heatmap = sns.heatmap(data,

                          square = True,

                          linewidths = .2,

                          cmap = 'YlGnBu',

                          cbar_kws = {'shrink': 0.8,'ticks' : [-1, -.5, 0, 0.5, 1]},

                          vmin = -1,

                          vmax = 1,

                          annot = True,

                          annot_kws = {'size': 12})



    #add the column names as labels

    ax.set_yticklabels(data.columns, rotation = 0)

    ax.set_xticklabels(data.columns)



    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
# Heatmap for Training set

heat_map(fish_df.corr())
heat_map(fish_df[['Length1','Length2','Length3']].corr())
# Defining a function to calculate Pearson-correlation and p-values w.r.t given label columns

def peason_test(data,label):

    """

    This function gives the resultant Person correlation and p-value for given set of numeric attrivutes w.r.t labelled column.

    Inputs asre as follows:

    "data" : dataframe of numeric columns . eg : df[['A','B',C]]

    "label": Name of labelled column which is present in dataframe in data input. eg : 'label'

    """

    

    print('Pearson Correlation and p-values continous values\n')

    for i in list(data.columns):

        p_cor, p_val = pearsonr(data[label],data[i])

        print('For '+i+' :')

        print('    Pearson correlation :'+str(round(p_cor,5)))

        print('    p-value             :'+str(p_val))
# Defining column lists which will be useful further

num_cols =   ['Weight','Length1','Length2','Length3','Height','Width']

cat_cols =   ['Species']

label_cols = ['Weight']

label = 'Weight'
# Calculating Pearsons correlation tests 

peason_test(fish_df[num_cols],'Weight')
# transform data can be used to transform a given set of dataframe using data cleansing.

# Onehot Encoded values for Species are also created



def transform_data(data):

    """

    Function used to transform a given set of dataframe. Eg Train or Test dataframe

    This is used to 

    --> fix the data with Null values

    --> remove the unnecessary columns.

    --> create Labelencoded or OneHotEncoded attributes

    """

    data.drop(['Length1','Length2'],axis=1,inplace=True)

    result_df = pd.get_dummies(data,columns=['Species'],prefix=['Species'])

    return result_df
# Transforming the Training set

fish_train = transform_data(fish_df)
fish_train
# Redefining the lists of attributes 

num_cols = ['Weight','Length3','Height','Width']

ohe_cols = ['Species_Bream','Species_Parkki','Species_Perch','Species_Pike','Species_Roach','Species_Smelt','Species_Whitefish']

independent_features = ['Length3', 'Height', 'Width', 'Species_Bream','Species_Parkki', 'Species_Perch', 'Species_Pike', 'Species_Roach','Species_Smelt', 'Species_Whitefish']
# Defining a simple function for Scatter plot useful for analysis

def scatter_plot(data,hue=None,kind='scatter'):

    """

    Scatter plot function defined for understanding the data relations between numeric attributes in dataframe.

    'data' is used to input the dataframe with the set of numeric attributes. Eg : df[['A','B','C']]

    """

    sns.pairplot(data,hue=hue,kind=kind,corner=False)
# Scatter plot for Numeric data

scatter_plot(fish_train[num_cols])
# Creating poynomial_bestfit which outputs the best polynomial for for a given data with label



def Poynomial_bestfit(data,X_col,y_col,degrees=list(range(1,5)),cv=5):

    """

    Poynomial_bestfit find the best of degree for a given dataset.

    Inputs are as follows:

    'data' : dataframe as an input which has dependent and independent attributes. eg : df

    'X_col': independent variables as a list. eg : ['A','B','C']

    'y_col': dependent variables as a string input. eg : 'label'

    """

    plt.rcParams["figure.figsize"] = [9,4]    

    degrees = degrees 

    best_score = 0

    best_degree = 0

    score_l = []

    for degree in degrees:

            X = np.asanyarray(data[X_col])

            y = data[[y_col]]

            poly_features = PolynomialFeatures(degree = degree)

            X_poly = poly_features.fit_transform(X)

            polynomial_regressor = LinearRegression(normalize=False)

            polynomial_regressor.fit(X_poly, y)

            scores = cross_val_score(polynomial_regressor, X_poly, y, cv=cv)

            scores_mean = round(scores.mean(),2)

            score_l.append(scores_mean)

            if scores_mean > best_score :

                best_score = scores_mean

                best_degree = degree

    plot_df = pd.DataFrame({'Degrees': degrees,'Scores': score_l})

    plot_df.plot(kind='line',x='Degrees',y='Scores')

    plt.title('Scores for Polynomial fit for attributes ')

    print('Best Score  is :'+str(best_score))

    print('Best Degree is :'+str(best_degree))

    plt.show()
Poynomial_bestfit(fish_train,independent_features,'Weight')
# rmse_check function is useful in giving the RMSE for different Regression models



def rmse_check(X,y,degree=1,cv=5):

    """

    Function to be used to check the RMSE for different Regression techniques to as to compare errors on different models.

    The Models include Polynomial regression for a degree as degree.

    Decision Tree regressor model and Random forest regressor.

    Inputs include dataframe of Independent variables as X and Dependent variables as y for a given no of splits as cv.

    Outputs as RMSE errors for a given model.

    This helps us to decide on which model is best fit for a given training data.

    """



    X = np.asanyarray(X)



    poly_features = PolynomialFeatures(degree = degree)

    X_poly = poly_features.fit_transform(X)

    poly_reg = LinearRegression()

    scores = cross_val_score(poly_reg,X_poly,y,scoring="neg_mean_squared_error", cv=cv)

    poly_rmse_scores = np.sqrt(-scores)

    print('RMSE for Polynomial regression of degree '+str(degree)+' is :'+str(poly_rmse_scores.mean()))

    

    tree_reg = DecisionTreeRegressor()

    scores = cross_val_score(tree_reg, X,y,scoring="neg_mean_squared_error", cv=cv)

    tree_rmse_scores = np.sqrt(-scores)

    print('RMSE for Decision Tree regressor is :'+str(tree_rmse_scores.mean()))

    

    forest_reg = RandomForestRegressor()

    scores = cross_val_score(forest_reg, X,y,scoring="neg_mean_squared_error", cv=cv)

    forest_rmse_scores = np.sqrt(-scores)

    print('RMSE for Random Forest regressor is :'+str(forest_rmse_scores.mean()))
# Calculating RMSE for Training data. Data is split into 10 folds and means of scores is calculated

rmse_check(fish_train[independent_features], fish_train[label],2,10)
# poly_model_trains the model for entire training data and returns the predicted labels along with model

def poly_model_train(X,y,degree,train_set=False):

    """

    Used to train the Polynomial model for a given degree. 

    Inputs are Independent attributes as X and Dependent variables as y for a given degree as degree

    Returns the trained Polynomial model as poly_reg and Predicted label for training set as yhat_train.

    """

    poly_features = PolynomialFeatures(degree = degree)

    X = np.asanyarray(X)

    X_poly = poly_features.fit_transform(X)

    poly_reg = LinearRegression()

    poly_reg.fit(X_poly, y)

    yhat_train = poly_reg.predict(X_poly)

    return (poly_reg,yhat_train)
# Training the test data for polynomial regression of degree 2

poly_reg_model , yhat_train = poly_model_train(fish_train[independent_features],fish_train[label],2)
# Defining a distribution plot function to plot Actual vs Predicted labels

def distributionPlot(RedFunction, BlueFunction, RedName, BlueName,x_label,y_label,title):

    """

    This is used to create a distribution plots with Actual label vs Predicted label.

    Useful for understanding where is the gap in prediction for different values of y

    """

    width = 8

    height = 6

    plt.figure(figsize=(width, height))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)



    plt.title(title)

    plt.xlabel(x_label)

    plt.ylabel(y_label)



    plt.show()

    plt.close()
# Distribition Plot for Training data 

distributionPlot(fish_train[label], yhat_train, "Actual Values (Train Polynomial reg)", "Predicted Values (Train Polynomial reg)", 

                 'Weight of Fish','Propotion of Fish',

                 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution')
# Creating a Test dataset

fish_test = test_set.copy()
# Transforming the Test dataset using transform_data function

fish_test = transform_data(fish_test)
# Creating function poly_model_test for Predicting the test dataset using model created using Training data

def poly_model_test(X,y,poly_model,degree):

    """

    This is used for predicting the Test dataset scores of polynomial regression for a given model input as poly_reg_model.

    This function prints the R2 value for the give test data and returns predicted label as  yhat_test

    """

    poly_features = PolynomialFeatures(degree = degree)

    X = np.asanyarray(X)

    X_poly_test = poly_features.fit_transform(X)

    yhat_test = poly_model.predict(X_poly_test)

    print('R2 for test data '+str(r2_score(yhat_test ,y)))

    return(yhat_test)
# Checking the R2 score for Test data using function poly_model_test

yhat_test = poly_model_test(fish_test[independent_features],fish_test[label],poly_reg_model,2)
# Distribution Plot on Test dataset



distributionPlot(fish_test[label], yhat_test, "Actual Values (Test)", "Predicted Values (Test)", 

                 'Weight of Fish','Propotion of Fish',

                 'Distribution  Plot of  Predicted Value Using Testing Data vs Testing Data Distribution')