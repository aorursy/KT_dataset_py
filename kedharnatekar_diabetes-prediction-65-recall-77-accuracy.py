# Importing necessary modules



import pandas as pd

import matplotlib as mp

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import os



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import f_classif

from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegressionCV

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier
def diabetes_data_import():

    """

    Function useful for importing a file and converting it to a dataframe

    """

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    print(fileDir)

    relativeDir = '/kaggle/input/diabetes-dataset/diabetes2.csv'

    filename = os.path.join(fileDir,relativeDir)

    datafile = pd.read_csv(relativeDir)

    return datafile
# Importing the dataset file

diab_df = diabetes_data_import()

diab_df
diab_df.describe()
diab_df.isna().any()
# Defining a function for Horizonal bar plot 



def plot_counts_bar(data,column,fig_size=(9,4),col='blue',col_annot='grey',water_m=False,water_text='KedNat'):

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

# Plotting the labels to check the distribution

plot_counts_bar(diab_df,'Outcome',(8,4),col='green',col_annot='blue')
# Defining a function for Stratified split on a given column

def strat_shuffle_split(data,column,testsize=0.2):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data[column]):    

        strat_train_set = data.loc[train_index]    

        strat_test_set = data.loc[test_index]

        return(strat_train_set,strat_test_set)
# Splitting into train and test dataset on basis of Stratified split for label

train_set,test_set = strat_shuffle_split(diab_df,'Outcome')
# A check on Outcome in Test Dataset after split

plot_counts_bar(test_set,'Outcome',(8,4),col='Purple',col_annot='Blue')
# Setting up train set 

diab_df = train_set.copy()
diab_num = diab_df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y = diab_df['Outcome']
# Understanding the data distribution for each independent feature w.r.t label Outcome

sns.pairplot(diab_df,hue='Outcome')
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
# Correlation Matrix Heatmap for Featutes

heat_map(diab_num.corr())
# Defining a Function that gives the stats for no of zeros and nulls in a given dataset



def get_stats(data,columns,check_zero = True):

    '''

    Function get_stats gives the insights of bad data like Nulls of zeros in a given dataframe

    Mandatory Inputs to this function:

    data    : Dataframe name

    columns : Columns in dataframe that needs to be checked 

    Optional inputs to this function:

    check_zero : True if no of zeros needs to be checked

    '''

    print('Count of records in dataframe '+str(data.shape[0])+'\n')

    for i in columns:

        is_na_c = 0

        zero_c = 0

        is_na = data[i].isna().any()

        if is_na == True:

            is_na_c = data[i].isna().count()

        if check_zero == True:

            zero_c = data[i][data[i]<=0].count()

        print('Column :'+str(i))

        print('   No of Nulls :'+str(is_na_c)+'   No of Zeros or less :'+str(zero_c))
# Getting stats on Test dataset

get_stats(diab_num,list(diab_num.columns))
# Defining an Imputer to fix zero values. Same will be used for Train and Test dataset. 

# This will exclude Pregnancies and Insulin



from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=0, strategy='median')

imp.fit(diab_num[['Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']])

imp.statistics_
# Function impute_transform can be used to fit a imputer on the given dataset



def impute_transform(data,imp):

    '''

    impute_transform used to fix the dataframe 'data' with given Imputer instance 'imp'.

    It returns a transformed data in form of dataframe 

    '''

    print('\nStats before Imputing :\n')

    get_stats(data,list(data.columns))

    imp_df = imp.transform(data)

    imp_df = pd.DataFrame(imp_df,columns=list(data.columns))

    print('\nStats after Imputing :\n')

    get_stats(imp_df,list(imp_df.columns))

    return imp_df
# Function data_transform is used to Merge Imputed dataframe with Pregnancies and return the cleaned dataframe



def data_transform(data):

    '''

    data_transform merges the transformed dataframe using impute_transform along with Feature 'Pregnancies'

    '''

    imputed = impute_transform(data[['Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']],imp)

    df = pd.merge(data[['Pregnancies']],imputed,on=data.index)

    df.drop(columns=['key_0'],inplace=True)

    return df
# Creating Independent feature list in X where data is tranformed using imputer



X = data_transform(diab_num)
# Annova test results for features in X



anova_num = f_classif(X, y)

x=0

for i in X:

    print('F value for '+i+' is '+str(anova_num[0][x])+' and p-value is '+str(anova_num[1][x]))

    x+=1
# Selecting Best 3 features based on Annova test 



def k_best_select(X,y,classifier,k):

    '''

    'X' features for predict 'y' using classifier as 'classifier' with no of features to be selected as k

    This function returns the dataframe

    '''

    selector = SelectKBest(classifier, k = 3)

    selector.fit_transform(X, y)

    cols = selector.get_support(indices=True)

    X_logreg = X.iloc[:,cols]

    return X_logreg
X_logreg = k_best_select(X,y,f_classif,3)
# Defining a function to process Logistic regression Algorithm with splits and Standardization



def logistic_reg(X,y,cv=5,standardize=True):

    '''

    Features representing 'X' for labels 'y' for a Cross validation splits as 'cv'

    standardize = True uses StandardScaler before fitting the data

    '''

    if standardize== True:

        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

    clf = LogisticRegressionCV(cv=cv, random_state=0).fit(X, y)

    yhat = clf.predict(X)

    print('Accuracy score using Logistic regression :'+str(clf.score(X, y)))

    print ('\nClassification Report given below :\n'+str(classification_report(y, yhat)))

logistic_reg(X_logreg,y,5,True)
# Defining a function which helps in finding the best fit parameters for a given model using RandomizedSearchCV



def best_fit_search(X,y,estimator,param,n_iter,cv=5,scoring='accuracy',return_model = False):

    '''

   This function uses RandomizedSearchCV to search the optimal fit for given set of parameters 'param'.

   'X' and 'Y' are Features and Labels for a given algorithm 'estimator' for a RandomizedSearchCV that runs for iterations 'n_iter'.

    No of splits is defined using 'cv' and 'scoring' defines scoring pattern.

    'return_model' if True then the functions returns the bestfit model.

    '''

    search = RandomizedSearchCV(estimator=estimator, param_distributions=param, n_iter=n_iter, n_jobs=-1, cv=cv, random_state=42,scoring=scoring)

    result = search.fit(X,y)

    #print('Best parameters for fit : '+str(result.best_params_)+'\n')



    if return_model == False:

        print('Best parameters for fit : '+str(result.best_params_)+'\n')

        print('Best score for fit :'+str(result.best_score_)+'\n')

        print('Best Estimator :'+str(result.best_estimator_)+'\n')

    model = result.best_estimator_

    model.fit(X,y)

    yhat = model.predict(X)

    print('Classification Report \n'+str(classification_report(y, yhat)))

    if return_model == True:

        return model
# Finding Best fit for K-Nearest Neighbors

params = {'n_neighbors' : list(range(2,20))}

best_fit_search(X,y,KNeighborsClassifier(),params,18,cv=5,scoring='recall')
data=[[3,0.71 , 0.84,0.63 , 0.73]]

pd.DataFrame(data,columns=['K-Values','Train_Recall','Train_Accuracy','Test_Recall','Test_Accuracy'])
params = {'max_features' : [2,3,4,6] , 'max_depth' : [2,3,4,5,6,7,8] ,'n_estimators': [100]}

best_fit_search(X,y,RandomForestClassifier(),params,28,cv=5,scoring='recall')
data = [[100,3,3,0.53 , 0.79,0.48 , 0.73],

        [100,3,8,0.92 , 0.96,0.59 , 0.75],

        [100,4,6,0.78 , 0.89,0.56 , 0.74],

        [100,4,5,0.74 , 0.86,0.57 , 0.74],

        [100,3,4,0.64 , 0.83,0.50 , 0.73],

        [100,4,4,0.64 , 0.82,0.56 , 0.75]]

pd.DataFrame(data,columns=['n_estimators','min_child_weight','max_depth','Train_Recall','Train_Accuracy','Test_Recall','Test_Accuracy']).sort_values(by =['Test_Recall','Test_Accuracy'],ascending=False)
params = {

'max_depth' : [2,3,4],

'min_child_weight' : [1,2,3,4,5],

'n_estimators' : [100,200,300]

}

best_fit_search(X,y,XGBClassifier(),params,40,cv=5,scoring='recall')
data = [[100,3,3,0.83 , 0.96,0.63 , 0.75],

        [100,3,2,0.81 , 0.90,0.65 , 0.77],

        [100,1,2,0.85 , 0.91,0.63 , 0.77],

        [200,1,3,1,1,0.61 , 0.75],

        [200,5,3,0.95 , 0.98 ,0.65 , 0.75],

        [100,5,3,0.90 , 0.94,0.65,0.77],

        [200,5,4,0.99,1,0.63 ,0.72],

        [300,5,3,0.99 , 1.00,0.67,0.76]]

pd.DataFrame(data,columns=['n_estimators','min_child_weight','max_depth','Train_Recall','Train_Accuracy','Test_Recall','Test_Accuracy']).sort_values(by =['Test_Recall','Test_Accuracy'],ascending=False)
# Lets build the model using XGBoost best test Recall 



params = {'max_depth' : [2],'min_child_weight' : [3],'n_estimators' : [100]}

model = best_fit_search(X,y,XGBClassifier(),params,1,cv=5,scoring='recall',return_model=True)
# Creating a copy of Test dataset

diab_df = test_set.copy()
diab_num = diab_df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y_test = diab_df['Outcome']
# Tranforming the Testdatset  features

X_test = data_transform(diab_num)
# Predicting the Outcomes

yhat_test = model.predict(X_test)
print('\n'+str(classification_report(y_test, yhat_test)))