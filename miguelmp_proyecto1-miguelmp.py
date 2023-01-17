# some imports

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
 
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rc('font', size=12) 
plt.rc('figure', figsize = (12, 5))

# Settings for the visualizations
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})

import pandas as pd
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 50)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# create output folder
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('output/session1'):
    os.makedirs('output/session1')
## load data
train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 
test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 
print(type(train_set))
train_set.head()
# print the dataset size
print("There is", train_set.shape[0], "samples")
print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset
train_set.head()
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.
# It is important to check each of the to understand it.
# we can see the type of each features as follows
train_set.dtypes
# print those categorical features
train_set.select_dtypes(include=['object']).head()

print(pd.Series.to_numpy(train_set["Type"].value_counts()))
# We can check how many different type there is in the dataset using the folliwing line
train_set["Type"].value_counts()

sns.countplot(y="Type", data=train_set, color="c")
sns.distplot(train_set["Price"])
plt.show()
## the features

features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']
## DEFINE YOUR FEATURES
X = train_set[features].fillna(0)
y = train_set[['Price']]

## the model
# KNeighborsRegressor
from sklearn import neighbors
n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)
model = neighbors.KNeighborsRegressor(n_neighbors)
print(type(model))
## fit the model
model.fit(X, y)

## predict training set
y_pred = model.predict(X)
print("y_pred: ")
print(y_pred.shape)
## Evaluate the model and plot it
from sklearn.metrics import mean_squared_error, r2_score
print("----- EVALUATION ON TRAIN SET ------")
print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))
print("R^2: ",r2_score(y, y_pred))


plt.scatter(y, y_pred)
plt.xlabel('Price')
plt.ylabel('Predicted price');
plt.show()

## predict the test set and generate the submission file
X_test = test_set[features].fillna(0)
y_pred = model.predict(X_test)

df_output = pd.DataFrame(y_pred)
df_output = df_output.reset_index()
df_output.columns = ['index','Price']

df_output.to_csv('output/session1/baseline.csv',index=False)
def apply_numerical_preprocessing(df,num_features):
    #Just applies the below functions to the numerical features.
    fill_nan(df,num_features)
    clean_outlayers(df,num_features)
    normalize(df,num_features)
def describe(df,num_features):
    for f in num_features:
        print(f+": ")
        print(df.loc[:,f].describe())
        print()
def clean_outlayers(df,num_features,q=0.01):
    """
    I considered the outlayers anything below/above the 1/99 percentile.
    I replaced them with those values so, even though I change them, they still
    are in the extremes and the model takes that into account.
    
    This could work fine for the "true" outlayers, but if an outlayer was a typo mistake it could be a bad solution.
    """
    for f in num_features:
        q_high=df.loc[:,f].quantile(q=1-q)
        q_low =df.loc[:,f].quantile(q=q)
        arr_f=pd.Series.to_numpy(df.loc[:,f])
        arr_f[arr_f>q_high]=q_high
        arr_f[arr_f<q_low]=q_low
        df.loc[:,f]=arr_f
def fill_nan(df,num_features):
    """
    First of all, I ONLY fill the nans in the numerical features.
    
    I decided trying both mean and median and median gave best results.
    
    The one thing about median that I like is that you are using a value that already exists in the dataset. 
    In the case of AreaBuilt it is not really a problem to use the mean, because there are a lot of different values
    (I consider it a continuous variable), but in the case of Rooms, a discrete variable, it could lead to a new dicrete value (ex: 1.8 Rooms)
    that did not exist previously and could potentially cause problems.
    
    Finally I made the Landsize feature to be filled with Zeros because it had sense that if a house did not had Landsize,
    it just was left blank. (In the train_set there were no nans in the Landsize column, but I did realize that later).
    """
    for f in num_features:
        if(f=='Landsize'):
            df.loc[:,f]=df.loc[:,f].fillna(value=0.)
        else:
            median=pd.Series.median(df.loc[:,f])
            df.loc[:,f]=df.loc[:,f].fillna(value=median)
def normalize(df,num_features):
    """
    Just normalizing the values so that in the Knn every feature represented in a dimension had the same "impact" because of the distance. 
    """
    for f in num_features:
        max_value=df.loc[:,f].max()
        min_value=df.loc[:,f].min()
        if(max_value-min_value != 0):
            df.loc[:,f]=(df.loc[:,f]-min_value)/(max_value-min_value)
        else:
            #Just in case
            df.loc[:,f]=df.loc[:,f]/(max_value)

            
def clean_dummies(df,selected_dummies):
    """
    This function was meant to try and use only some values from the categorical features, but that did not end up giving a better result,
    so finally it was not used.
    """
    
    #selected_dummies is a list of tuples in which for every element s
    #s[0] is the name of a column.
    #s[1] are the values selected for that column.
    for s in selected_dummies:
        col_name=s[0]
        to_remove=df[col_name].unique().tolist()
        for value in s[1]:
            # We remove the values we want to keep from the to_remove list.
            to_remove.remove(value)
        #Every value in the to_remove list is replaced with a nan (so it is easier to make dummies of the column)
        df[col_name]=df[col_name].replace(to_replace=to_remove,value=np.nan)

def make_dummies(df,cat_features):
    """
    Function to make the "Dummies". 
    This will transform each categorical feature into a lot of numerical features. (One for every unique value).
    """
    for f in cat_features:
        dummies = pd.get_dummies(df[f],prefix=f, dummy_na=False)
        df = df.drop(f,1)
        df = pd.concat([df,dummies], axis=1 )
    return df
def get_dummies_names(df, cat_features):
    """
    Returns a list with the name of the dummies made in the make_dummies() function.
    """
    dummies=[]
    for cat in cat_features:
        for col in df.columns:
            if cat in col:
                dummies.append(col)
    return dummies
def clean_zeros_for_log(df,num_features):
    """
    For cleaning the Zeros I tried to maintain something reasonable for the different num characteristics.
    I wanted it to be the least "Hardcoded" posible solution, so that it didn't depend a lot on the guessing.
    I tried to avoid adding just a 0.1, because it can be very different for different data entries.
    
    Finally I decided that all the Zeros could be replaced with a value smallest than the 1 percentil (without the Zeros),
    so i tried that *0.1 and it worked fine and consistent. 
    I know my solution it still is kind of hardcoded but depends much more on the data than just an addition.
    """
    for f in num_features:
        if(0 in df[f].unique()):
            df[f]=df[f].replace(to_replace=0,value=np.nan)
            q=df[f].quantile(q=0.01)
            df[f]=df[f].replace(to_replace=np.nan,value=q*0.1)
            #df[f]=df[f].replace(to_replace=np.nan,value=q)
def apply_log(df,log_features):
    """To apply the Log to some of the numerical features first I had to do some preprocessing:
    
    First we fill the nan values with something reasonable (mean/median).
    
    Afterwards we want the outlayers to disappear.
    
    Then we need to ensure there are no Zeros or negative numbers, because the Log function is only defined in the positive domain.
    (In this implementation I don't care of the negative numbers because I was not going to use any variable with posible negative values)
    
    Later we apply the Log function.
    
    Last but not least, we normalize the results.
    
    """
    fill_nan(df,log_features)
    clean_outlayers(df,log_features)
    clean_zeros_for_log(df,log_features)
    for f in log_features:
        df[f]=np.log(df.loc[:,f])
    normalize(df,log_features)

#Useful to visualize the relation of categorical features and the price.
sns.catplot(x="Type", y="Price", kind="box", data=train_set)
#Useful to visualize the relation of numerical features and the price. It even does a regression line, the blue line in the plot.
sns.jointplot(data=train_set, x="BuildingArea", y="Price",xlim=(0,1000),ylim=(0,6000000),kind="reg")
relevantes=[]

#All categorical features
all_cate=['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea','Regionname']
#Cate is just to select the ones I want to visualize
cate=['Suburb',  'Type','SellerG', 'Postcode', 'CouncilArea', 'Propertycount']
for f in cate:
    rel=[]
    da=train_set.copy()
    columna=f
    price_mean=train_set.loc[:,"Price"].mean()
    col=da.loc[:,[columna,"Price"]]
    # Arbitrary number to get something that was at least a considerable number of samples
    num=100
    value_counts=col[columna].value_counts()
    print(value_counts.shape)
    #value_counts.drop(labels=value_counts.loc[value_counts<num].index,axis="index")
    #Drop the 
    index=value_counts.loc[value_counts<num].index
    for ind in index:
        drop_index=col[col[columna]==ind].index
        col=col.drop(labels=drop_index,axis="index")
    col.count()
    col.loc[:,"Price"]=col.loc[:,"Price"]-price_mean
    #col.head(20)
    
    #sns.catplot(x=f, y="Price", kind="box", data=col)

    nombre=[]
    diferencia=[]
    for cat in col.loc[:,columna].unique():
        nombre.append
        index_suburb=col[col[columna]==cat].index
        mean_diff = col.loc[index_suburb,"Price"].mean()

        nombre.append(cat)
        diferencia.append(mean_diff)
    #Here we have a new DataFrame with the name of every value in the Categorical Feature and the difference of means.
    df_diff=pd.DataFrame({
        "Diff":np.array(diferencia,dtype=np.float64),
        "Nombre":nombre
    })
    #df_diff.loc[:,"Diff"]=pd.DataFrame.abs(df_diff.loc[:,"Diff"])
    print("_________",f,"__________")
    df_diff_aux=df_diff.sort_values(by="Diff",ascending=False).head(3)
    print(df_diff_aux)
    relevantes.append((f,df_diff_aux.loc[:,"Nombre"].tolist()))
    df_diff_aux=df_diff.sort_values(by="Diff",ascending=True).head(3)
    print(df_diff_aux)
    #relevantes.append((f,df_diff_aux.loc[:,"Nombre"].tolist()))
relevantes
sns.catplot(x="Type", y="Price", kind="box", data=train_set)
def count_popular(df,cat_features):
    # Just to get an idea of how many different values every categorical feature has.
    for f in cat_features:
        print("__________"+f+"___________")
        print("Number of unique values: ",df.loc[:,f].nunique())
        print("MOST POPULAR: ")
        print(df.loc[:,f].value_counts().sort_values(ascending=False).head(10))
        print()
count_popular(train_set,cate)

count_popular(train_set,["Landsize"])

fe=['Price',
    'Rooms',
    'Bathroom',
    'Car',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude']

method = 'pearson'

#Just for the desired columns/features
corr=train_set.copy()
drop_list=[x for x in corr.columns if x not in fe]
corr=corr.drop(labels=drop_list,axis="columns")
corr_log = corr.copy()
apply_log(corr_log,fe)

#Log(values). Pearson correlation
log=corr_log.corr(method=method).loc["Price"]
#Original values. Pearson correlation
corr=corr.corr(method=method).loc["Price"]

correlaciones = pd.DataFrame({
    "Normal":corr,
    "Log":log
})
correlaciones
nombre_columna="BuildingArea"
cleaned_train=train_set.copy()
#apply_numerical_preprocessing(cleaned_train,[nombre_columna])


df_log = train_set.copy()

#fill_nan(df_log,[nombre_columna])
#clean_outlayers(df_log,[nombre_columna])
#clean_zeros_for_log(df_log,[nombre_columna])
apply_log(df_log,[nombre_columna])
minn=df_log[nombre_columna].min()
maxx=df_log[nombre_columna].min()
max_columna=cleaned_train[nombre_columna].quantile(q=0.99)
print("Correlación Pearson ",nombre_columna,":",cleaned_train.corr().loc[:,"Price"].loc[nombre_columna])
sns.jointplot(data=cleaned_train, x=nombre_columna, y="Price",xlim=(0,max_columna*1.1),ylim=(0,6000000),kind="reg")
print("Correlación Pearson LOG ",nombre_columna,":",df_log.corr().loc[:,"Price"].loc[nombre_columna])
sns.jointplot(data=df_log, x=nombre_columna, y="Price",xlim=(0,1),ylim=(0,6000000),kind="reg")
# KNN model

train_set_run = train_set.copy()

# Selected features:
num_features = ['Rooms',"Bathroom","Lattitude","Longtitude"]
log_features = ['Landsize']
cat_features = ['Type']

#Trying with selected dummies.
# selected_dummies=dummies_prueba # Only if using a selection of categorical values

# PREPROCESSING THE FEATURES:

# Numerical features
apply_numerical_preprocessing(train_set_run,num_features)
apply_numerical_preprocessing(test_set,num_features)

# Log features (Also numerical but with different treatment)
apply_log(train_set_run,log_features)
apply_log(test_set,log_features)

# Categorical features
"""
Trying with selected dummies.
This code is the one which should be used in case of trying to put only the most interesting values of the categorical columns.

clean_dummies(train_set_run,selected_dummies)
clean_dummies(test_set,selected_dummies)
for sd in selected_dummies:
    if not sd[0] in cat_features:
        cat_features.append(sd[0])
"""
train_set_run = make_dummies(train_set_run,cat_features)
test_set = make_dummies(test_set,cat_features)
cat_dummies  = get_dummies_names(train_set_run,cat_features)



# Final features

features = num_features + cat_dummies + log_features
total_number_features=len(features)
print("Number of features: ",total_number_features)
## DEFINE YOUR FEATURES
X = train_set_run[features]
y = train_set[['Price']]

## the model
# KNeighborsRegressor
from sklearn import neighbors
n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)
model = neighbors.KNeighborsRegressor(n_neighbors)

## fit the model
model.fit(X, y)

## predict training set
y_pred = model.predict(X)

## Evaluate the model and plot it
from sklearn.metrics import mean_squared_error, r2_score
print("----- EVALUATION ON TRAIN SET ------")
rmse=np.sqrt(mean_squared_error(y, y_pred))
print("RMSE",rmse)
r2=r2_score(y, y_pred)
print("R^2: ",r2)

# To get the correct shape I had to use .ravel() 
plot_y      = pd.DataFrame.to_numpy(y).ravel()
plot_y_pred = y_pred.ravel()

result = pd.DataFrame({"Price":plot_y,
                      "Predicted Price":plot_y_pred})

# I prefer this plot.
sns.jointplot(data=result, x="Price", y="Predicted Price",xlim=(0,9000000),ylim=(0,9000000),kind="reg")
print(result.corr())
"""
plt.scatter(y, y_pred)
plt.xlabel('Price')
plt.ylabel('Predicted price');
plt.show()
"""
## predict the test set and generate the submission file

X_test = test_set[features]
y_pred = model.predict(X_test)

df_output = pd.DataFrame(y_pred)
df_output = df_output.reset_index()
df_output.columns = ['index','Price']

df_output.to_csv('/kaggle/input/mlub-housing-house-prediction/sampleSubmission.csv',index=False)
