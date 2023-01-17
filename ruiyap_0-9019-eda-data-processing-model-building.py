# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Load necessary packages for EDA

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt # for data visualization use

import seaborn as sns # for data visualization use

import scipy.stats as stats

import warnings

warnings.filterwarnings("ignore") # to filter out warnings messages
df =  pd.read_csv("../input/housesalesprediction/kc_house_data.csv")



## Take a quick pick on the data

print(df.info()) 

print("\n\n",df.nunique()) # Return the number of unique values in each feature

print("\n\n",df.head()) #Return the first 5 row of the dataframe
## Drop the "id" column. It has nothing to do with the "price" column

df = df.drop("id",axis=1)
## Plot 1

plt.figure(figsize=(12,9))

plt.subplot(121)

plt.title("Price Distribution")

sns.distplot(df["price"])



## Plot 2

plt.subplot(122)

plt.scatter(range(df.shape[0]),np.sort(df["price"].values))

plt.title("Price Curve Distribution",fontsize=15)

plt.ylabel("Amount ($)",fontsize=12)



plt.show()
num_cols = df.select_dtypes(exclude="object").columns.tolist() # Taking all numerical feature's name

num_cols.remove("price") # Remove the target variable 



for col in num_cols:

    sns.lmplot(data=df,x=col,y="price")

    plt.show()
# Import packages

import folium

from folium.plugins import HeatMap



# Define a function to generate the basemap

def generateBaseMap(default_loc=[df.lat.loc[1],df.long.loc[1]]): #set the default location to the first house geographical position

    base_map = folium.Map(location=default_loc,control_scale=True)

    return base_map



#Generate the Heatmap

basemap = generateBaseMap()

#Add a weightage for heatmap density purpose

df_map = df.copy()

df_map["weightage"] = 1



# Add Icon to the option map

s = folium.FeatureGroup(name="icon").add_to(basemap)

# Add marker for the house with the highest price

max_price = df.loc[df.price.idxmax()]

folium.Marker([max_price["lat"],max_price["long"]],popup="Highest Price:${}".format(max_price.price),icon=folium.Icon(color="green")).add_to(s)



# Add heatmap

HeatMap(data=df_map[["lat","long","weightage"]].groupby(["lat","long"]).sum().reset_index().values.tolist(),radius=8,max_zoom=13,name="Heat Map").add_to(basemap)

folium.LayerControl(collapsed=False).add_to(basemap)



#Show the diagram

basemap
## Lower Triangle Correlation Matrix

mask = np.zeros_like(df.corr(),dtype=bool)

mask[np.triu_indices_from(mask)] = True



## Plot Spearman Correlation Matrix

plt.figure(figsize=(12,9))

plt.title("Spearman Correlation Matrix",fontsize=16)

sns.heatmap(df.corr(),annot=True,fmt=".2f",mask=mask, square=True,linewidth=1.0,annot_kws={"fontsize":10})

plt.show()



## Plot top 10 most correlated features

feats = df.corr().nlargest(10,"price")["price"].index

cm = np.corrcoef(df[feats].values.T)

plt.figure(figsize=(12,9))

plt.title("Top 10 Correlated Features",fontsize=16)

sns.heatmap(cm,annot=True,fmt=".2f",linewidths=1.0,square=True,xticklabels=feats,yticklabels=feats,annot_kws={"fontsize":14})

plt.show()
## Drop sqft_above feature

df = df.drop("sqft_above",axis=1)
## Missing data

print(df.isnull().any()) # Return True if there's a present of missing data
from sklearn.preprocessing import StandardScaler



price_scaled = StandardScaler().fit_transform(df["price"][:,np.newaxis]) #scale the price

high_range = price_scaled[price_scaled[:,0].argsort()][-10:]

low_range = price_scaled[price_scaled[:,0].argsort()][:10]



print("Outer range (low) of the distribution :\n{}".format(low_range))

print("Outer range (high) of the distribution :\n{}".format(high_range))
#Remove the outliers

outliers = df["price"].sort_values(ascending=False)[:3].index 

df = df.drop(outliers,axis=0)
## define a function

presence = lambda x: 1 if x > 0 else 0 #Return 1 if there the house has the feature or else 0 for absence of that feature

features = ["bedrooms","bathrooms","waterfront","view","sqft_basement","yr_renovated"]



for feat in features:

    df["Has_"+feat] = df[feat].transform(presence)



## Create new columns based on the date feature

df["date"] = pd.to_datetime(df["date"]) # Convert to datetime

yr = lambda x: x.year ; month = lambda x: x.month 



df["yr_sold"] = df["date"].transform(yr)

df["month_sold"] = df["date"].transform(month)



## Create a new column based the age renovation

df["age_rnv"] = df["yr_sold"] - df["yr_built"]

df.loc[df["Has_yr_renovated"] == 1,"age_rnv"] = df["yr_sold"] - df["yr_renovated"]



## Remove yr_renovated as we have age_rnv

df = df.drop("yr_renovated",axis=1)

## Remove date as we have yr_sold and month_sold

df = df.drop("date",axis=1)

## Remove zipcode

df = df.drop("zipcode",axis=1)



## Let's check on the all the features

print(df.head())
# Let's look at the Target Variable first

from scipy.stats import skew, kurtosis

from scipy.stats import shapiro



# Check the p value from the Shapiro-Wilk Test

stc , p_value = shapiro(df.price.values)

print("p value of price feature is {}".format(p_value))
# Comparing the feature's skewness & kurtosis before and after normalization through visualization

fig,ax = plt.subplots(1,2,figsize=(12,9))



#Before Normalization

skewness = format(skew(df.price),".2f") ; kurt = format(kurtosis(df.price),".2f")

sns.distplot(df["price"],ax=ax[0])

ax[0].legend(["Skewness: {}\nKurtosis: {}".format(skewness,kurt)],loc="upper_right") #Note using legend to plot the skewness and kurtosis is not a corrrect & professional way, I did it for the purpose of cleaner visualization. 

ax[0].set(title="Before Normalization")



#After Normalization

df["price"] = np.log1p(df["price"]) # Normaliza the "price" feature

skewness = format(skew(df.price),".2f") ; kurt = format(kurtosis(df.price),".2f")

sns.distplot(df["price"],ax=ax[1])

ax[1].legend(["Skewness: {}\nKurtosis: {}".format(skewness,kurt)],loc="upper_right")

ax[1].set(title="After Normalization")

plt.show()
# import packages

from scipy.stats import boxcox_normmax

from scipy.special import boxcox1p



# Now let's normalize continuous numerical predictors that are not drawn from a normal distribution.

cols = ["sqft_living","sqft_lot","sqft_basement","sqft_living15","sqft_lot15"]



norm_test = lambda x: shapiro(x)[1] < 0.05 # Check if the p value is less than 0.05

num_feats = df[cols].apply(norm_test)



for col in cols:

    fig, ax = plt.subplots(1,2)

    sns.distplot(df[col],ax=ax[0])

    skewness = format(skew(df[col]),".2f") ; kurt = format(kurtosis(df[col]),".2f")

    ax[0].legend(["Skewness: {}\nKurtosis: {}".format(skewness,kurt)],loc="upper_right")

    ax[0].set(title="Before Normalization")

    

    df[col] = boxcox1p(df[col],boxcox_normmax(df[col]+1)) #Normalize the feature

    sns.distplot(df[col],ax=ax[1])

    skewness = format(skew(df[col]),".2f") ; kurt = format(kurtosis(df[col]),".2f")

    ax[1].legend(["Skewness: {}\nKurtosis: {}".format(skewness,kurt)],loc="upper_right")

    ax[1].set(title="After Normalization")

    plt.show()
all_feats = df.columns.tolist()



#Create an empty datatframe

percent = pd.DataFrame(columns = ["frequency"],index = all_feats) 



# Calculate the percentage of the mode value in the feature

for col in all_feats:

    percent.loc[col,"frequency"] = df[col].value_counts().iloc[0]/df.shape[0]  * 100



bias_feats = percent[percent["frequency"] > 99.94].index.tolist() # Threshold of 99.94%



##Remove the biased features

df = df.drop(bias_feats,axis=1)
# import models 

from sklearn.linear_model import Ridge,Lasso,ElasticNet,HuberRegressor,BayesianRidge

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor,AdaBoostRegressor, VotingRegressor

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor



# import necessary packages for model building

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold,cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



# Split the data into train and test set

y = df.price.values ; X = df.drop("price",axis=1).values # target variable would be the price and the else would be the predictors



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=2)



# Create an empty list for pipeline 

pipeline_models = []



# Assign all models into the a list

seed = 2

models = [Ridge(tol=10,random_state=seed),

          Lasso(tol=1,random_state=seed),

          ElasticNet(random_state=seed),

          HuberRegressor(),

          BayesianRidge(),

          RandomForestRegressor(random_state=seed),

          ExtraTreesRegressor(random_state=seed),

          BaggingRegressor(random_state=seed),

          GradientBoostingRegressor(),

          XGBRegressor(),

          DecisionTreeRegressor(),

          KNeighborsRegressor(),

          AdaBoostRegressor(random_state=seed)]



model_names = ["Ridge","Lasso","Elastic","Hub_Reg","BayRidge","RFR","ETR","BR","GBoost_Reg","XGB_Reg","DT_Reg","KNN_Reg","Ada_Reg"] # All models' labels



# Assign each model to a pipeline

for name,model in zip(model_names,models):

    pipeline = ("Scaled_"+name,

                Pipeline([("Scaler",StandardScaler()),

                         (name,model)

                         ]))

    pipeline_models.append(pipeline)
# # Create a dataframe to store all the model's cross validation score

evaluate = pd.DataFrame(columns=["model","cv","std","cv_all"]) #note: "cv" for mean cv & "cv_all" for later visualization use



for name,model in pipeline_models:

    kfold = KFold(n_splits=7) # 7 times

    cv = cross_val_score(model,X_train,y_train,cv=kfold,n_jobs= -1,scoring="r2")

    

    row = evaluate.shape[0]

    evaluate.loc[row,"model"] = name

    evaluate.loc[row,"cv"] = cv.mean()

    evaluate.loc[row,"std"] = "+/- {}".format(cv.std())

    evaluate.loc[row,"cv_all"] = cv

    

evaluate = evaluate.sort_values("cv",ascending=False)

## Visualization on the cv score



fig, ax = plt.subplots (1,2,figsize=(18,12))

b = sns.barplot(x=evaluate["model"],y=evaluate["cv"],ax=ax[0],palette = sns.cubehelix_palette(evaluate.shape[0]))

for rec in b.patches:

    height = rec.get_height()

    ax[0].text(rec.get_x()+rec.get_width()/2,height * 1.01, round(height,4),ha="center")

ax[0].set(title="All models' CV score")

ax[0].set_xticklabels(evaluate["model"].tolist(),rotation=90)



sns.boxplot(x=evaluate["model"].tolist(),y=evaluate["cv_all"].tolist(),ax=ax[1])

ax[1].set(title="All models' CV distribution scores")

ax[1].set_xticklabels(evaluate["model"].tolist(),rotation=90)

plt.show()
## Create list to store all the combinations

votings = []



# XGBRegressor only                                 

votings.append(("Scaled_XGBR",Pipeline([("Scaler",StandardScaler()),

                                        (("XGB_Reg",XGBRegressor()))

                                       ])))



# All models

votings.append(("Scaled_XGBR_ETR_GBoostR_BR",Pipeline([("Scaler",StandardScaler()),

                                                       ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                                  ("ETR",ExtraTreesRegressor(random_state=seed)),

                                                                                  ("GBoost_Reg",GradientBoostingRegressor()),

                                                                                  ("BR",BaggingRegressor())

                                                                                 ]))

                                                     ])))



# XGBR with ETR combinations               

votings.append(("Scaled_XGBR_ETR",Pipeline([("Scaler",StandardScaler()),

                                            ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                       ("ETR",ExtraTreesRegressor(random_state=seed))

                                                                      ]))

                                           ])))



#XGBR with ETR & GBoost_Reg combinations              

votings.append(("Scaled_XGBR_ETR_GBoostR",Pipeline([("Scaler",StandardScaler()),

                                                      ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                                 ("ETR",ExtraTreesRegressor(random_state=seed)),

                                                                                 ("GBoost_Reg",GradientBoostingRegressor())

                                                                                ]))

                                                   ])))



#XGBR with ETR & BR combinations

votings.append(("Scaled_XGBR_ETR_BR",Pipeline([("Scaler",StandardScaler()),

                                               ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                          ("ETR",ExtraTreesRegressor(random_state=seed)),

                                                                          ("BR",BaggingRegressor())

                                                                       ]))

                                             ])))

#XGBR with GBoost_Reg combinations              

votings.append(("Scaled_XGBR_GBoostR",Pipeline([("Scaler",StandardScaler()),

                                                ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                           ("GBoost_Reg",GradientBoostingRegressor())

                                                                          ]))

                                               ])))



#XGBR with GBoost_Reg & BR combinations              

votings.append(("Scaled_XGBR_GBoostR_BR",Pipeline([("Scaler",StandardScaler()),

                                                   ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                              ("GBoost_Reg",GradientBoostingRegressor()),

                                                                              ("BR",BaggingRegressor())

                                                                             ]))

                                                   ])))



#XGBR with BR combinations                                 

votings.append(("Scaled_XGBR_BR",Pipeline([("Scaler",StandardScaler()),

                                           ("Voting",VotingRegressor([("XGB_Reg",XGBRegressor()),

                                                                      ("BR",BaggingRegressor())

                                                                     ]))

                                           ])))



#Create an empty dataframe to store the best score

evaluate_vote = pd.DataFrame(columns=["model","cv","std"])



## train the model and assign the cv score to evaluate_vote

for name, model in votings:

    kfold = KFold(n_splits=7)

    cv = cross_val_score(model,X_train,y_train,scoring="r2",cv=kfold,n_jobs=-1)

    

    row = evaluate_vote.shape[0]

    evaluate_vote.loc[row,"model"] = name

    evaluate_vote.loc[row,"cv"] = cv.mean()

    evaluate_vote.loc[row,"std"] = "+/- {}".format(cv.std())

    

evaluate_vote = evaluate_vote.sort_values("cv",ascending=False)

## Check on the scores

print(evaluate_vote)                                          

                                          
# Visualization

fig, ax = plt.subplots (1,1,figsize=(18,12))

b = sns.barplot(x=evaluate_vote["model"],y=evaluate_vote["cv"],ax=ax,palette = sns.cubehelix_palette(evaluate.shape[0]))

for rec in b.patches:

    height = rec.get_height()

    ax.text(rec.get_x()+rec.get_width()/2,height * 1.01, round(height,4),ha="center")

ax.set(title="All Combinations Voting Regressor cv scores")

ax.set_xticklabels(evaluate_vote["model"].tolist(),rotation=90)

plt.show()