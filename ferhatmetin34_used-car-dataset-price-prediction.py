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
from warnings import filterwarnings

filterwarnings("ignore")

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows',None)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from scipy import stats

from folium import plugins

import branca.colormap as cm

import folium

!pip install researchpy

import researchpy

!pip install dython

from dython import nominal

from scipy.stats import shapiro,kstest

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.metrics import mean_squared_error

!pip install feature_engine

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder,OrdinalCategoricalEncoder,OneHotCategoricalEncoder

from tpot import TPOTRegressor

from statsmodels.tools.eval_measures import mse,rmse

from sklearn.metrics import r2_score,mean_squared_error

import xgboost as xgb

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

from sklearn.experimental import enable_hist_gradient_boosting  

from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score,KFold

from xgboost import plot_importance

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.inspection import partial_dependence

from sklearn.inspection import plot_partial_dependence
data=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")

df=data.copy()

df.head()
df=df.drop(columns=["id","url","region_url","image_url","description","vin","county"],axis=1)

df.head()
m = folium.Map([44 ,68], zoom_start=5,width="%100",height="%100")

locations = list(zip(df.dropna().lat, df.dropna().long))

icons = [folium.Icon(icon="airbnb", prefix="fa") for i in range(len(locations))]



cluster = plugins.MarkerCluster(locations=locations,popups=df["region"].tolist())

m.add_child(cluster)

m
df.isnull().sum().to_frame()
df.describe().T
df.describe(include=["object"]).T
pd.DataFrame(df.isnull().sum()/len(df),columns=["Missing_Rate"]).plot.bar(figsize=(12,5));

plt.axhline(0.05,color="red");
df.dropna().shape
msno.matrix(df);
msno.bar(df);
msno.heatmap(df);
# H0 : eksik değerlerin ortaya çıkması X özniteliğinin içerdiği farklı değerlerle dağılımı rastgeledir

# H1 : eksik değerlerin ortaya çıkması X özniteliğinin içerdiği farklı değerlerle dağılımı rastgele değildir

testKolon="manufacturer"

for column in df.select_dtypes(include=["object"]).columns :



        crosstab = pd.crosstab(df[column], df[testKolon])

        

        chi_square_value,pval,degrees_of_freedom,table=stats.chi2_contingency(crosstab)  

        print(column,testKolon)

        print('chi_square_value : ',chi_square_value,'\np value : ',pval)

        print('degrees of freedom : ',degrees_of_freedom,'\n')

        

        if pval <0.05:

            print("H0 rejected\n")

        else:

            print("H0 accepted\n")

        

      

    # Actually I applied this to learn how to do chi2 test for missing values.
df=df.dropna()

df.shape
df.head()
df.info()
df.odometer=df.odometer.astype(int)

df.year=df.year.astype(int)

df.dtypes
plt.figure(figsize=(16,5));

sns.countplot(df.manufacturer).set_xticklabels(labels=df.manufacturer.value_counts().index ,rotation=90);
df.hist(figsize=(7,7));
def diagnostic_plots(df, variable):

    

    plt.figure(figsize=(12, 5))



    plt.subplot(1, 3, 1)

    sns.distplot(df[variable], bins=30,kde_kws={'bw': 1.5})

    plt.title('Histogram')

    

    plt.subplot(1, 3, 2)

    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.ylabel('RM quantiles')



    plt.subplot(1, 3, 3)

    sns.boxplot(y=df[variable])

    

    

    plt.title('Boxplot')

    

    plt.show()

    

    

to_plot_labels=df.manufacturer.value_counts().nlargest(10).index
for i in to_plot_labels:

    print("--"*10,str(i).upper(),"--"*10,end="\n")

    print(stats.describe(df[df["manufacturer"]==i]["price"]))

    diagnostic_plots(df[df["manufacturer"]==i],"price")
df.eq(0).sum().to_frame()

df[df.price==0].shape[0]
df=df.drop(df[df["price"]==0].index)
stats.describe(df.price)
nominal.associations(df,figsize=(20,10),mark_columns=True);
plt.figure(figsize=(12,5))

corr=df.corr(method="spearman").abs()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr,annot=True,cmap="coolwarm",mask=mask);
df.drop(df[(df.price<500 )|( df.price>28000)].index)["price"].describe()
stats.describe(df.drop(df[(df.price<500 )|( df.price>28000)].index)["price"])
sns.boxplot(df.drop(df[(df.price<500 )|( df.price>28000)].index)["price"]);
diagnostic_plots(df,"price");
df_cleaned=df.copy()

df_cleaned=df.drop(df[(df.price<500 )|( df.price>28000)].index)

diagnostic_plots(df_cleaned,"price")
plt.figure(figsize=(17,5))



plt.subplot(141);

sns.distplot(df_cleaned.price);

print("Normality Test before boxcox transformation:",stats.shapiro(df_cleaned.price))



plt.subplot(142);

sns.boxplot(df_cleaned.price);



plt.subplot(143);

sns.distplot(stats.boxcox(df_cleaned.price)[0]);



plt.subplot(144);

sns.boxplot(stats.boxcox(df_cleaned.price)[0]);

print("Normality Test after boxcox transformation:",stats.shapiro(stats.boxcox(df_cleaned.price)[0]))

stats.boxcox(df_cleaned.price)[0][:5]
stats.probplot(stats.yeojohnson(df_cleaned.price)[0],dist="norm", plot=plt);
stats.probplot(df_cleaned.price**1/2,dist="norm", plot=plt);
stats.probplot(1/df_cleaned.price,dist="norm", plot=plt);
stats.probplot(df_cleaned.price**(1/1.5),dist="norm",plot=plt);
plt.figure(figsize=(12,5))

corr=df_cleaned.corr(method="spearman").abs()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr,annot=True,cmap="coolwarm",mask=mask);
researchpy.correlation.corr_pair(df_cleaned.select_dtypes(exclude="object"))
df_cleaned.select_dtypes(include="object").columns[1:]
nominal.associations(df_cleaned,figsize=(20,10),mark_columns=True); # for nominal and categorical (Cramer's V)
for i in df_cleaned.select_dtypes(include="object").columns[1:]:

    print(str(i) + " and " + "model")

    

    crosstab, res = researchpy.crosstab(df_cleaned[i], df_cleaned["model"], test= "chi-square")

    print(res);
for i in to_plot_labels:

    print("--"*10,str(i).upper(),"--"*10,end="\n")

    print(stats.describe(df_cleaned[df_cleaned["manufacturer"]==i]["price"]))

    diagnostic_plots(df_cleaned[df_cleaned["manufacturer"]==i],"price")

    
cols=["manufacturer","condition","cylinders","fuel","title_status","transmission","drive","size","type","paint_color"]



for i in cols:

    plt.figure(figsize=(12,5));

    sns.countplot(df_cleaned[i]).set_xticklabels(labels=df_cleaned[i].value_counts().index,rotation=90);

    

    plt.show();


for i in cols:

    df_cleaned.groupby(i)["price"].mean().sort_values(ascending=False).plot.bar(figsize=(16,5));

    plt.title("Mean Price According to " + str(i))

    plt.show();
df_cleaned.head()
df_cleaned=df_cleaned.drop(columns=["lat","long","model"],axis=1)

df_cleaned.head()
df_cleaned.state.value_counts().count()
df_cleaned.region.value_counts().count()
df_cleaned=df_cleaned.drop(columns=["region"],axis=1)
nominal.associations(df_cleaned,figsize=(20,10),mark_columns=True,cmap="coolwarm");
df_cleaned.head()
def find_skewed_boundaries(df, variable, distance):



    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    #stats.iqr(df[variable])

    print("IQR Value :",IQR)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)

    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)



    return upper_boundary, lower_boundary







find_skewed_boundaries(df_cleaned,"price",1.5)
upper_odo,lower_odo=find_skewed_boundaries(df_cleaned,"odometer",1.5)

upper_odo,lower_odo
df_cleaned[(df_cleaned.odometer>upper_odo)].shape,df_cleaned[~(df_cleaned.odometer>upper_odo)].shape
df_cleaned[df_cleaned.odometer<0]
plt.figure(figsize=(12,5));

plt.subplot(121);

sns.boxplot(df_cleaned.odometer);



plt.subplot(122);

sns.boxplot(df_cleaned[~(df_cleaned.odometer>upper_odo)]["odometer"]);
df_cleaned=df_cleaned[~(df_cleaned.odometer>upper_odo)]

df_cleaned.head()
X=df_cleaned.drop(columns=["price"])

y=df_cleaned["price"]

X.head()
y.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,

                                               test_size=0.20,

                                                random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
multi_cat_cols = []



for col in X_train.columns:



    if X_train[col].dtypes =='O': # if variable  is categorical

    

        if X_train[col].nunique() > 10: # and has more than 10 categories

            

            multi_cat_cols.append(col)  # add to the list

            

            print(X_train.groupby(col)[col].count()/ len(X_train)) # and print the percentage of observations within each category

            

            print()
for col in ['manufacturer', 'state', 'title_status']:



    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )

    plt.figure(figsize=(12,5));

    # make plot with the above percentages

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)



    # add a line at 5 % to flag the threshold for rare categories

    fig.axhline(y=0.0125, color='red')

    fig.set_ylabel('Percentage')

    plt.show()
X_train.manufacturer.value_counts().to_frame()
X_train.manufacturer.value_counts().index[-10:]
X_train["year"]=2020-X_train["year"]

X_test["year"]=2020-X_test["year"]

X_train.head()
rare_encoder = RareLabelCategoricalEncoder(

    tol=0.0125,  # minimal percentage to be considered non-rare

    n_categories=10, # minimal number of categories the variable should have to re-cgroup rare categories

    variables=["manufacturer","state","title_status"] # variables to re-group

)  
rare_encoder.fit(X_train)
rare_encoder.encoder_dict_
X_train = rare_encoder.transform(X_train)

X_test = rare_encoder.transform(X_test)
rare_encoder = RareLabelCategoricalEncoder(

    tol=0.05,  # minimal percentage to be considered non-rare

    n_categories=3, # minimal number of categories the variable should have to re-cgroup rare categories

    variables=["title_status"],

    replace_with='NotClean' # variables to re-group

)  



rare_encoder.fit(X_train)

X_train = rare_encoder.transform(X_train)

X_test = rare_encoder.transform(X_test)
X_train.title_status.value_counts()
X_train.head()
X_train.condition.value_counts()
X_train.condition=X_train.condition.replace({"salvage":1,"new":2,"fair":3,"like new":4,"good":5,"excellent":6})

X_test.condition=X_test.condition.replace({"salvage":1,"new":2,"fair":3,"like new":4,"good":5,"excellent":6})


X_train_encoded=pd.get_dummies(X_train,drop_first=True)

X_test_encoded=pd.get_dummies(X_test,drop_first=True)

X_train_encoded.head()
models=[]



models.append(XGBRegressor(random_state=42,tree_method="hist",max_depth=5))

models.append((LGBMRegressor(random_state=42)))

models.append(RandomForestRegressor(random_state=42,max_depth=5))

models.append(ExtraTreesRegressor(random_state=42,bootstrap=True,max_depth=5))

models.append(HistGradientBoostingRegressor(random_state=42))



r2_values_test = []

r2_values_train=[]

rmse_values_test=[]

mse_values_test=[]

for model in models:

    

    model_=model.fit(X_train_encoded,y_train)

    y_pred=model_.predict(X_test_encoded)

    

    r2_train=model_.score(X_train_encoded,y_train)

    r2_values_train.append(r2_train)

    

    r2 = model_.score(X_test_encoded,y_test)

    r2_values_test.append(r2)

       

    

    rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))   

    rmse_values_test.append(rmse_test)

    

    mse_test=mean_squared_error(y_test,y_pred)

    mse_values_test.append(mse_test)

    

result=pd.DataFrame(list(zip(r2_values_test,r2_values_train)),columns=["r2_score_test","r2_score_train"])

result["rmse_test"] =rmse_values_test

result["mse_test"]=mse_values_test

result["model"]=["XGBoost","LGBM","RF","ExtraTree","HGBoost"]  

result
xgb_=XGBRegressor(random_state=42)

xgb_model=xgb_.fit(X_train_encoded,y_train)

preds=xgb_model.predict(X_test_encoded)
xgb_model.score(X_train_encoded,y_train),xgb_model.score(X_test_encoded,y_test)
y_test.mean(),y_test.std()

preds.mean(),preds.std()
pd.DataFrame(list(zip(y_test,preds)),columns=["test","preds"]).head(10)
ax=plot_importance(xgb_model,max_num_features=15,height=0.5);

fig = ax.figure

fig.set_size_inches(12, 5)
xgb.to_graphviz(xgb_model)
X_train,X_test,y_train,y_test=train_test_split(X,y,

                                               test_size=0.20,

                                                random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.head()
X_test.head()
X_train["year"]=2020-X_train["year"]

X_test["year"]=2020-X_test["year"]

X_train.head()
X_test.head()
pipeline_used_car=Pipeline([

    

    ("encoder_rare_label",RareLabelCategoricalEncoder(tol=0.05,n_categories=7, variables=["manufacturer","state","title_status"])),

    

    ("encoder_rare_label_",RareLabelCategoricalEncoder(tol=0.05, n_categories=7, variables=["title_status"],replace_with='NotClean')),  

    

    ("categorical_encoder",OrdinalCategoricalEncoder(encoding_method='ordered',variables=['condition'])),

    

    ("categorical_encoder_",OneHotCategoricalEncoder(drop_last=False)),

    

    ("xgb",XGBRegressor(random_state=42))

    

    

        

])
pipeline=make_pipeline(RareLabelCategoricalEncoder(tol=0.05,n_categories=7, variables=["manufacturer","state","title_status"]),

                     RareLabelCategoricalEncoder(tol=0.05, n_categories=7, variables=["title_status"],replace_with='NotClean'),

                      OrdinalCategoricalEncoder(encoding_method='ordered',variables=['condition']),

                      OneHotCategoricalEncoder(drop_last=False),

                      XGBRegressor(random_state=42))
pipeline_used_car.fit(X_train,y_train)
pipeline.fit(X_train,y_train)
preds=pipeline_used_car.predict(X_test)

r2_score(y_test,preds)
predicts=pipeline.predict(X_test)

r2_score(y_test,predicts)
param_grid={'categorical_encoder__encoding_method': ['ordered', 'arbitrary'],

            

             'xgb__max_depth': [None, 1, 3]}
grid_search = GridSearchCV(pipeline_used_car, param_grid,cv=5,n_jobs=-1,scoring="r2")
grid_search.fit(X_train, y_train)
print(("best score from grid search: %.3f" % grid_search.score(X_train, y_train)))
grid_search.best_estimator_
grid_search.cv_results_['params']
grid_search.cv_results_['mean_test_score']
print(("best xgboost regressor from grid search: %.3f"% grid_search.score(X_test, y_test)))