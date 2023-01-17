import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/housing-simple-regression/Housing.csv')
df.head()
df.info()
target_df = pd.DataFrame(df['price'])
target_df.head()
df.head()
df.drop('price',axis=1,inplace=True)
df.head()
def load_file(file):

    return pd.read_csv(file)



def consolidate_data(df1,df2,key=None,left_index=False,right_index=False):

    return pd.merge(left=df1,right=df2,how='inner',on=key,left_index=left_index,right_index=right_index)



def clean_df(raw_df):

    clean_df = raw_df.drop_duplicates()

    #clean_df = clean_df(clean_df.price>0)

    return clean_df



    

def one_hot_encode_feature_df(df,num_vars=None,map_vars=None,cat_vars=None):

    num_df = df[num_vars]

    map_df = df[map_vars].replace({'no': 0, 'yes': 1})

    cat_df = pd.get_dummies(df[cat_vars],drop_first=True)

    return pd.concat([num_df,map_df,cat_df],axis=1)





def get_target_df(df,target):

    return df[target]



def train_model(model,feature_df,target_df,num_procs,mean_mse,cv_std):

    neg_mse = cross_val_score(model,feature_df,target_df,cv=2,n_jobs=num_procs,scoring='neg_mean_squared_error')

    mean_mse[model] = -1.0*np.mean(neg_mse)

    cv_std[model] = np.std(neg_mse)

    

def print_summary(model,mean_mse,cv_std):

    print('\nModel:\n',model)

    print('Avearge MSE:\n',mean_mse[model])

    print('Standard Deviation during CV:\n',cv_std[model])

    

def save_results(model,mean_mse,predictions,feature_importances):

    with open('model.txt','w') as file:

        file.write(str(model))

    feature_importances.to_csv('feature_importances.csv')

    np.savetxt('predictions.csv',predictions,delimiter=',')    
train_feature_file = '/kaggle/input/housing-simple-regression/Housing.csv'



categories_vars = ['furnishingstatus']

numeric_vars    = ['area','bedrooms','bathrooms','stories','parking']

mapping_vars    = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']

target_var      = ['price']



print("Loading data")

feature_df = load_file(train_feature_file)



#raw_df_train = consolidate_data(feature_df,target_df)



#clean_train_df = shuffle(clean_data(raw_df_train)).reset_index()





print("Encoding data")

feature_df = one_hot_encode_feature_df(df,num_vars=numeric_vars,map_vars=mapping_vars,cat_vars=categories_vars)
feature_df.head()
target_df.head()
models =   []

mean_mse = {}

cv_std   = {}

res      = {}



num_procs = 5



verbose_lvl = 5
lr = LinearRegression()

lr_std_pca = make_pipeline(StandardScaler(),PCA(),LinearRegression())

rf = RandomForestRegressor(n_estimators=100,n_jobs=num_procs,max_depth=10,min_samples_split=20,\

                          max_features='auto',verbose=verbose_lvl)

gbm = GradientBoostingRegressor(n_estimators=100,max_depth=5,loss='ls',verbose=verbose_lvl)



models.extend([lr,lr_std_pca,rf,gbm])



print("Beginning Cross Validate")

for model in models:

    train_model(model,feature_df,target_df,num_procs,mean_mse,cv_std)

    print_summary(model,mean_mse,cv_std)
model = min(mean_mse,key=mean_mse.get)

print('\nPrediction calculated model using lowest MSE:')

print(model)



model.fit(feature_df,target_df)

predictions = model.predict(feature_df)