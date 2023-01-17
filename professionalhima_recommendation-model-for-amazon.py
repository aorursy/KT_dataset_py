import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import surprise
df = pd.read_csv('../input/Amazon - Movies and TV Ratings.csv')
df.head()
df.shape
df_org = df.copy()
df.describe().T
#Movie with highest views
df.describe().T['count'].sort_values(ascending=False)[:1].to_frame() #---Movie127
#Movie with highest Ratings
df.drop('user_id',axis=1).sum().sort_values(ascending=False)[:1].to_frame()  #---Movie127
df.drop('user_id',axis=1).mean().sort_values(ascending=False)[:5].to_frame()
df.describe().T['count'].sort_values(ascending=True)[:5].to_frame()
from surprise import Reader
from surprise import accuracy
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import cross_validate
df_melt = df.melt(id_vars = df.columns[0],value_vars=df.columns[1:],var_name="Movies",value_name="Rating")
df_melt
rd = Reader()
data = Dataset.load_from_df(df_melt.fillna(0),reader=rd)
data
trainset, testset = train_test_split(data,test_size=0.25)
#Using SVD (Singular Value Descomposition)
svd = SVD()
svd.fit(trainset)
pred = svd.test(testset)
accuracy.rmse(pred)
accuracy.mae(pred)
cross_validate(svd, data, measures = ['RMSE', 'MAE'], cv = 3, verbose = True)
def repeat(ml_type,dframe):
    rd = Reader()
    data = Dataset.load_from_df(dframe,reader=rd)
    print(cross_validate(ml_type, data, measures = ['RMSE', 'MAE'], cv = 3, verbose = True))
    print("--"*15)
    usr_id = 'A3R5OBKS7OM2IR'
    mv = 'Movie1'
    r_u = 5.0
    print(ml_type.predict(usr_id,mv,r_ui = r_u,verbose=True))
    print("--"*15)

repeat(SVD(),df_melt.fillna(df_melt['Rating'].mean()))
#repeat(SVD(),df_melt.fillna(df_melt['Rating'].median()))
#trying grid search and find optimum hyperparameter value for n_factors
from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs':[20,30],
             'lr_all':[0.005,0.001],
             'n_factors':[50,100]}
gs = GridSearchCV(SVD,param_grid,measures=['rmse','mae'],cv=3)
data1 = Dataset.load_from_df(df_melt.fillna(df_melt['Rating'].mean()),reader=rd)
gs.fit(data1)
gs.best_score
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
