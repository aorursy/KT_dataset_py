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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

from scipy import stats

import plotly.express as ex
r_data = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
r_data.head(3)
r_data.isna().sum().to_frame()
r_data.Style.fillna(r_data.Style.mode()[0],inplace=True)
Style_dict = {r_data.Style.unique()[i]: i+1 for i in range(0,len(r_data.Style.unique()))}

Brand_dict = {r_data.Brand.unique()[i]: i+1 for i in range(0,len(r_data.Brand.unique()))}

Country_dict = {r_data.Country.unique()[i]: i+1 for i in range(0,len(r_data.Country.unique()))}



r_data.Style.replace(Style_dict,inplace=True)

r_data.Brand.replace(Brand_dict,inplace=True)

r_data.Country.replace(Country_dict,inplace=True)
r_data = r_data[r_data['Stars'] != 'Unrated']

r_data.head(3)
#Variety Feature Extraction



#number of words extraction

def number_of_words(sir):

    return len(sir.split(' '))

r_data['Number_Of_Words'] = r_data.Variety.apply(number_of_words)



meat_types = ['chicken','beef','duck','pork','shrimp','turkey']

noodle_types = ['udon','soba','ramen','egg','shirataki','hokkien','noodles']



def contain_flavor(sir):

    stn = sir.lower()

    return 1 if 'flavor' in stn else 0



def meat_scanner(sir):

    stn = sir.lower()

    for meat in meat_types:

        if meat in stn:

            return meat

    return 'unspecified'



def noodle_scanner(sir):

    stn = sir.lower()

    for noodle in noodle_types:

        if noodle in stn:

            return noodle

    return 'unspecified'

    

    

def is_spicy(sir):

    stn = sir.lower()

    spc = ['spicy','hot','flaming','chili']

    for t in spc:

        if t in stn:

            return 1

    return 0



r_data['Contains_Flavor'] = r_data.Variety.apply(contain_flavor)

r_data['Meat_Type'] = r_data.Variety.apply(meat_scanner)

r_data['Noodle_Type'] = r_data.Variety.apply(noodle_scanner)

r_data['Is_Spicy'] = r_data.Variety.apply(is_spicy)



meat_type_dict = {r_data['Meat_Type'].unique()[i]: i+1 for i in range(0,len(r_data['Meat_Type'].unique()))}

noodle_type_dict = {r_data['Noodle_Type'].unique()[i]: i+1 for i in range(0,len(r_data['Noodle_Type'].unique()))}



r_data['Meat_Type'].replace(meat_type_dict,inplace=True)

r_data['Noodle_Type'].replace(noodle_type_dict,inplace=True)
r_data



w_data = r_data[['Brand','Style','Country','Number_Of_Words','Contains_Flavor','Meat_Type',

                'Noodle_Type','Is_Spicy','Stars']].copy()
w_data.head(10)
Brand_dict = {key : val for val,key in Brand_dict.items()}

Country_dict = {key : val for val,key in Country_dict.items()}

Style_dict = {key : val for val,key in Style_dict.items()}

noodle_type_dict= {key : val for val,key in noodle_type_dict.items()}

meat_type_dict =  {key : val for val,key in meat_type_dict.items()}
plt.figure(figsize=(20,11))

top_25_brands =w_data['Brand'].value_counts().sort_values(ascending=False)[:25]

ax = sns.countplot(w_data[w_data['Brand'].isin(top_25_brands.index)]['Brand'],order = top_25_brands.index,

                  palette='mako')





ax.set_xticklabels([Brand_dict[int(ax.get_xticklabels()[i].get_text())] for i in range(0,25)],rotation=90)



ax.patches[0].set_fc('r')

ax.patches[1].set_fc((0.7,0.0,0.0))

ax.patches[2].set_fc((0.5,0.0,0.0))

ax.set_title('Distribution of the number of prodcuts made by different brands [Top 25]',fontsize=18)

plt.show()

w_data.Stars = w_data.Stars.astype('float64')

pivot = w_data.groupby(by='Country').sum()

pivot = pivot.sort_values(by='Stars',ascending=False)

top_25_country_by_score = pivot[:25]

plt.figure(figsize=(20,11))

ax = sns.barplot(pivot.index,pivot.Stars,order = top_25_country_by_score.index,palette='mako')

ax.set_xticklabels([Country_dict[int(ax.get_xticklabels()[i].get_text())] for i in range(0,25)],rotation=90)

ax.patches[0].set_fc('r')

ax.patches[1].set_fc((0.7,0.0,0.0))

ax.patches[2].set_fc((0.5,0.0,0.0))

ax.set_title('Distribution of prodcuts made by different countries [Top 25]',fontsize=18)



plt.show()
plt.figure(figsize=(20,11))

for t in Style_dict.keys():

    ax = sns.distplot(w_data[w_data['Style']==t]['Stars'],hist=False,label = Style_dict[t],kde_kws={'lw':4} )

plt.legend(prop={'size':20})

ax.set_title('Distributions of different ramen style rankings ',fontsize=18)

pivot = w_data.groupby(by='Style').mean()

pivot = pivot.sort_values(by='Stars',ascending=False)





plt.figure(figsize=(20,11))

ax = sns.barplot(x=pivot.index,y=pivot['Stars'],order=pivot.index,palette='mako')

ax.set_xticklabels([Style_dict[int(ax.get_xticklabels()[i].get_text())] for i in range(0,7)],rotation=90)

ax.patches[0].set_fc('r')

ax.patches[1].set_fc((0.7,0.0,0.0))

ax.patches[2].set_fc((0.5,0.0,0.0))

ax.set_title('Distribution of average rankings across different styles',fontsize=18)



plt.show()
plt.figure(figsize=(20,11))

ax = sns.boxplot(x='Style',y='Stars',hue='Meat_Type',data=w_data)

l = plt.legend()

for i in range(0,6):

    l.get_texts()[i].set_text(list(meat_type_dict.values())[i])



    

ax.set_title('Distribution of rankings of different meat types across different styles',fontsize=18)    

plt.show()
labels = {x+1:list(noodle_type_dict.values())[x] for x in range(0,len(noodle_type_dict.values()))}

tmp = w_data.copy()

tmp.Noodle_Type = tmp.Noodle_Type.replace(labels)

ex.box(tmp,x='Style',y='Stars',color='Noodle_Type',title='Distribution of rankings of different noodle types across different styles',height=800,

      labels=labels)



plt.figure(figsize=(20,11))

ax = sns.distplot(w_data[w_data['Is_Spicy']==1]['Stars'],hist=False,label = "Spicy",kde_kws={'lw':4} )

ax = sns.distplot(w_data[w_data['Is_Spicy']==0]['Stars'],hist=False,label = "Not Spicy",kde_kws={'lw':4} )





plt.legend(prop={'size':20})



ax.set_title('Distribution of stars in spicy and not spicy products',fontsize=18)    

plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(w_data[w_data['Contains_Flavor']==1]['Stars'],hist=False,label = "Flavor In Var",kde_kws={'lw':4} )

ax = sns.distplot(w_data[w_data['Contains_Flavor']==0]['Stars'],hist=False,label = "Flavor Not In Var",kde_kws={'lw':4} )



plt.legend(prop={'size':20})





ax.set_title('Distribution of stars in labels containg the word Flavor and not',fontsize=18)    

plt.show()
plt.figure(figsize=(20,11))

ax = sns.jointplot(w_data['Stars'],w_data['Number_Of_Words'],kind='kde',cmap='mako',height=14)

meats = pd.get_dummies(w_data['Meat_Type'])

meats = meats.rename(meat_type_dict,axis=1)

w_data = w_data.join(meats[1:])

w_data.drop('Meat_Type',axis=1,inplace=True)

corrs = w_data.corr('pearson')

plt.figure(figsize=(20,11))

ax = sns.heatmap(corrs,annot=True,cmap='mako')
from sklearn.feature_selection import chi2,f_classif,SelectKBest

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor
selector = SelectKBest(f_classif,k=5)

w_data = w_data.fillna(0)

cols = [w_data.columns[i] for i in range(0,len(w_data.columns)) if w_data.columns[i] != 'Stars']

X = selector.fit_transform(w_data[cols],w_data['Stars'])



selected_columns = [cols[i] for i in range(0,len(cols)) if selector.get_support()[i] == True]

selected_columns



x_train,x_test,y_train,y_test = train_test_split(X,w_data['Stars'])



xgb_model = XGBRegressor(n_estimators=300,learning_rate=0.03,random_state=42)

xgb_model.fit(x_train,y_train,early_stopping_rounds=7,eval_set=[(x_test[:5],y_test[:5])],verbose=False)

predictions = xgb_model.predict(x_test)

#xgb_model.score(x_test,y_test)

xgb_score = np.sqrt(mean_squared_error(predictions,y_test))

xgb_score



print('XGB RMSE: ' ,xgb_score)
LR_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',LinearRegression())])

cv_scores = -1*cross_val_score(LR_pipe,x_train,y_train,cv=5,scoring='neg_mean_squared_error')

cv_scores.mean()



print('LinearRegression Cross Validation RMSE: ' ,cv_scores.mean())
LR_pipe.fit(x_train,y_train)

LR_predictions = LR_pipe.predict(x_test)

LR_score = np.sqrt(mean_squared_error(LR_predictions,y_test))

LR_score



print('LinearRegression RMSE: ' ,LR_score)
RF_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',RandomForestRegressor(random_state=42,

                                                                                    n_estimators=60,max_leaf_nodes=22))])

cv_scores = -1*cross_val_score(RF_pipe,x_train,y_train,cv=5,scoring='neg_mean_squared_error')

cv_scores.mean()



print('RandomForest Cross Validation RMSE: ' ,cv_scores.mean())
RF_pipe.fit(x_train,y_train)

RF_predictions = RF_pipe.predict(x_test)

RF_score = np.sqrt(mean_squared_error(RF_predictions,y_test))

RF_score



print('RandomForest RMSE: ' ,RF_score)
ADA_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',AdaBoostRegressor(random_state=42,

                                                                                 learning_rate=0.3,n_estimators=30))])

cv_scores = -1*cross_val_score(ADA_pipe,x_train,y_train,cv=5,scoring='neg_mean_squared_error')

cv_scores.mean()



print('AdaBoost Cross Validation RMSE: ' ,cv_scores.mean())
ADA_pipe.fit(x_train,y_train)

ADA_predictions = ADA_pipe.predict(x_test)

ADA_score = np.sqrt(mean_squared_error(ADA_predictions,y_test))

ADA_score



print('AdaBoost RMSE: ' ,ADA_score)
KNN_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',KNeighborsRegressor(n_neighbors=24))])

cv_scores = -1*cross_val_score(KNN_pipe,x_train,y_train,cv=5,scoring='neg_mean_squared_error')

cv_scores.mean()

KNN_pipe.fit(X,w_data['Stars'])



print('KNN Cross Validation RMSE: ' ,cv_scores.mean())

xgb_model.fit(X,w_data['Stars'])

ADA_pipe.fit(X,w_data['Stars'])

LR_pipe.fit(X,w_data['Stars'])

RF_pipe.fit(X,w_data['Stars'])

glb = xgb_model.predict(X)

ada = ADA_pipe.predict(X)

rf = RF_pipe.predict(X)

lr = LR_pipe.predict(X)

knn = KNN_pipe.predict(X)
plt.figure(figsize=(20,11))

ax=sns.lineplot(y=w_data['Stars'],x=np.arange(0,w_data.shape[0]),label='Actual')

ax=sns.lineplot(y=glb,x=np.arange(0,w_data.shape[0]),label='XGB Prediction')

ax=sns.lineplot(y=ada,x=np.arange(0,w_data.shape[0]),label='ADABoost Prediction')

ax=sns.lineplot(y=rf,x=np.arange(0,w_data.shape[0]),label='RandomForest Prediction')

ax=sns.lineplot(y=lr,x=np.arange(0,w_data.shape[0]),label='LinearRegression Prediction')

ax=sns.lineplot(y=lr,x=np.arange(0,w_data.shape[0]),label='KNN Prediction')







ax.set_title('Predicted values against real values',fontsize=20)

plt.show()
fp = (glb*0.60+rf*0.21 + knn*0.2)
print('RMSE Of Stacked Predictions: ',np.sqrt(mean_squared_error(w_data['Stars'],fp)))
plt.figure(figsize=(20,11))

ax=sns.lineplot(y=w_data['Stars'],x=np.arange(0,w_data.shape[0]),label='Actual')

ax=sns.lineplot(y=fp,x=np.arange(0,w_data.shape[0]),label='Stacked Prediction')



ax.set_title('Final Prediction values against real values',fontsize=20)

plt.show()