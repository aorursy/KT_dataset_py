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
#pandas and numpy

import pandas as pd

import numpy as np



# Matplotlib and Seaborn

import matplotlib.pyplot as plt

import seaborn as sns



# Plotly Packages

from plotly import tools

!pip install chart_studio

import chart_studio.plotly as ply



 

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot#renderer framework



init_notebook_mode(connected=True)# to display figures in the notebook



#library to convert latitude and longitude values

from scipy.cluster.vq import kmeans2, whiten

#other ibraries

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")



df.head()
df.describe()
#A new attribute "Region" is added based on Latitude and Longitude

np.random.seed(12345678)

coordinates = np.array(df[['X5 latitude','X6 longitude']])





x,df['Region'] = kmeans2(whiten(coordinates), 4, iter = 20,) 

df['Region'].loc[df['Region']==0]='Region0'

df['Region'].loc[df['Region']==1]='Region1'

df['Region'].loc[df['Region']==2]='Region2'

df['Region'].loc[df['Region']==3]='Region3'
#rename the columns

df = df.rename(columns={'X1 transaction date':'Transact_Dt','X2 house age':'House_Age',

                        'X3 distance to the nearest MRT station':'Distance_Mrt',

                        'X4 number of convenience stores':'Stores_nearby',

                        'Y house price of unit area':'price_per_unitarea'

                       })
#New categorical variable age_cat is added as 'NewHome' <15 years, 'MiddleAgedHome' between 15 to 25,

#'OldHome' > 25 based on age of the house

df['house_age_cat']=np.nan





for homes in [df]:

    homes.loc[homes['House_Age'] <15,'house_age_cat'] = 'NewHome'

    homes.loc[(homes['House_Age'] >=15) &(homes['House_Age'] <30),'house_age_cat']='MiddleAgedHome'

    homes.loc[homes['House_Age'] >=30,'house_age_cat']='OldHome'

#A new categorical variable store_cat is added based on the number of Nearby Stores

df['store_cat']=np.nan





for homes in [df]:

    homes.loc[homes['Stores_nearby'] <5,'store_cat'] = 'Less'

    homes.loc[homes['Stores_nearby'] >=5,'store_cat']='More'

    

#A new categorical variable mrt_dist_cat based on the distance to nearest Metro Station



df['mrt_dist_cat']=np.nan





for homes in [df]:

    homes.loc[homes['Distance_Mrt'] <500,'mrt_dist_cat'] = 'closeby'

    homes.loc[(homes['Distance_Mrt'] >=500) &(homes['Distance_Mrt'] <1250),'mrt_dist_cat']='ShortDistance'

    homes.loc[homes['Distance_Mrt'] >=1000,'mrt_dist_cat']='LongDistance'
from datetime import datetime

def convert_date(dt):

    yyyy = int(dt)

    mm =int((dt - yyyy)*12)

    if mm ==0:

        mm=1

    datestr=str(yyyy)+str(mm).rjust(2,'0')+'01'

    return datestr



    



df['Transact_Dt1']=list(map(convert_date, df['Transact_Dt']))

df['DaysElapsed'] = list(map(lambda x :(datetime.now()-datetime.strptime(x,'%Y%m%d')).days, df['Transact_Dt1']))
#drop the columns not used

original_df=df.copy()

df=df.drop(columns={'Transact_Dt','No','X5 latitude','X6 longitude'}, axis=1)

original_df=original_df.drop(columns={'No','Region','house_age_cat','store_cat','mrt_dist_cat','Transact_Dt','Transact_Dt1'}, axis=1)

#original_df=original_df.drop(columns={'Transact_Dt','No','X5 latitude','X6 longitude'}, axis=1)

original_df.head()
#check the distribution of the unit price. We can determine if the price is normally distributed or skewed. 

# We can check the log distribution to eliminate skewness

price_list= df['price_per_unitarea'].values

price_list_log = np.log(df['price_per_unitarea'])
trace0 = go.Histogram(

                     x=price_list,

                     histnorm='probability',

                     name="Unit Price Distribution",

                     marker = dict(color = '#FA5858',)

                    )



trace1 = go.Histogram(

    x=price_list_log,

    histnorm='probability',

    name="Unit Price Distribution using Log",

    marker = dict(

        color = '#58FA82',

    )

)

fig = tools.make_subplots(rows=1, cols=2,

                         subplot_titles=('Price Distribution','Log Price Distribution'),

                         print_grid=False)

fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)

fig['layout'].update(showlegend=True, title='Unit House Price Distribution',bargap=0.05)

iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


#pie chart of home categories

labels= df['house_age_cat'].unique().tolist()

amount= df['house_age_cat'].value_counts().tolist()



colors=['#FA5858','#58FA82','#e6ffb3']



trace = go.Pie(labels=labels,values=amount,

              hoverinfo='label+percent',

              textinfo='value',

              textfont=dict(size=20),

              marker = dict(colors=colors, line=dict(color='#000000',width=2)))



layout = go.Layout(title="Amount by Age Category")

fig = go.Figure(data=[trace],layout=layout)

iplot(fig, filename='basic_pie_chart')
House_age= [df['House_Age'].values.tolist()]

label=["Age of House Distribution"]



colors=["#b3d9ff"]

fig = ff.create_distplot(House_age,label,colors=colors)

fig['layout'].update(title='Normal Distribution <br> Central Limit Theorem Condition')

iplot(fig,filename='Basic Distplot')
corr=df.corr()

hm=go.Heatmap(z=corr.values,

             x=corr.index.values.tolist(),

             y=corr.index.values.tolist())

data=[hm]

layout=go.Layout(title="Correlation Heatmap")

fig = dict(data=data, layout=layout)

iplot(fig, filename='labelled-heatmap')
#Does  the number of convenince stores depends on region

stores_r0 = df['Stores_nearby'].loc[df['Region']=='Region0'].values

stores_r1 = df['Stores_nearby'].loc[df['Region']=='Region1'].values

stores_r2 = df['Stores_nearby'].loc[df['Region']=='Region2'].values

stores_r3 = df['Stores_nearby'].loc[df['Region']=='Region3'].values



trace0 = go.Box(

    y=stores_r0,

    name = 'region 0',

    boxmean= True,

    marker = dict(

        color = 'rgb(214, 12, 140)',

    )

)

trace1 = go.Box(

    y=stores_r1,

    name = 'region 1',

    boxmean= True,

    marker = dict(

        color = 'rgb(0, 128, 128)',

    )

)



trace2 = go.Box(

    y=stores_r2,

    name = 'region 2',

    boxmean= True,

    marker = dict(

        color = 'rgb(247, 186, 166)',

    )

)



trace3 = go.Box(

    y=stores_r3,

    name = 'region 3',

    boxmean= True,

    marker = dict(

        color = 'rgb(247, 186, 166)',

    )

)

data = [trace0, trace1, trace2,trace3]



layout = go.Layout(title="Convenience Stores <br> by region", xaxis=dict(title="Region", titlefont=dict(size=16)),

                  yaxis=dict(title="Stores", titlefont=dict(size=16)))



fig = go.Figure(data=data, layout=layout)

iplot(fig)
import statsmodels.api as sm

from statsmodels.formula.api import ols





lm = ols("Stores_nearby ~ Region", data=df).fit()

print(lm.summary())
# Create subpplots

f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,8))





sns.stripplot(x="store_cat", y="price_per_unitarea", hue="house_age_cat",data=df, ax=ax1, linewidth=1, palette="Reds")

ax1.set_title("Relationship between Price & Stores by house age")





sns.stripplot(x="mrt_dist_cat", y="price_per_unitarea", hue="house_age_cat", data=df, ax=ax2, linewidth=1, palette="Set1")

ax2.set_title("Relationship of Price & Distance to mrt by house age")



sns.stripplot(x="Region", y="price_per_unitarea", hue="house_age_cat", data=df, ax=ax3, linewidth=1, palette="Set3")

ax3.set_title("Relationship between Price & Region and house age")



plt.show() 
fig = ff.create_facet_grid(

    df,

    x='Stores_nearby',

    y='price_per_unitarea',

    color_name='Region',

    show_boxes=False,

    marker={'size': 10, 'opacity': 1.0},

    colormap={'Region0': 'rgb(255, 0, 0)', 'Region1': 'rgb(0, 255, 0)',

             'Region2': 'rgb(0, 0, 255)','Region3':'rgb(255,255,255)'}

)

251, 232, 238





fig['layout'].update(title="Price vs Stores by Region", width=800, height=600, plot_bgcolor='rgb(251, 251, 251)', 

                     paper_bgcolor='rgb(255, 255, 255)')





iplot(fig, filename='facet - custom colormap')
pointspos1 = [-0.9,-1.1,-0.6,-0.3]

pointspos2 = [0.45,0.55,1,0.4]

showLegend = [True,False,False,False]

 

data = []

for i in range(0,len(pd.unique(df['Region']))):

    male = {

            "type": 'violin',

            "x": df['Region'][(df['store_cat'] == 'Less') &  (df['Region'] == pd.unique(df['Region'])[i])],

            "y": df['price_per_unitarea'][  (df['Region'] == pd.unique(df['Region'])[i]) ],

            "legendgroup": 'Less Stores',

            "scalegroup": 'Less Stores',

            "name": 'Less Stores',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "points": 'all',

            "pointpos": pointspos1[i],

            "jitter": 0,

            "scalemode": 'count',

            "meanline": {

                "visible": True

            },

            "line": {

                "color": '#DF0101'

            },

            "marker": {

                "line": {

                    "width": 2,

                    "color": '#F78181'

                }

            },

            "span": [

                0

            ],

            "showlegend": showLegend[i]

        

    }

    data.append(male)

   

    female = {

            "type": 'violin',

            "x": df['Region'][(df['store_cat'] == 'More') &  (df['Region'] == pd.unique(df['Region'])[i])],

            "y": df['price_per_unitarea'][  (df['Region'] == pd.unique(df['Region'])[i]) ],



            "legendgroup": 'More Stores',

            "scalegroup": 'More Stores',

            "name": 'More Stores',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "points": 'all',

            "pointpos": pointspos2[i],

            "jitter": 0,

            "scalemode": 'count',

            "meanline": {

                "visible": True

            },

            "line": {

                "color": '#00FF40'

            },

            "marker": {

                "line": {

                    "width": 2,

                    "color": '#81F781'

                }

            },

            "span": [

                0

            ],

            "showlegend": showLegend[i]

        

    }

    data.append(female)

          



fig = {

    "data": data,

    "layout" : {

        "title": "Price Distribution by Region and  Number of Stores",

        "yaxis": {

            "zeroline": False,

            "title": "Home Price per Unit Area",

            "titlefont": {

                "size": 16

            }

        },

        "violingap": 0,

        "violingroupgap": 0,

        "violinmode": "overlay"

    }

}





iplot(fig, filename='violin/advanced', validate = False)
#Region2 and Region 3 distributions 

price_r3_more=df.loc[(df['Region']=='Region3') & (df['store_cat'] == 'More'),'price_per_unitarea']

price_r3_less=df.loc[(df['Region']=='Region3') & (df['store_cat'] == 'Less'),'price_per_unitarea']



price_r2_more=df.loc[(df['Region']=='Region2') & (df['store_cat'] == 'More'),'price_per_unitarea']

price_r2_less=df.loc[(df['Region']=='Region2') & (df['store_cat'] == 'Less'),'price_per_unitarea']



trace0 = go.Box(y=price_r3_more,

               name='Region3 with More Stores',

               marker=dict(color='#3D9970'))



trace1 = go.Box(y=price_r3_less,

               name='Region3 with Less Stores',

               marker=dict(color='#FF4136'))



trace2 = go.Box(y=price_r2_more,

               name='Region2 with More Stores',

               marker=dict(color='#1121F8'))



trace3 = go.Box(y=price_r2_less,

               name='Region2 with Less Stores',

               marker=dict(color='#C362C2'))



fig = go.Figure()



fig.add_trace(trace0)

fig.add_trace(trace1)

fig.add_trace(trace2)

fig.add_trace(trace3)



fig.update_layout(title='Deeper Look Into region 2 and Region 3 Prices',

                  xaxis=dict(title='Stores'),

                   yaxis= dict(title='Price Per unit Area'))

fig.show()
r3_more_stores= df.loc[(df['Region']=='Region3') & (df['store_cat'] == 'More')]

r3_less_stores = df.loc[(df['Region']=='Region3') & (df['store_cat'] == 'Less')]



trace0 = go.Scatter(

    y=r3_more_stores['price_per_unitarea'],

    x=r3_more_stores['House_Age'],

    name='More Stores',

    mode='markers',

    marker=dict(size=10,

                color='#DF0101')

)

trace1 = go.Scatter(

    y=r3_less_stores['price_per_unitarea'],

    x=r3_less_stores['House_Age'],

    name='Less Stores',

    mode='markers',

    marker=dict(size=10,

                color='#00FF40')

)



data=[trace0, trace1]



layout=dict(

    title='Influence of Nearby Stores in Home Price in Region 3',

    yaxis=dict(zeroline=False,

              title='Price Per Unit Area',

              titlefont=dict(

                  size=16)

              ),

    xaxis=dict(zeroline=False,

              title='Age of the House',

              titlefont=dict(

                  size=16)

              )

)



fig = go.Figure(data=data, layout=layout)

fig.show()
r2_more_stores= df.loc[(df['Region']=='Region2') & (df['store_cat'] == 'More')]

r2_less_stores = df.loc[(df['Region']=='Region2') & (df['store_cat'] == 'Less')]



trace0 = go.Scatter(

    y=r2_more_stores['price_per_unitarea'],

    x=r2_more_stores['House_Age'],

    name='More Stores',

    mode='markers',

    marker=dict(size=10,

                color='#DF0101')

)

trace1 = go.Scatter(

    y=r2_less_stores['price_per_unitarea'],

    x=r2_less_stores['House_Age'],

    name='Less Stores',

    mode='markers',

    marker=dict(size=10,

                color='#00FF40')

)



data=[trace0, trace1]



layout=dict(

    title='Influence of Nearby Stores in Home Price in Region 2',

    yaxis=dict(zeroline=False,

              title='Price Per Unit Area',

              titlefont=dict(

                  size=16)

              ),

    xaxis=dict(zeroline=False,

              title='Age of the House',

              titlefont=dict(

                  size=16)

              )

)



fig = go.Figure(data=data, layout=layout)

fig.show()
#creating contigency table

region_stores = pd.crosstab(df['store_cat'],df['Region']) 

region_stores
plt.style.use('seaborn-whitegrid')

ticks = df['house_age_cat'].unique()

colors = ['#ff2424', '#90ee90']



ax=sns.catplot(x="house_age_cat",y="price_per_unitarea",hue="store_cat",

              col="mrt_dist_cat",data=df,palette=colors,aspect=0.6,kind="swarm")

ax.set_xticklabels(labels = ticks, rotation=45)



plt.show()
plt.style.use('seaborn-whitegrid')

ticks = df['Region'].unique()

colors = ['#ff2424', '#90ee90']



ax=sns.catplot(x="Region",y="price_per_unitarea",hue="store_cat",

              col="mrt_dist_cat",data=df,palette=colors,aspect=0.6,kind="swarm")

ax.set_xticklabels(labels = ticks, rotation=45)



plt.show()
#Median home price by region

r0price=np.median(df.loc[df['Region']=='Region0','price_per_unitarea' ]) 

r1price=np.median(df.loc[df['Region']=='Region1' ,'price_per_unitarea'])

r2price=np.median(df.loc[df['Region']=='Region2' ,'price_per_unitarea'])

r3price=np.median(df.loc[df['Region']=='Region3' ,'price_per_unitarea'])



lst=[r0price,r1price,r2price,r3price]

data = [go.Scatterpolar(

  r = lst,

  theta = ['Region0', 'Region1', 'Region2', 'Region3'],

  fill = 'toself'

)]



layout = go.Layout(

    title="Median Home Price by Region",

    paper_bgcolor = "rgb(255, 255, 224)",

  polar = dict(

    radialaxis = dict(

      visible = False,

      range = [0, max(lst)]

    )

  ),

  showlegend = False

)





fig = go.Figure(data=data, layout=layout)

fig.show()
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))

sns.scatterplot(x="Distance_Mrt", y="price_per_unitarea", hue="store_cat", data=df, palette="Set1", ax=ax1)

ax1.set_title("Price Vs Distance to MRT by Stores")

ax1.annotate('Cluster 1 \n Homes with nearby MRT \n and more Stores \n', xy=(500, 60), xytext=(1000, 80),

             arrowprops=dict(facecolor='black'),

             fontsize=12)

ax1.annotate('Cluster 2 \n Homes with closeby MRT\n and less Stores \n', xy=(1500, 30), xytext=(2000, 50),

             arrowprops=dict(facecolor='blue'),

             fontsize=12)

ax1.annotate('Cluster 3 \nHomes with far MRT\n and less Stores \n', xy=(4000, 20), xytext=(5000, 40),

             arrowprops=dict(facecolor='green'),

             fontsize=12)



sns.scatterplot(x="House_Age", y="price_per_unitarea", hue="store_cat", data=df, palette="Set1", ax=ax2)

ax2.set_title(" Price Vs HouseAge by distance to nearest MRT")

ax2.annotate('Cluster1 \n New homes \n with more Stores', xy=(5,65), xytext=(10, 80),

            arrowprops=dict(facecolor='black'),

            fontsize=12)

ax2.annotate('Cluster2\n New homes \n with Less Stores', xy=(12,20), xytext=(1, 5),

            arrowprops=dict(facecolor='Green'),

            fontsize=12)

ax2.annotate('Cluster3 \nOld homes \n with More Stores', xy=(35,50), xytext=(40, 55),

            arrowprops=dict(facecolor='Red'),

            fontsize=12)

# In this section we will preprocess our data

# First we should split our original data.



from sklearn.model_selection import train_test_split,KFold,cross_val_score



# Shuffle our dataset before splitting

np.random.seed(42)

original_df = original_df.sample(frac=1, random_state=1)



X = original_df.drop(columns={'price_per_unitarea'}, axis=1)

y = original_df["price_per_unitarea"]



# Split into both training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import OneHotEncoder





# Separate numerics and categorical values

numerics = X_train.select_dtypes(exclude="object")

categoricals = X_train.select_dtypes(include="object")



# Pipelines

numerical_pipeline = Pipeline([

    ("select_numeric", DataFrameSelector(numerics.columns.tolist())),

    ("std_scaler", StandardScaler()),

])



categorical_pipeline =  Pipeline([

    ("select_numeric", DataFrameSelector(categoricals.columns.tolist())),

#   ("std_scaler", CategoricalEncoder(encoding="onehot-dense")),

    ("std_scaler", OneHotEncoder()),

])







main_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', numerical_pipeline),

    ('cat_pipeline', categorical_pipeline)

])



# Scale our features from our training data

scaled_xtrain = main_pipeline.fit_transform(X_train)







from sklearn.linear_model import LinearRegression, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score,GridSearchCV
def display_scores(scores):

    print('Scores :', scores)

    print('Mean :', scores.mean())

    print('Std Deviation :', scores.std())
#linear regression model

linear_reg=LinearRegression()

linear_reg.fit(scaled_xtrain,y_train )

y_train_predict_lin=linear_reg.predict(scaled_xtrain)

print('Linear Regression RMSE: %.2f'% np.sqrt(mean_squared_error(y_train,y_train_predict_lin)))

print('Linear Regression R-squared: %.2f'% r2_score(y_train,y_train_predict_lin))

#CV for linear_regression

lin_scores=cross_val_score(linear_reg,scaled_xtrain,y_train,scoring='neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
#Decision Tree regression model

tree_reg=DecisionTreeRegressor()

tree_reg.fit(scaled_xtrain,y_train )

y_train_predict_tree=tree_reg.predict(scaled_xtrain)

print('Decision Tree Regression RMSE: %.2f'% np.sqrt(mean_squared_error(y_train,y_train_predict_tree)))

print('Decision Tree Regression R-squared: %.2f'% r2_score(y_train,y_train_predict_tree))

#CV for DecisionTree Regression

tree_scores=cross_val_score(tree_reg,scaled_xtrain,y_train,scoring='neg_mean_squared_error',cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)

display_scores(tree_rmse_scores)
#RandomForest regression model

forest_reg=RandomForestRegressor()

forest_reg.fit(scaled_xtrain,y_train )

y_train_predict_forest=forest_reg.predict(scaled_xtrain)

print('Decision Tree Regression RMSE: %.2f'% np.sqrt(mean_squared_error(y_train,y_train_predict_forest)))

print('Decision Tree Regression R-squared: %.2f'% r2_score(y_train,y_train_predict_forest))

#CV for RandomForest Regression

forest_scores=cross_val_score(forest_reg,scaled_xtrain,y_train,scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
param_grid=[{'n_estimators':[3,10,30,60,100],'max_features':[2,4,6]},

             {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]

forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg,param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(scaled_xtrain,y_train)

grid_search.best_params_
cvres=grid_search.cv_results_

for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):

    print(np.sqrt(-mean_score),params)
feature_importances=grid_search.best_estimator_.feature_importances_

attributes=['House_Age', 'Distance_Mrt','Stores_nearby','X5 latitude','X6 longitude','DaysElapsed']

sorted(zip(feature_importances,attributes), reverse=True)
final_model=grid_search.best_estimator_

scaled_xtest = main_pipeline.fit_transform(X_test)

final_pred=final_model.predict(scaled_xtest)

print('Final Model RMSE on test set: %.2f'% np.sqrt(mean_squared_error(y_train,y_train_predict_tree)))

print('Final Model R-squared on test set : %.2f'% r2_score(y_train,y_train_predict_tree))
