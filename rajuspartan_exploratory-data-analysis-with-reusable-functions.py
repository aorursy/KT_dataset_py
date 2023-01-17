# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pyo.init_notebook_mode()

import warnings

from numpy import percentile

warnings.filterwarnings("ignore")

#train=pd.read_csv('C://Users//nagaraju//Documents//kaggle//Boxoffice//tmdb-box-office-prediction//train.csv')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

subfile1=pd.read_csv('../input/sample_submission.csv')

subfile2=pd.read_csv('../input/sample_submission.csv')
##### Flatening Json columns

def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d

train.belongs_to_collection = train.belongs_to_collection.map(lambda x: len(get_dictionary(x))).clip(0,1)

train.genres = train.genres.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

train.production_companies = train.production_companies.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

train.production_countries = train.production_countries.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

train.spoken_languages = train.spoken_languages.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

train.Keywords = train.Keywords.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
#Deleting few columns for time being to improve ease of analysis

train.drop(['id','belongs_to_collection','homepage','poster_path','status','cast','crew'],axis=1,inplace=True)

####Number of genres each movie has

train['numofgenres']=train['genres'].map(lambda x: len(x.split(",")))

####Number of langauges spoken

train['numofspokenlanguages']=train['spoken_languages'].map(lambda x: len(x.split(",")))

####Number of production countries for that movie

train['numofproductioncountries']=train['production_countries'].map(lambda x:len(x.split(",")))

#####Number of production companies for that movie

train['numofproductioncompanies']=train['production_companies'].map(lambda x:len(x.split(",")))

#####Indicates whether a movie to "Comedy " genre or not

train['Comedy'] = np.where( train['genres'].str.contains('Comedy') , 1,0)

#####Indicates whether a movie to "Action " genre or not

train['Action'] = np.where( train['genres'].str.contains('Action') , 1,0)

train['Horror'] = np.where( train['genres'].str.contains('Horror') , 1,0)

train['Thriller'] = np.where( train['genres'].str.contains('Thriller') , 1,0)

train['Animation'] = np.where( train['genres'].str.contains('Animation') , 1,0)



#####Indicates whether a movie produced in "US " country or not

train['Unitedstates'] = np.where( train['production_countries'].str.contains('United States') , 1,0)

train['UK'] = np.where( train['production_countries'].str.contains('United Kingdom') , 1,0)

train['Russia'] = np.where( train['production_countries'].str.contains('Russia') , 1,0)

train['Japan'] = np.where( train['production_countries'].str.contains('Japan') , 1,0)

train['India'] = np.where( train['production_countries'].str.contains('India') , 1,0)
###Imputing budget variable. Few columns have ZERO budget value which is not possible

train['budget']= train['budget'].replace(0,train['budget'].mean())

###Release date 

def date_features(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_quarter'] = df['release_date'].dt.quarter

    df['release_dow'] = df['release_date'].dt.dayofweek

    df.drop(columns=['release_date'], inplace=True)

    return df



train = date_features(train)

train['release_year']=np.where(train['release_year'] > 2017, train['release_year'] - 100 , train['release_year'])

def scatterplot(data,cont1,cont2):

    data = [go.Scatter(

        x = data[cont1],

        y = data[cont2],

        mode = 'markers',

        marker = dict(      # change the marker style

            size = 10,

            color = 'rgb(51,204,153)',

            symbol = 'pentagon',

            line = dict(

                width = 2,

            )

        )

        )]

    layout = go.Layout(

        title = 'Scatterplot ' + '' + cont1 + '     vs    ' + cont2, # Graph title

        xaxis = dict(title = cont1), # x-axis label

        yaxis = dict(title = cont2), # y-axis label

        hovermode ='closest' # handles multiple points landing on the same vertical

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



scatterplot(train,'budget','revenue')

scatterplot(train,'popularity','revenue')

scatterplot(train,'runtime','revenue')

def Linecharts(data,cont1,cont2,cat1):

    data=data.groupby(cat1).mean().reset_index()

    trace0 = go.Scatter(

        x = data[cat1],

        y = data[cont1],

        mode = 'lines',

        name = cont1

    )

    trace1=go.Scatter(

        x=data[cat1],

        y=data[cont2],

        mode='lines',

        name=cont2

    )

    data = [trace0,trace1]  # assign traces to data

    layout = go.Layout(

        title = 'Line charts',

        xaxis = dict(title = cat1), # x-axis label

        yaxis = dict(title = cont2),

        hovermode='closest'# y-axis label

    )

    fig = go.Figure(data=data,layout=layout)

    pyo.iplot(fig)



Linecharts(train,'budget','revenue','release_year')

Linecharts(train,'budget','revenue','release_month')

Linecharts(train,'budget','revenue','release_dow')
def Barcharts(data,cat1,cont1):

    data=data.groupby(cat1).mean().reset_index()

    trace1 = go.Bar(

    x=data[cat1],  

    y=data[cont1],

    name = cont1,

    marker=dict(color='#FFD700') 

    )

    data = [trace1]

    layout = go.Layout(

        title='Bar charts',

        xaxis = dict(title = cat1), # x-axis label

        yaxis = dict(title = cont1), # y-axis label

        barmode='stack'

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Barcharts(train,'numofgenres','revenue')

Barcharts(train,'numofspokenlanguages','revenue')

Barcharts(train,'numofgenres','popularity')

Barcharts(train,'original_language','revenue')

Barcharts(train,'numofproductioncountries','revenue')

Barcharts(train,'numofproductioncompanies','revenue')



#Barcharts(train,'Comedy','revenue')
def Barcharts(data,cat1,cat2,cat3,cat4,cat5,cont1):

    data1=data.groupby(cat1).mean().reset_index()

    trace1 = go.Bar(

    x=data1[cat1],  

    y=data1[cont1],

    name = cat1,

    marker=dict(color='rgb(49,54,149)') 

    )

    data2=data.groupby(cat2).mean().reset_index()

    trace2=go.Bar(

    x=data2[cat2],

    y=data2[cont1],

    name=cat2,

    marker=dict(color='rgb(254,224,144)')

    )

    data3=data.groupby(cat3).mean().reset_index()

    trace3=go.Bar(

    x=data3[cat3],

    y=data3[cont1],

    name=cat3,

    marker=dict(color='rgb(171,217,233)')

    )

    data4=data.groupby(cat4).mean().reset_index()

    trace4=go.Bar(

    x=data4[cat4],

    y=data4[cont1],

    name=cat4,

    marker=dict(color='rgb(215,48,39)')

    )

    data5=data.groupby(cat5).mean().reset_index()

    trace5=go.Bar(

    x=data5[cat5],

    y=data5[cont1],

    name=cat5,

    marker=dict(color='#CD7F32')

    )

    data = [trace1,trace2,trace3,trace4,trace5]

    layout = go.Layout(

        title='Bar charts',

        xaxis = dict(title = "Noncomedy vs Comedy, NonAction Vs Action, NonHorror vs Horror, Nonthriller vs Thriller, NonAnimation vs Animation"), # x-axis label

        yaxis = dict(title = cont1)

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)

Barcharts(train,'Comedy','Action','Horror','Thriller','Animation','revenue')

Barcharts(train,'Comedy','Action','Horror','Thriller','Animation','popularity')

Barcharts(train,'Unitedstates','UK','Russia','Japan','India','popularity')

Barcharts(train,'Unitedstates','UK','Russia','Japan','India','revenue')

def Bar_Stack_charts(data,cat1,cont1,cont3):

    data=data.groupby(cat1).mean().reset_index()

    trace1 = go.Bar(

    x=data[cat1],  # NOC stands for National Olympic Committee

    y=data[cont1],

    name = cont1,

    marker=dict(color='#FFD700') # set the marker color to gold

    )

    

    trace3 = go.Bar(

    x=data[cat1],  # NOC stands for National Olympic Committee

    y=data[cont3],

    name = cont3,

    marker=dict(color='#CD7F32') # set the marker color to gold

    )

    data = [trace1,trace3]

    layout = go.Layout(

        title='Bar charts',

        xaxis = dict(title = cat1), # x-axis label

        yaxis = dict(title = cont1 + '  ' +'  ' + cont3), # y-axis label

        barmode='stack'

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Bar_Stack_charts(train,'numofgenres','budget','revenue')

Bar_Stack_charts(train,'numofspokenlanguages','budget','revenue')

Bar_Stack_charts(train,'numofproductioncompanies','budget','revenue')

Bar_Stack_charts(train,'numofproductioncountries','budget','revenue')

def Bubblecharts(data,cont1,cont2,cont3,cat1,cat2):

    data = [go.Scatter(

            x=data[cont1],

            y=data[cont2],

            text=data[cat1],  # use the new column for the hover text

            mode='markers',

            marker=dict(size=0.4*data[cont3],color=data[cat2])

            

        )]

    layout = go.Layout(

        title='Bubble chart' + '  '  + cont1 + ' vs  ' + cont2,

        xaxis = dict(title = cont1), # x-axis label

        yaxis = dict(title = cont2), # y-axis label

        hovermode='closest'

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)





Bubblecharts(train,'budget','revenue','popularity','original_title','Comedy')

Bubblecharts(train,'budget','revenue','popularity','original_title','Animation')

Bubblecharts(train,'budget','revenue','popularity','original_title','Horror')

Bubblecharts(train,'budget','revenue','popularity','original_title','UK')

Bubblecharts(train,'budget','revenue','popularity','original_title','Thriller')
def Boxplots(data,cont1):

    data = [

    go.Box(

        y=data[cont1],

        name=cont1

        )

    #go.Box(

     #   y=data[cont2],

      #  name=cont2

       # )

    ]

    layout = go.Layout(

        title = 'Box plots' + '   ' + cont1 

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)



Boxplots(train,'budget')

Boxplots(train,'revenue')
def Histogram(data,cat1,bins):

    data = [go.Histogram(

        x=data[cat1],

        nbinsx=bins

    )]

    layout = go.Layout(

    title="Distribution of values"

    )

    fig = go.Figure(data=data, layout=layout)

    pyo.iplot(fig)

Histogram(train,'genres',15)

Histogram(train,'budget',5)

Histogram(train,'revenue',5)
from plotly import tools

def Heatmaps(data,cat1,cat2,cont1,cont2):

    trace1 = go.Heatmap(

    x=data[cat1],

    y=data[cat2],

    z=data[cont1],

    text=data['original_title'],

   colorscale='Jet' # add max/min color values to make each plot consistent

    )

    trace2 = go.Heatmap(

    x=data[cat1],

    y=data[cat2],

    z=data[cont2],

    text=data['original_title'],

    colorscale='Jet' # add max/min color values to make each plot consistent

    )

    #trace3 = go.Heatmap(

    #x=data[cat1],

    #y=data[cat2],

    #z=data[cont3],

    #colorscale='Jet' # add max/min color values to make each plot consistent

    #)

    fig = tools.make_subplots(rows=1, cols=2,

    subplot_titles=('Plot1','Plot2'),

    shared_yaxes = True,  # this makes the hours appear only on the left

    )

    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 1, 2)

    #fig.append_trace(trace3, 1, 3)

    fig['layout'].update(      # access the layout directly!

    title='Distribution of values'

    )

    pyo.iplot(fig)

 



#Heatmaps(train,'genres','spoken_languages','revenue')

Heatmaps(train,'release_year','original_language','revenue','budget')

Heatmaps(train,'release_year','release_month','revenue','budget')

#Heatmaps(train,'release_year','Comedy','revenue','budget')



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

subfile1=pd.read_csv('../input/sample_submission.csv')

subfile2=pd.read_csv('../input/sample_submission.csv')

def dataprep(data):

    def get_dictionary(s):

        try:

            d = eval(s)

        except:

            d = {}

        return d

    #data.belongs_to_collection = data.belongs_to_collection.map(lambda x: len(get_dictionary(x))).clip(0,1)

    data.genres = data.genres.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.production_companies = data.production_companies.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.production_countries = data.production_countries.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.spoken_languages = data.spoken_languages.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.Keywords = data.Keywords.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.cast = data.cast.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    data.crew = data.crew.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

    ####Number of genres each movie has

    data['numofgenres']=data['genres'].map(lambda x: len(x.split(",")))

    ####Number of langauges spoken

    data['numofspokenlanguages']=data['spoken_languages'].map(lambda x: len(x.split(",")))

    ####Number of production countries for that movie

    data['numofproductioncountries']=data['production_countries'].map(lambda x:len(x.split(",")))

    #####Number of production companies for that movie

    data['numofproductioncompanies']=data['production_companies'].map(lambda x:len(x.split(",")))

    data['numofproductioncompanies']=data['production_companies'].map(lambda x:len(x.split(",")))

    data['numofkeywords']=data['Keywords'].map(lambda x:len(x.split(",")))

    data['numofcast']=data['cast'].map(lambda x:len(x.split(",")))

    data['numofcrew']=data['crew'].map(lambda x:len(x.split(",")))

    def date_features(df):

        df['release_date'] = pd.to_datetime(df['release_date'])

        df['release_year'] = df['release_date'].dt.year

        df['release_month'] = df['release_date'].dt.month

        df['release_quarter'] = df['release_date'].dt.quarter

        df['release_dow'] = df['release_date'].dt.dayofweek

        df.drop(columns=['release_date'], inplace=True)

        return df



    data = date_features(data)

    data['release_year']=np.where(data['release_year'] > 2017, data['release_year'] - 100 , data['release_year'])

    data['budget']= data['budget'].replace(0,data['budget'].mean())

    k=pd.cut(np.array(data['release_year']),10,retbins=True)

    def yearsegment(m):

        for i in range(len(k[1])):

            if(i!=10):

                if (m > k[1][i] and m < k[1][i + 1] ):

                    return (i + 1)

            else:

                return (i + 1)

    data['year_group']=data['release_year'].apply(lambda x: yearsegment(x))

       

    data['budget']=np.log(data['budget'])

    data['original_language'] = np.where( data['original_language']=='en', 1,0)

    data["overview"]=data["overview"].replace(np.nan,"empty string")

    data['Comedy'] = np.where( data['genres'].str.contains('Comedy') , 1,0)

    data['Action'] = np.where( data['genres'].str.contains('Action') , 1,0)

    data['Horror'] = np.where( data['genres'].str.contains('Horror') , 1,0)

    data['Thriller'] = np.where( data['genres'].str.contains('Thriller') , 1,0)

    data['Animation'] = np.where( data['genres'].str.contains('Animation') , 1,0)

    data['Science Fiction'] = np.where( data['genres'].str.contains('Science Fiction') , 1,0)

    data['Drama'] =  np.where( data['genres'].str.contains('Drama') , 1,0)

    data['Crime'] =  np.where( data['genres'].str.contains('Crime') , 1,0)

    data['Family'] =  np.where( data['genres'].str.contains('Family') , 1,0)

    data['Adventure'] =  np.where( data['genres'].str.contains('Adventure') , 1,0)

    data['Romance'] =  np.where( data['genres'].str.contains('Romance') , 1,0)

    data['Istagline']= np.where( data['tagline'].isnull() , 1,0)

    data['US']=np.where( data['production_countries'].str.contains('United States of America') , 1,0)

    data['Waltdisney']=np.where(data['production_companies'].str.contains('Walt Disney Pictures') , 1,0)

    data['Warnerbros']=np.where(data['production_companies'].str.contains('Warner Bros.'),1,0)

    data.drop(['id','title','imdb_id','belongs_to_collection','spoken_languages',

               'production_countries','production_companies','cast','crew','homepage',

               'poster_path','status','title','tagline','Keywords',

               'original_title','release_year'],axis=1,inplace=True)

    data=pd.get_dummies(data,columns=['release_month'])

    del data['release_month_12']

    data=pd.get_dummies(data,columns=['release_quarter'])

    del data['release_quarter_4']

    data=pd.get_dummies(data,columns=['release_dow'])

    del data['release_dow_6']

    data=pd.get_dummies(data,columns=['year_group'])

    del data['year_group_11']

    #data['runtimeseg']=[lambda x:1 if x<100 else 0 for x in data['runtime']]

    

    

dataprep(train)

dataprep(test)



train['runtimeseg']=[1 if x<100 else 0 for x in train['runtime']]

del train['runtime']

test['runtimeseg']=[1 if x<100 else 0 for x in test['runtime']]

del test['runtime']



    

#dataset=pd.concat([train[['genres','overview']].reset_index(drop=True),test[['genres','overview']]],axis=0)

##dataset["genres"]=dataset["genres"].replace(np.nan,"empty string")

##corpus = dataset['genres']

##vectorizer = TfidfVectorizer(max_features=20,analyzer='word',stop_words=set(stopwords.words('english')))

##x = vectorizer.fit_transform(corpus)

##DS=pd.SparseDataFrame(x.todense().tolist(),columns=vectorizer.get_feature_names())

##DS.reset_index(drop=True, inplace=True)

##dataset.reset_index(drop=True, inplace=True)

##dataset = pd.concat([DS, dataset], axis=1)

###dataset=pd.concat([DS.reset_index(drop=True),dataset],axis=1)

##

##dataset.drop(['10749', '12', '18', '27', '28', '35', '53', '80', '878'],axis=1,inplace=True)

#del dataset['genres']

#    

#dataset["overview"]=dataset["overview"].replace(np.nan,"empty string")

#my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

#vectorizer=TfidfVectorizer(max_features=50,analyzer='word',stop_words=my_stop_words)

#response=vectorizer.fit_transform(dataset['overview'])

#DS1=pd.SparseDataFrame(response.todense().tolist(),columns=vectorizer.get_feature_names())

#DS1.reset_index(drop=True, inplace=True)

#dataset.reset_index(drop=True, inplace=True)

#dataset = pd.concat([DS1, dataset], axis=1)

##dataset=pd.concat([DS1.reset_index(drop=True),dataset],axis=1)

#del dataset['overview']

#

#train1=dataset.head(3000)

#test1=dataset.iloc[3000:]

#test1.reset_index(drop=True, inplace=True)

#train=pd.concat([train, train1], axis=1)

#test=pd.concat([test,test1],axis=1)



#del train['id']

#del test['id']

del train['genres']

del train['overview']

del test['genres']

del test['overview']



train['revenue']=np.log(train['revenue'])

y = train['revenue'].values

df1=train

del df1['revenue']

X = df1.values



import lightgbm as lgb

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)









model = XGBRegressor()

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, verbose=True)



results = model.evals_result()

print(results)

results['validation_0']['rmse']

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



feature_importance = pd.DataFrame(model.feature_importances_, columns = ['importance'])

feature_importance['feature'] = train.columns

feature_importance.sort_values(by='importance', inplace = True, ascending = False)

feature_importance.reset_index(drop = True, inplace = True)

feature_importance



x_validation = test.values

y_test_p_xgb = pd.Series(model.predict(x_validation))

y_test_p_xgb=np.exp(y_test_p_xgb)

y_test_p_xgb=pd.DataFrame(y_test_p_xgb)

y_test_p_xgb.columns=['revenue']

subfile1['revenue']=y_test_p_xgb['revenue']

xgbsubile=subfile1



xgbsubile.to_csv('sub15.csv')



params_lgb = {'drop_rate': [0.0977], 'feature_fraction': [0.60],

              'lambda_l1': [0.0391], 'lambda_l2': [26.68],

              'learning_rate': [0.0132],

              'max_drop': [67.0], 'min_data_in_leaf': [1.0],

              'num_leaves': [32.0], 'num_trees': [700.0]}



params_lgb = {k:v[0] for k,v in params_lgb.items()}



lg = lgb.LGBMRegressor(

                        objective = 'regression',

                        metric = 'rmse',

                        early_stopping_round = 50,

                        drop_rate = params_lgb['drop_rate'],

                        feature_fraction = params_lgb['feature_fraction'],

                        lambda_l1 = params_lgb['lambda_l1'],

                        lambda_l2 = params_lgb['lambda_l2'],

                        learning_rate = params_lgb['learning_rate'],

                        max_drop = int(params_lgb['max_drop']),

                        min_data_in_leaf = int(params_lgb['min_data_in_leaf']),

                        num_leaves = int(params_lgb['num_leaves']),

                        num_trees = int(params_lgb['num_trees']))





lg.fit(X_train, y_train, eval_set=[(X_train, y_train)])



feature_importance = pd.DataFrame(lg.feature_importances_, columns = ['importance'])

feature_importance['feature'] = train.columns

feature_importance.sort_values(by='importance', inplace = True, ascending = False)

feature_importance.reset_index(drop = True, inplace = True)

feature_importance



x_validation = test.values

y_test_p = pd.Series(lg.predict(x_validation))

y_test_p=np.exp(y_test_p)

y_test_p=pd.DataFrame(y_test_p)

y_test_p.columns=['revenue']

subfile2['revenue']=y_test_p['revenue']

lgbsubfile=subfile2





lgbsubfile.to_csv('sub14-lg.csv')





df=pd.merge(xgbsubile,lgbsubfile,how='inner',on='id')

df['revenue']=((df['revenue_x'] +  df['revenue_y']) / 2 )

del df['revenue_x'], df['revenue_y']

df.to_csv('submissionfile.csv')
