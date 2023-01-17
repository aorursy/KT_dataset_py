import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import sklearn as sk

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder



import eli5

from eli5.sklearn import PermutationImportance



init_notebook_mode(connected=True)

pd.options.display.float_format = '{:0,.4f}'.format
df = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',engine='python',parse_dates=['deadline','launched'])

df['diff'] = df['deadline'] - df['launched']



# convert the time delta to numeric days

def tointerval(r):

    return pd.Timedelta(r).days



df['duration'] = df['diff'].apply(tointerval)

sample = df.sample(n=9366)

df.head(3)
success_main_category_df = (df.query('state == "successful"')

                              .groupby(['main_category'])

                              .agg('count')

                              .reset_index()

                           )



main_category_count = (df.groupby(['main_category'])

                         .agg('count')

                         .reset_index()

                      )



fail_main_category_df = (df.query('state == "failed"')

                           .groupby(['main_category'])

                           .agg('count')

                           .reset_index()

                        )

# TODO add the ratio of success of different categories

# pd.concat([main_category_count,main_category_count])



fig1 = make_subplots(

    rows=1, cols=3,

    specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

fig1.add_trace(go.Pie(labels=main_category_count['main_category'], values=main_category_count['ID'],title='Total',name='Total'),row=1, col=1)

fig1.add_trace(go.Pie(labels=success_main_category_df['main_category'], values=success_main_category_df['ID'],title='Success',name='Success'),row=1, col=2)

fig1.add_trace(go.Pie(labels=fail_main_category_df['main_category'], values=fail_main_category_df['ID'],title='Failed',name='Failed'),row=1, col=3)

fig1.update_layout(title='Counts of differenct mian categories',width=1000,height=480)

iplot(fig1)
main_category = df.groupby(['main_category']).agg(np.median)

main_category = main_category.reset_index().sort_values(by='goal',ascending=True)

fig2 = make_subplots(

    rows=1, cols=3,

    subplot_titles=("goal", "backers", "pledged"))

fig2.add_trace(go.Bar(y=main_category['main_category'], x=main_category['goal'],marker={'color':'RoyalBlue'},orientation='h',name='median goal'),

              row=1, col=1)

fig2.add_trace(go.Bar(y=main_category['main_category'], x=main_category['backers'],marker={'color':'LightSeaGreen'},orientation='h',name='median backers'),

              row=1, col=2)

fig2.add_trace(go.Bar(y=main_category['main_category'], x=main_category['usd pledged'],marker={'color':'crimson'},orientation='h',name='median usd pledged'),

              row=1, col=3)

fig2.update_layout(showlegend=False,height=500,title='Median Goal, Backers and usd pledged of different categories')

iplot(fig2)
fig3 = px.box(sample,x='state',y='goal',log_y=True,height=400,width=800,title='Box plot for different state and goal')

iplot(fig3)
fig4 = px.box(sample,x='state',y='backers',log_y=True,height=400,width=800,title='Amount of backers with different project state')

iplot(fig4)
map_df = df.groupby(['state','currency']).agg('count').reset_index()

fig5 = go.Figure(go.Heatmap(

          x = map_df['state'],

          y = map_df['currency'],

          z = map_df['ID'],

          type = 'heatmap',

          colorscale = ['gold','mediumturquoise']))

fig5.update_layout(width=800,height=400,title='Currency counts in different state')

iplot(fig5)
fig6 = px.box(sample,x='state',y='usd pledged',color='currency',log_y=True,height=400,title='')

iplot(fig6)
map_df = df.groupby(['state','country']).agg('count').reset_index()

# px.density_heatmap(sample,x='state',y='currency',z='usd pledged')

fig7 = go.Figure(go.Heatmap(

          x = map_df['state'],

          y = map_df['country'],

          z = map_df['ID'],

          type = 'heatmap',

          colorscale = ['gold','mediumturquoise']))

fig7.update_layout(width=800,height=400,title='Counts of each state in different country')

fig7.show()
fig8 = px.box(sample,x='state',y='duration',title='The relationship between state and duration of projects')

iplot(fig8)
# convert the time delta to numeric days

def tointerval(r):

    return pd.Timedelta(r).days



def gen_duration(df):

    df['diff'] = df['deadline'] - df['launched']

    df['duration'] = df['diff'].apply(tointerval)

    return df.drop(['diff','deadline','launched'],axis=1)



# convert the state to numeric

def gen_result_value(r):

    if r == 'successful':

        return 1

    else:

        return 0



# make new column based on state condition

def gen_result_column(df): 

    df['result'] = df['state'].apply(gen_result_value)

    return df.drop('state',axis=1)



# convert categories to numeric

def encode_category(df):

    encoder = LabelEncoder()

    for column in ['main_category','category']:

        df[column] = encoder.fit_transform(df[column])

    return df



# main category median goal

def gen_median_category_goal(df,index_column):

    pivot_median = (df[[index_column,'goal']].groupby(index_column)

                                             .agg(np.median)

                                             .reset_index()

                                             .rename(columns={'goal':f'median_{index_column}_goal'})

                   )

    return df.merge(pivot_median,on=index_column)



def concat_one_hot_encoder(df):

    one_hot = pd.get_dummies(df['main_category'])

    return pd.concat([df,one_hot],axis=1)



# in order to simplify the problem, we rule out 

# the state of undefined, suspended,canceled and live. 

model_val = (df[['deadline','goal','launched','state','backers','usd pledged','main_category','category']]

            .dropna()

            .query('(state == "failed") or (state == "successful")')

            .pipe(gen_duration)

            .pipe(gen_result_column)

            .pipe(concat_one_hot_encoder)

            .pipe(encode_category)

            .pipe(gen_median_category_goal,'main_category')

            .pipe(gen_median_category_goal,'category')

)

model_val.head()
from sklearn.model_selection import train_test_split

# shuffle the sample

model_val = sk.utils.shuffle(model_val)



# assign X exculde the result, nparray

X = model_val.drop("result",axis=1)

# assign y as column 'result'

y = model_val["result"]



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

model_features = [i for i in model_val.columns if i != 'result']
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
# TODO compelete the envolution, cross validation, roc,aoc etc

model.score(X_test,y_test)
model_features_importance = pd.DataFrame(model.feature_importances_,index=model_features,columns=['importance']).reset_index().sort_values(by='importance',ascending=True).rename(columns={'index':'feature'})

fig9 = px.bar(model_features_importance,x='importance',y='feature',orientation='h',width=600,log_x=True,title='Relative importance of features')

iplot(fig9)
PI = PermutationImportance(model,random_state=1).fit(X_train,y_train)

PI_df = eli5.explain_weights_df(PI,feature_names=model_features).sort_values(by='weight',ascending=True)

fig10 = px.bar(PI_df,x='weight',y='feature',orientation='h',width=800,log_x=True,title='Permutation importance of each feature')

iplot(fig10)
conbimed_importance = PI_df.merge(model_features_importance,on='feature',how='inner')

fig11 = px.scatter(conbimed_importance,x='importance',y='weight',hover_data=['feature','weight','std'],log_x=True,log_y=True)

iplot(fig11)
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot



pdp_goals = pdp.pdp_isolate(model=model, dataset=X_train, model_features=model_features, feature='backers')



# plot it

pdp.pdp_plot(pdp_goals, 'backers',x_quantile=True)

plt.show()
# Create the data that we will plot

pdp_goal = pdp.pdp_isolate(model=model, dataset=X_train, model_features=model_features, feature='goal')



# plot it

pdp.pdp_plot(pdp_goal, 'goal',x_quantile=True)

plt.show()
# Create the data that we will plot

pdp_usd_pledged = pdp.pdp_isolate(model=model, dataset=X_train, model_features=model_features, feature='usd pledged')



# plot it

pdp.pdp_plot(pdp_usd_pledged, 'usd pledged',x_quantile=True)

plt.show()
# Create the data that we will plot

pdp_usd_pledged = pdp.pdp_isolate(model=model, dataset=X_train, model_features=model_features, feature='duration')



# plot it

pdp.pdp_plot(pdp_usd_pledged, 'duration')

plt.show()
# Create the data that we will plot

pdp_median_category_goal = pdp.pdp_isolate(model=model, dataset=X_train, model_features=model_features, feature='median_category_goal')



# plot it

pdp.pdp_plot(pdp_median_category_goal, 'median_category_goal')

plt.show()