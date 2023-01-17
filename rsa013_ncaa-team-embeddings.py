# Define and register a kaggle renderer for Altair

# Source for this: https://www.kaggle.com/jakevdp/altair-kaggle-renderer



import altair as alt

import json

from IPython.display import HTML



KAGGLE_HTML_TEMPLATE = """

<style>

.vega-actions a {{

    margin-right: 12px;

    color: #757575;

    font-weight: normal;

    font-size: 13px;

}}

.error {{

    color: red;

}}

</style>

<div id="{output_div}"></div>

<script>

requirejs.config({{

    "paths": {{

        "vega": "{base_url}/vega@{vega_version}?noext",

        "vega-lib": "{base_url}/vega-lib?noext",

        "vega-lite": "{base_url}/vega-lite@{vegalite_version}?noext",

        "vega-embed": "{base_url}/vega-embed@{vegaembed_version}?noext",

    }}

}});

function showError(el, error){{

    el.innerHTML = ('<div class="error">'

                    + '<p>JavaScript Error: ' + error.message + '</p>'

                    + "<p>This usually means there's a typo in your chart specification. "

                    + "See the javascript console for the full traceback.</p>"

                    + '</div>');

    throw error;

}}

require(["vega-embed"], function(vegaEmbed) {{

    const spec = {spec};

    const embed_opt = {embed_opt};

    const el = document.getElementById('{output_div}');

    vegaEmbed("#{output_div}", spec, embed_opt)

      .catch(error => showError(el, error));

}});

</script>

"""



class KaggleHtml(object):

    def __init__(self, base_url='https://cdn.jsdelivr.net/npm'):

        self.chart_count = 0

        self.base_url = base_url

        

    @property

    def output_div(self):

        return "vega-chart-{}".format(self.chart_count)

        

    def __call__(self, spec, embed_options=None, json_kwds=None):

        # we need to increment the div, because all charts live in the same document

        self.chart_count += 1

        embed_options = embed_options or {}

        json_kwds = json_kwds or {}

        html = KAGGLE_HTML_TEMPLATE.format(

            spec=json.dumps(spec, **json_kwds),

            embed_opt=json.dumps(embed_options),

            output_div=self.output_div,

            base_url=self.base_url,

            vega_version=alt.VEGA_VERSION,

            vegalite_version=alt.VEGALITE_VERSION,

            vegaembed_version=alt.VEGAEMBED_VERSION

        )

        return {"text/html": html}

    

alt.renderers.register('kaggle', KaggleHtml())

print("Define and register the kaggle renderer. Enable with\n\n"

      "    alt.renderers.enable('kaggle')")
#collapse_hide

from pathlib import Path

import numpy as np

import pandas as pd

np.random.seed(13)

import tensorflow as tf

import keras as k

from keras.models import Model

from keras.layers import Dense, Input, Dropout, Activation, Multiply, Lambda, Concatenate, Subtract, Flatten

from keras.layers.embeddings import Embedding

from keras.initializers import glorot_uniform, glorot_normal

from keras.optimizers import Adam

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.manifold.t_sne import TSNE

import altair as alt



alt.renderers.enable('kaggle')

np.random.seed(13)
#collapse_hide

dataLoc=Path('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage2/')



df_teams = pd.read_csv(dataLoc/'MTeams.csv')

teams_dict = df_teams[['TeamID','TeamName']].set_index('TeamID').to_dict()['TeamName']



df_regSeason_data = pd.read_csv(dataLoc/'MRegularSeasonCompactResults.csv')

df_regSeason_data.head() # cols = Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT
#collapse_hide

df_otherTourney_data = pd.read_csv(dataLoc/'MSecondaryTourneyCompactResults.csv').drop(columns='SecondaryTourney')

df_otherTourney_data.head() # cols = Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT
#collapse_hide

# Create team encoding that differentiates teams by year and school

def newTeamID(df):

    # df = df.sample(frac=1).reset_index(drop=True)

    df['Wnewid'] = df['Season'].astype(str) + df['WTeamID'].astype(str)

    df['Lnewid'] = df['Season'].astype(str) + df['LTeamID'].astype(str)

    return df



df_regSeason_data = newTeamID(df_regSeason_data)

df_otherTourney_data = newTeamID(df_otherTourney_data)



def idDicts(df):

    newid_W = list(df['Wnewid'].unique())

    newid_L = list(df['Lnewid'].unique())

    ids = list(set().union(newid_W,newid_L))

    ids.sort()

    oh_to_id = {}

    id_to_oh = {}

    for i in range(len(ids)):

        id_to_oh[ids[i]] = i 

        oh_to_id[i] = ids[i]



    return oh_to_id, id_to_oh



oh_to_id, id_to_oh = idDicts(df_regSeason_data)    



# add training data in swapped format so network sees both wins and losses

def swapConcat_data(df):



    df['Wnewid'] = df['Wnewid'].apply(lambda x: id_to_oh[x])

    df['Lnewid'] = df['Lnewid'].apply(lambda x: id_to_oh[x])



    loc_dict = {'A':-1,'N':0,'H':1}

    df['WLoc'] = df['WLoc'].apply(lambda x: loc_dict[x])



    swap_cols = ['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 'Lnewid', 'Wnewid']



    df_swap = df[swap_cols].copy()



    df_swap['WLoc'] = df_swap['WLoc']*-1



    df.columns = [x.replace('WLoc','T1_Court')

                   .replace('W','T1_')

                   .replace('L','T2_') for x in list(df.columns)]



    df_swap.columns = df.columns



    df = pd.concat([df,df_swap])



    df['Win'] = (df['T1_Score']>df['T2_Score']).astype(int)

    df['Close_Game']= abs(df['T1_Score']-df['T2_Score']) <3

    df['Score_diff'] = df['T1_Score'] - df['T2_Score']

    df['T2_Court'] = df['T1_Court']*-1

    df[['T1_Court','T2_Court']] = df[['T1_Court','T2_Court']] + 1



    cols = df.columns.to_list()



    df = df[cols].sort_index()

    df.reset_index(drop=True,inplace=True)





    return df



df_regSeason_full = swapConcat_data(df_regSeason_data.copy().sort_values(by='DayNum'))

df_otherTourney_full = swapConcat_data(df_otherTourney_data.copy())



# Convert to numpy arrays in correct format

def prep_inputs(df,id_to_oh, col_outputs):

    Xteams = np.stack([df['T1_newid'].values,df['T2_newid'].values]).T

    Xloc = np.stack([df['T1_Court'].values,df['T2_Court'].values]).T



    if len(col_outputs) <2:

        Y_outputs = df[col_outputs].values

        Y_outputs = Y_outputs.reshape(len(Y_outputs),1)

    else:

        Y_outputs = np.stack([df[x].values for x in col_outputs])



    return [Xteams, Xloc], Y_outputs



X_train, Y_train = prep_inputs(df_regSeason_full, id_to_oh, ['Win','Score_diff'])

X_test, Y_test = prep_inputs(df_otherTourney_full, id_to_oh, ['Win','Score_diff'])



# Normalize point outputs - Win/loss unchanged

def normalize_outputs(Y_outputs, stats_cache=None):

    if stats_cache == None:

        stats_cache = {}

        stats_cache['mean'] = np.mean(Y_outputs,axis=1)

        stats_cache['var'] = np.var(Y_outputs,axis=1)

    else: pass

    

    numOut = Y_outputs.shape[0]

    Y_normout = (Y_outputs-stats_cache['mean'].reshape((numOut,1)))/stats_cache['var'].reshape((numOut,1))



    return Y_normout, stats_cache



Y_norm_train, stats_cache_train = normalize_outputs(Y_train,None)

Y_norm_test, _ = normalize_outputs(Y_test,stats_cache_train)

Y_norm_train[0,:] = Y_train[0,:]

Y_norm_test[0,:] = Y_test[0,:]

#collapse_show

# build model



tf.keras.backend.clear_session()



def NCAA_Embeddings_Joint(nteams,teamEmb_size):

    team_input = Input(shape=[2,],dtype='int32', name='team_input')

    X_team = Embedding(input_dim=nteams, output_dim=teamEmb_size, input_length=2, embeddings_initializer=glorot_uniform(), name='team_encoding')(team_input)



    loc_input = Input(shape=[2,],dtype='int32', name='loc_input')

    X_loc = Embedding(input_dim=3, output_dim=1, input_length=2, embeddings_initializer=glorot_uniform(), name='loc_encoding')(loc_input)

    X_loc = Lambda(lambda z: k.backend.repeat_elements(z, rep=teamEmb_size, axis=-1))(X_loc)

    

    X = Multiply()([X_team,X_loc])

    X = Dropout(rate=.5)(X)

    X1 = Lambda(lambda z: z[:,0,:])(X)

    X2 = Lambda(lambda z: z[:,1,:])(X)



    D1 = Dense(units = 20, use_bias=True, activation='tanh')

    DO1 = Dropout(rate=.5)



    D2 = Dense(units = 10, use_bias=True, activation='tanh')

    DO2 = Dropout(rate=.5)



    X1 = D1(X1)

    X1 = DO1(X1)



    X1 = D2(X1)

    X1 = DO2(X1)



    X2 = D1(X2)

    X2 = DO1(X2)



    X2 = D2(X2)

    X2 = DO2(X2)



    X_sub = Subtract()([X1,X2])



    output_w= Dense(units = 1, use_bias=False, activation='sigmoid', name='win_output')(X_sub)

    output_p= Dense(units = 1, use_bias=False, activation=None, name='point_output')(X_sub)





    model = Model(inputs=[team_input, loc_input],outputs=[output_w,output_p],name='ncaa_embeddings_joint')



    return model



mymodel = NCAA_Embeddings_Joint(len(id_to_oh),15)

mymodel.summary()
#collapse_show

# Joint model

optimizer = Adam(learning_rate=.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

mymodel.compile(loss=['binary_crossentropy','logcosh'],

                loss_weights=[.5,400],

                optimizer=optimizer,

                metrics = ['accuracy'])

numBatch = round(X_train[0].shape[0]/50)

results = mymodel.fit(X_train, [*Y_norm_train], validation_data=(X_test, [*Y_norm_test]), epochs = 20, batch_size = numBatch,shuffle=True, verbose=True)
#collapse_hide

accuracy = results.history['win_output_accuracy']

val_accuracy = results.history['val_win_output_accuracy']

loss = results.history['win_output_loss']

val_loss = results.history['val_win_output_loss']

# summarize history for accuracy

plt.plot(accuracy)

plt.plot(val_accuracy)

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(loss)

plt.plot(val_loss)

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#collapse_hide

def transform_y(preds,stats_cache):

    preds = stats_cache['var'][1] * preds + stats_cache['mean'][1]

    return preds



preds = mymodel.predict(X_test)



tmp=0



x = transform_y(preds[1],stats_cache_train).reshape(-1)

y = transform_y(Y_norm_test[1],stats_cache_train).reshape(-1)





print('Pearson coefficient: ', round(stats.pearsonr(x, y)[0]*100)/100)

plt.scatter(x, y, alpha=0.08)

# plt.title('Scatter plot pythonspot.com')

plt.xlabel('Predicted point difference')

plt.ylabel('Actual point difference')

plt.show()
#collapse_hide

x = preds[0].reshape(-1)



plt.hist(x,bins=100)

# plt.title('Scatter plot pythonspot.com')

plt.xlabel('Predicted Win Probability')

plt.ylabel('Count')

plt.show()
#collapse_hide

embeddings = mymodel.layers[3].get_weights()[0]



t = TSNE(n_components=2)

embed_tsne = t.fit_transform(embeddings)



df_regSeason_full['T1_TeamName'] = df_regSeason_full['T1_TeamID'].apply(lambda x: teams_dict[x]) + '-' + df_regSeason_full['Season'].astype(str)

df_agg=df_regSeason_full.groupby('T1_TeamName').mean()

df_agg.reset_index(inplace=True,drop=False)

df_agg['Score_diff'] = -df_agg['Score_diff'] 

df_agg['Win'] = -df_agg['Win']

df_agg[['T1_TeamName','Win','Score_diff']]

df_agg.drop(columns='Season',inplace=True)



df_tourney_data = pd.read_csv(dataLoc/'MNCAATourneyCompactResults.csv')

df_tourney_data['WTeamName'] = df_tourney_data['WTeamID'].apply(lambda x: teams_dict[x]) + '-' + df_tourney_data['Season'].astype(str)

df_tourney_data['Wins'] = 0

df_wins = df_tourney_data[['WTeamName','Wins']].groupby('WTeamName').count()

tourneyWinners = [df_tourney_data.loc[df_tourney_data['Season']==s,'WTeamName'].values[-1] for s in df_tourney_data['Season'].unique()]



df_seeds = pd.read_csv(dataLoc/'MNCAATourneySeeds.csv')

df_seeds['TeamName'] = df_seeds['TeamID'].apply(lambda x: teams_dict[x]) + '-' + df_seeds['Season'].astype(str)

df_seeds['Seed'] = df_seeds['Seed'].str.extract(r'(\d+)')

df_seeds['WonTourney'] = df_seeds['TeamName'].apply(lambda x: True if x in tourneyWinners else False)

df_seeds = df_seeds[['TeamName','Seed','WonTourney']]



df_upsets = pd.read_csv('../input/ncaa-biggest-opening-weekend-upsets-from-cbs/Upsets.csv') # link to article: https://www.cbssports.com/college-basketball/news/march-madness-2019-the-10-biggest-upsets-ever-in-the-opening-weekend-of-the-tournament/

df_upsets['David']=df_upsets['David']+'-'+df_upsets['Season'].astype(str)

df_upsets['Goliath']=df_upsets['Goliath']+'-'+df_upsets['Season'].astype(str)

upsets = {}

for ii in df_upsets['David'].unique():

    upsets[ii] = 'Surprise'

for ii in df_upsets['Goliath'].unique():

    upsets[ii] = 'Bust'

df_seeds = pd.merge(left=df_seeds, right=df_wins, how='left', left_on='TeamName',right_index=True)

df_seeds['Wins'].fillna(0,inplace=True)



def upset(x):

    try:

        y = upsets[x]

    except:

        y = None

    return y

df_seeds['Upset'] = df_seeds['TeamName'].apply(lambda x: upset(x))



df = pd.DataFrame(embed_tsne,columns=['factor1','factor2'])

df['TeamName'] = [str(teams_dict[int(oh_to_id[x][-4:])]) + '-' + oh_to_id[x][:4] for x in df.index]

df['Season'] = [int(oh_to_id[x][:4])for x in df.index]



df = pd.merge(left=df, right=df_seeds, how='left', on='TeamName')

df = pd.merge(left=df, right=df_agg, how='left', left_on='TeamName',right_on='T1_TeamName')



df = df[['TeamName','Season','factor1','factor2','Win','Score_diff','Seed','Wins','Upset','WonTourney']]

df.columns = ['TeamName','Season','factor1','factor2','RegWins','RegPoint_diff','Seed','TourneyWins','Upset','WonTourney']



df2020 = df[df['Season']==2020].copy()



df.dropna(inplace=True,subset=['Seed'])



df['TourneyWinsScaled'] = df['TourneyWins']/df['TourneyWins'].max()

df['SeedScaled'] = df['Seed'].astype(int)/df['Seed'].astype(int).max()



df.head()
#collapse_hide

selector = alt.selection_single(empty='all', fields=['TeamName'])



base = alt.Chart(df).mark_point(filled=True,size=50).encode(

    color=alt.condition(selector,

                        alt.Color('WonTourney:N', scale=alt.Scale(scheme='tableau10')),

                        alt.value('lightgray') ),

    order=alt.Order('WonTourney:N', sort='ascending'),

    tooltip=['TeamName','Seed']

).properties(

    width=250,

    height=250

).add_selection(selector).interactive()



base.encode(alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-75,50])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-32,11])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1.05,-.3])) )
#collapse_hide

selector = alt.selection_single(empty='all', fields=['TeamName'])



base = alt.Chart(df).mark_point(filled=True,size=35).encode(

    color=alt.condition(selector,

                        alt.Color('Seed:Q', scale=alt.Scale(scheme='viridis',reverse=True)),

                        alt.value('lightgray') ),

    order=alt.Order('Seed:Q', sort='descending'),

    tooltip=['TeamName','Seed']

).properties(

    width=250,

    height=250

).add_selection(selector).interactive()



base.encode( alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-75,50])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-32,11])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1.05,-.3])) )
#collapse_hide

selector = alt.selection_single(empty='all', fields=['TeamName'])



base = alt.Chart(df).mark_point(filled=True,size=35).encode(

    color=alt.condition(selector,

                        alt.Color('TourneyWins:Q', scale=alt.Scale(scheme='viridis',reverse=False)),

                        alt.value('lightgray') ),

    order=alt.Order('TourneyWins:Q', sort='ascending'),

    tooltip=['TeamName','Seed']

).properties(

    width=250,

    height=250

).add_selection(selector).interactive()



base.encode( alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-75,50])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-32,11])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1.05,-.3])) )
#collapse_hide

selector = alt.selection_single(empty='all', fields=['TeamName'])



base = alt.Chart(df).mark_point(filled=True,size=50).encode(

    color=alt.condition(selector,

                        alt.Color('Upset:N', scale=alt.Scale(scheme='tableau10')),

                        alt.value('lightgray') ),

    order=alt.Order('Upset:N', sort='ascending'),

    tooltip=['TeamName','Seed']

).properties(

    width=250,

    height=250

).add_selection(selector).interactive()



base.encode( alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-75,50])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-32,11])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1.05,-.3])) )
#collapse_hide

select_year = alt.selection_single(

    name='select', fields=['Season'], init={'Season': 1985},

    bind=alt.binding_range(min=1985, max=2019, step=1))



selector = alt.selection_single(empty='all', fields=['TeamName'])



##

base = alt.Chart(df).mark_point(filled=True,size=50).encode(

    color=alt.condition(selector,

                        alt.Color('TourneyWins:Q', scale=alt.Scale(scheme='viridis',reverse=False)),

                        alt.value('lightgray') ),

    order=alt.Order('Seed:Q', sort='descending'),

    tooltip=['TeamName','Seed']

).properties(

    width=250,

    height=250

).add_selection(select_year).transform_filter(select_year).add_selection(selector).interactive()



base.encode( alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-75,50])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-32,11])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1.05,-.3])) )
#collapse_hide

## 2020 plot

selector = alt.selection_single(empty='all', fields=['TeamName'])



base = alt.Chart(df2020).mark_point(filled=True,size=50).encode(

    color=alt.condition(selector,

                        alt.Color('TeamName:N'),

                        alt.value('lightgray') ),

    order=alt.Order('RegWins:Q', sort='ascending'),

    tooltip=['TeamName']

).properties(

    width=250,

    height=250

).add_selection(selector).interactive()



base.encode( alt.X('factor1:Q', scale=alt.Scale(domain=[-70,60])), alt.Y('factor2:Q', scale=alt.Scale(domain=[-65,75])) )  | base.encode( alt.X('RegPoint_diff:Q', scale=alt.Scale(domain=[-21,26])),alt.Y('RegWins:Q',scale=alt.Scale(domain=[-1,0])) )