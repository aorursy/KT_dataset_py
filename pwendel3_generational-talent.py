

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush





from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV,GroupKFold

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer



from sklearn.pipeline import Pipeline,make_pipeline



from scipy.special import softmax



import matplotlib.patches as patches



import keras

from keras.callbacks import Callback, EarlyStopping

import tensorflow as tf

import keras.backend as K



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]

pd.set_option('mode.chained_assignment', None)
env = nflrush.make_env()
train_raw = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
def timesplit(x):

    x=x.split(':')

    return(60*int(x[0])+int(x[1]))
def heightsplit(x):

    x=x.split('-')

    return(12*int(x[0])+int(x[1]))
piv_cols=['PlayerHeight','PlayerWeight','PlayerAge','X','Y',

    'Orientation_x','Orientation_y','Dir_x','Dir_y','S','A','Dis',

              'is_off','is_rusher','is_rb','is_l','is_sk']
# cat_cols=['Week','Quarter','Down','OffenseFormation', 'DefendersInTheBox',

#  'outside','rain','snow','Turf','is_home']

cat_cols=['Week','Quarter','Down','OffenseFormation', 'DefendersInTheBox', 

          'outside','rain','snow','Turf','is_home']#

          #,'PossTeam','DefTeam']
cont_cols=['till_reg','YardsFromOwnGoal','PossScore','DefScore',

           'Distance','handoff_delay']
tall_pipe=make_column_transformer(

                                (make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()),['PlayerHeight','PlayerWeight','PlayerAge','X','Y',

                                                                                                    'Orientation_x','Orientation_y','Dir_x','Dir_y','S','A','Dis']),

                                ('passthrough',['is_off','is_rusher','is_rb','is_l','is_sk'])

                                 )

    

    





cont_pipe=make_column_transformer(

#         (make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse=False)),cat_cols),





        (make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),cont_cols)

)



cat_pipe=make_column_transformer(

#         (make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse=False)),cat_cols),





        (make_pipeline(SimpleImputer(strategy='most_frequent'),OrdinalEncoder()),cat_cols)

)

         


def feature_magic(train_in,tall_piper=tall_pipe,cat_piper=cat_pipe,cont_piper=cont_pipe,piv_cols=piv_cols,cat_cols=cat_cols,lefty=False,training=False):

    

    train=train_in.copy()

    if lefty:

        train_left=train.copy()

        train_left['PlayId']=-train_left['PlayId']

        train=pd.concat([train,train_left])

        train=train.reset_index(drop=True)



    train.loc[train.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"

    train.loc[train.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"



    train.loc[train.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr']= "BLT"

    train.loc[train.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"



    train.loc[train.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"

    train.loc[train.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"



    train.loc[train.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"

    train.loc[train.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"

    

    

    

    train['is_rusher']=train.NflId==train.NflIdRusher

    train['is_home']=train.Team=='home'

    train['is_off']=(train.is_home & (train['HomeTeamAbbr']==train['PossessionTeam'])) | (~train.is_home & (train['HomeTeamAbbr']!=train['PossessionTeam']))



    train['YardsFromOwnGoal']=train['YardLine']

    train.loc[train.FieldPosition!=train.PossessionTeam,'YardsFromOwnGoal']=50+50-train.loc[train.FieldPosition!=train.PossessionTeam,'YardsFromOwnGoal']





    train['to_left']=train.PlayDirection=='left'

    train.loc[train.to_left,'X']=120-train.loc[train.to_left,'X']

    

    train['X']=train['X']-train['YardsFromOwnGoal']



    train.loc[train.to_left,'Y']=(160/3)-train.loc[train.to_left,'Y']

    

    train['Y']=train.Y-(160/3)/2



    train.loc[train.PlayId<0,'Y']=-train.loc[train.PlayId<0,'Y']



    train.loc[train.to_left,'Orientation']=train.loc[train.to_left,'Orientation']-180

    train.loc[train.to_left,'Dir']=train.loc[train.to_left,'Dir']-180



    train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)



    deg_cols=['Orientation','Dir']

    for deg_col in deg_cols:

        train[deg_col+'_x']=np.sin(np.radians(train[deg_col]))

        train[deg_col+'_y']=np.cos(np.radians(train[deg_col]))

        train.loc[train.PlayId<0,deg_col+'_y']=-train.loc[train.PlayId<0,deg_col+'_y']

        

    



    train['TimeSnap']=pd.to_datetime(train.TimeSnap)

    train['PlayerBirthDate']=pd.to_datetime(train['PlayerBirthDate'],utc=True)    



    train['PlayerAge']=train['TimeSnap']-train['PlayerBirthDate']

    train['PlayerAge']=train['PlayerAge'].apply(lambda x:x.total_seconds()/(365.25*24*3600))





    train.PlayerHeight=train.PlayerHeight.apply(heightsplit)

    

    posmap={'CB':'DB','WR':'WR','G':'OL','T':'OL','DE':'DL','DT':'DL','OLB':'LB','TE':'TE','FS':'DB','C':'OL','RB':'RB','QB':'QB',

    'SS':'DB','ILB':'LB','MLB':'LB','NT':'DL','LB':'LB','OT':'OL','FB':'FB','OG':'OL','DB':'DB','S':'DB','HB':'RB','SAF':'DB','DL':'DL'}



    train.Position=train.Position.apply(lambda x: posmap[x])

       

    train['is_rb']=(train.Position=='RB')|(train.Position=='FB')

    train['is_l']=(train.Position=='OL')|(train.Position=='DL')|(train.Position=='TE')|(train.Position=='FB')

    train['is_sk']=(train.Position=='WR')|(train.Position=='DB')|(train.Position=='TE')

    

    train=train.sort_values(['PlayId','Y','X','is_off'])

    

    

    

    trains=train.copy()

    train_run=train[train.is_rusher]

    

    if training:

        train[piv_cols]=tall_piper.fit_transform(train[piv_cols])

    else:

        train[piv_cols]=tall_piper.transform(train[piv_cols])    

                            

    convdat=train.groupby('PlayId',as_index=False).apply(lambda x:x[piv_cols].values)

    

    convdat=np.dstack(convdat,)

    convdat=np.swapaxes(convdat,0,2)

    #print(convdat.shape)

    convdat=convdat.reshape(convdat.shape[0],convdat.shape[1],convdat.shape[2],1)



    

    train_run.GameClock=train.GameClock.apply(timesplit)

    train_run['till_reg']=train_run.GameClock+(4-train_run.Quarter)*60*15

       

    

    train_run['outside']=(train_run.StadiumType.str.lower().str.contains(r'ou[a-z]+')|

    train_run.StadiumType.str.lower().str.contains(r'open')|

    train_run.StadiumType.str.lower().str.contains(r'heinz field')|

                      train_run.StadiumType.str.lower().str.contains(r'bowl')

                     )



    train_run['Turf']=~(train_run.Turf.str.lower()=='grass')|(train_run.Turf.str.lower().str.contains(r'natur'))



    train_run['rain']=(train_run.GameWeather.str.lower().str.contains(r'rain')|

    train_run.GameWeather.str.lower().str.contains(r'shower'))



    train_run['snow']=(train_run.GameWeather.str.lower().str.contains(r'snow'))

    

    train_run['PossScore']=np.select([train_run.is_home,~train_run.is_home],[train_run.HomeScoreBeforePlay,train_run.VisitorScoreBeforePlay])

    train_run['DefScore']=np.select([train_run.is_home,~train_run.is_home],[train_run.VisitorScoreBeforePlay,train_run.HomeScoreBeforePlay])



    train_run['PossTeam']=np.select([train_run.is_home,~train_run.is_home],[train_run.HomeTeamAbbr,train_run.VisitorTeamAbbr])

    train_run['DefTeam']=np.select([train_run.is_home,~train_run.is_home],[train_run.VisitorTeamAbbr,train_run.HomeTeamAbbr])



    

    td=pd.to_datetime(train_run.TimeHandoff)-pd.to_datetime(train_run.TimeSnap)

    train_run['handoff_delay']=td.dt.total_seconds()

    

    

    if training:

        yards=train_run.Yards

        train_run=train_run.drop(columns='Yards')

        train_cont=cont_piper.fit_transform(train_run)

        

        train_cat=cat_piper.fit_transform(train_run)

        train_cat=train_cat.reshape(train_cat.shape+(1,))

        

        train_run['Yards']=yards

    else:

        train_cont=cont_piper.transform(train_run)

        

        train_cat=cat_piper.transform(train_run)

        train_cat=train_cat.reshape(train_cat.shape+(1,))

        train_cat=train_cat.swapaxes(0,1)

        train_cat=train_cat.tolist()



    train_run.reset_index(drop=True,inplace=True)

    

    return(trains,train_run,convdat,train_cont,train_cat,tall_piper,cont_piper,cat_piper)
trainn,train_runn,train_conv,train_cont,train_cat,tall_pipe,cont_pipe,cat_pipe=feature_magic(train_raw,lefty=True,training=True)
train_cont[0]
train_cat.shape
train_raw.head()
trainn.head()
train_runn.head()
sns.heatmap(train_conv[55].reshape(train_conv[0].shape[0],train_conv[0].shape[1]),yticklabels=piv_cols)
sns.heatmap(train_conv[1].reshape(train_conv[0].shape[0],train_conv[0].shape[1]),yticklabels=piv_cols)
test_mask=((train_runn.Season==2018)&((train_runn.Week>=13)))&(train_runn.PlayId>0)

train_mask=~((train_runn.Season==2018)&((train_runn.Week>=13)))
cv_cat=train_cat[train_mask]

cv_cat=cv_cat.swapaxes(0,1)

cv_cat=cv_cat.tolist()





test_cat=train_cat[test_mask]

test_cat=test_cat.swapaxes(0,1)

test_cat=test_cat.tolist()



train_catt=train_cat.swapaxes(0,1)

train_catt=train_catt.tolist()
cv_cont=train_cont[train_mask]

test_cont=train_cont[test_mask]
train_cont[test_mask].shape
cv_conv=train_conv[train_mask]



test_conv=train_conv[test_mask]
Y_c = np.zeros((train_runn.shape[0], 199))

for idx, target in enumerate(train_runn['Yards']):

    Y_c[idx][99 + target] = 1

    

cv_Y_c=Y_c[train_mask]

test_Y_c=Y_c[test_mask]
from keras.layers import Dense

from keras.models import Sequential,Model

from keras.callbacks import Callback, EarlyStopping

from keras.layers import Dropout, PReLU, BatchNormalization, concatenate,ELU,AveragePooling2D, GaussianNoise, Activation,Dense, Conv2D, Flatten,MaxPooling2D,Embedding,Input,Reshape,Concatenate

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy

def crps_loss(y_true, y_pred):

    return K.mean(K.square(K.clip(K.cumsum(y_true, axis=1),0,1) - K.clip(K.cumsum(y_pred, axis=1),0,1)), axis=1)
def ordinal_loss(y_true, y_pred):

    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')

    return (1.0 + weights) * categorical_crossentropy(y_true, y_pred)
def ordinal_loss_2(y_true, y_pred):

    weights = K.cast(K.square(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')

    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)
def ordinal_loss_h(y_true, y_pred,delta=10):

    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')



    mae=(1.0 + weights) * categorical_crossentropy(y_true, y_pred)

    quadratic=K.minimum(mae, delta)

    linear=mae - quadratic

    

    return 0.5 * K.square(quadratic) + delta * linear
cat_col_map=dict(train_runn[cat_cols].nunique())
def cont_model(cat_cols=cat_cols,cont_cols=cont_cols):

    

    inputs=[]

    embeddings=[]

    

    for i in cat_col_map.keys():

        input=Input(shape=(1,))

        vals=cat_col_map[i]

        embs=int(np.sqrt(vals))

        embedding=Embedding(vals,embs,input_length=1)(input)

        embedding=Reshape(target_shape=(embs,))(embedding)

        inputs.append(input)

        embeddings.append(embedding)

    

    cont_input=Input(shape=(len(cont_cols),))

    inputs.append(cont_input)

    embeddings.append(cont_input)

    cont_x=Concatenate()(embeddings)

    

    #cont_x=cont_input

    

    cont_x=Dense(32,input_dim=inputs, activation=None)(cont_x)

    cont_x=BatchNormalization()(cont_x)

    cont_x=PReLU()(cont_x)

    cont_x=Dropout(0.7)(cont_x)

    cont_x=GaussianNoise(0.2)(cont_x)

    

    cont_x=Dense(32, activation=None)(cont_x)

    cont_x=BatchNormalization()(cont_x)

    cont_x=PReLU()(cont_x)

    cont_x=Dropout(0.5)(cont_x)

    cont_x=GaussianNoise(0.2)(cont_x)

    

    

    cont_model=Model(inputs,cont_x)

    

    return cont_model


cont_model_=cont_model()

#cont_model_.summary()
def conv_model2(cols):

    conv_input=Input(shape=(cols,22,1))

    

    conv_x=conv_input



    conv_3=Conv2D(64, kernel_size=(cols,3), activation='relu',

                     input_shape=(cols,22,1),padding='same')(conv_x)

#    conv_x=MaxPooling2D(pool_size=(1,3))(conv_x)

    

    

    conv_5=Conv2D(64, kernel_size=(cols,5), activation='relu',

                     input_shape=(cols,22,1),padding='same')(conv_x)

#    conv_x=Conv2D(32, kernel_size=(1,1), activation='relu')(conv_x)

#    conv_x=MaxPooling2D(pool_size=(1,3))(conv_x)

    

#     conv_x=Conv2D(32, kernel_size=(1,1), activation='tanh')(conv_x)

#     conv_x=MaxPooling2D(pool_size=(1,3))(conv_x)



    conv_7=Conv2D(64, kernel_size=(cols,7), activation='relu',

                     input_shape=(cols,22,1),padding='same')(conv_x)



    

    conv_out=concatenate([conv_3,conv_5,conv_7],axis=3)

    

    conv_out=Flatten()(conv_out)

    conv_out=BatchNormalization()(conv_out)

    conv_out=Dropout(0.7)(conv_out)

    conv_out=GaussianNoise(0.2)(conv_out)

    

    conv_out=Dense(32, activation=None)(conv_out)

    conv_out=BatchNormalization()(conv_out)

    conv_out=PReLU()(conv_out)

    

    

    conv_model=Model(conv_input,conv_out)



    return conv_model

def conv_model(cols,player_width):

    conv_input=Input(shape=(cols,22,1))

    

    conv_x=conv_input



    conv_x=Conv2D(64, kernel_size=(cols,player_width), activation='relu',

                  input_shape=(cols,22,1),padding='same')(conv_x)

    conv_x=BatchNormalization(axis=1)(conv_x)

    

    conv_x=AveragePooling2D(pool_size=(1,player_width),strides=1)(conv_x)

    

    conv_x=Conv2D(32, kernel_size=(1,player_width), activation='relu',

                      padding='same')(conv_x)

    conv_x=BatchNormalization(axis=1)(conv_x)

    

    conv_x=AveragePooling2D(pool_size=(1,player_width),strides=1)(conv_x)

    

    conv_x=Conv2D(32, kernel_size=(1,player_width), activation='relu',

                      padding='same')(conv_x)

    conv_x=BatchNormalization(axis=1)(conv_x)

    

    conv_x=AveragePooling2D(pool_size=(1,player_width),strides=1)(conv_x)

    



    

    conv_x=Flatten()(conv_x)

    conv_x=BatchNormalization()(conv_x)

    conv_x=Dropout(0.5)(conv_x)

    conv_x=GaussianNoise(0.2)(conv_x)

    



    conv_x=Dense(32, activation=None)(conv_x)

    conv_x=BatchNormalization()(conv_x)

    conv_x=PReLU()(conv_x)

    

    conv_model=Model(conv_input,conv_x)



    return conv_model

cont_inputs=train_cont.shape[1]

cont_inputs
conv_inputs=train_conv.shape[1]

#conv_model_=conv_model(conv_inputs)

conv_model_=conv_model(conv_inputs,3)

conv_model_.summary()


# cont_model_=cont_model(cont_inputs)



models=concatenate([cont_model_.output,conv_model_.output])

models=BatchNormalization()(models)

models=Dropout(0.7)(models)

models=GaussianNoise(0.2)(models)



combout=Dense(32,activation=None)(models)

combout=BatchNormalization()(combout)

combout=PReLU()(combout)

combout=Dropout(0.7)(combout)

combout=GaussianNoise(0.2)(combout)



# combout=Dense(32,activation=None)(models)

# combout=BatchNormalization()(combout)

# combout=PReLU()(combout)

# combout=Dropout(0.5)(combout)

# combout=GaussianNoise(0.2)(combout)



combout=Dense(32,activation=None)(models)

combout=BatchNormalization()(combout)

combout=PReLU()(combout)

combout=Dropout(0.5)(combout)

combout=GaussianNoise(0.2)(combout)



combout=Dense(199, activation='softmax')(combout)
#conv_model_.summary()


big_model=Model(inputs=cont_model_.input+[conv_model_.input],outputs=combout)
big_model.summary()
np.random.seed(214)

#from tensorflow.random import set_seed

#set_seed(214)
#losss=ordinal_loss_h(delta=10)

#opt=R

big_model.compile(optimizer=Adam(learning_rate=0.005,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-07,

    amsgrad=False), loss=ordinal_loss,

                 metrics=[crps_loss]

                 )





es=EarlyStopping(patience=10, min_delta=5e-5, restore_best_weights=True, monitor='val_crps_loss')



big_model.fit(cv_cat+[cv_cont,cv_conv], cv_Y_c, callbacks=[es], epochs=100, batch_size=512, 

              verbose=True,validation_data=(test_cat+[test_cont,test_conv],test_Y_c) )





# big_model.compile(optimizer=Adam(learning_rate=0.001,

#     beta_1=0.9,

#     beta_2=0.999,

#     epsilon=1e-07,

#     amsgrad=False), loss=crps_loss,metrics=[ordinal_loss])



# es=EarlyStopping(patience=10, min_delta=5e-5, restore_best_weights=True, monitor='val_loss')



# big_model.fit([cv_cont,cv_conv], cv_Y_c, callbacks=[es], epochs=100, batch_size=256, 

#               verbose=True,validation_data=([test_cont,test_conv],test_Y_c) )

train_preds=big_model.predict(train_catt+[train_cont,train_conv])

#test_preds=big_model.predict([test_cont,test_conv])
#predslon=np.sum(test_preds*np.arange(-99,100),axis=1)

train_predslon=np.sum(train_preds*np.arange(-99,100),axis=1)

#test_predslon=np.sum(train_preds*np.arange(-99,100),axis=1)
sns.distplot(train_predslon,kde=False)
train_runn['pred']=train_predslon

def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12, 6.33)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='lightgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

            plt.text(x, 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

        ax.plot([x, x], [22.91, 23.57], color='white')

        ax.plot([x, x], [29.73, 30.39], color='white')



    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

                 color='yellow')

    return fig, ax
def show_play(play_id,x='X',y='Y',mag='S',ang='Dir',trained=False):

    

    play=trainn[trainn.PlayId==play_id]

    

    yards=play.Yards.values[0]

    yardline=play.YardsFromOwnGoal.values[0]+10

    yardgain=yards+yardline

    play['X']=play['X']+play['YardsFromOwnGoal']

    if trained:

        play[piv_cols]=sp.inverse_transform(play[piv_cols])

    

    

    off=play[play.is_off]

    deff=play[~play.is_off]

    rusher=play[play.is_rusher]



    fig, ax = create_football_field()

    plt.scatter(x=off[x],y=off[y]+53.3/2,c='red',s=30,label='Offense')

    plt.scatter(x=deff[x],y=deff[y]+53.3/2,c='blue',s=30,label='Defense')

    plt.scatter(x=rusher[x],y=rusher[y]+53.3/2,color='black',s=15,label='BallCarrier')

    

    plt.plot([yardline,yardline],[0,53.3],

             color='grey')

    

    plt.plot([yardgain,yardgain],[0,53.3],

             color='gold')



    for i, row in off.iterrows():

        ax.arrow(row[x], row[y]+53.3/2, row[mag]*row[ang+'_x'], row[mag]*row[ang+'_y'], head_width=0.5, head_length=0.7, ec='orange')





    for i, row in deff.iterrows():

        ax.arrow(row[x], row[y]+53.3/2, row[mag]*row[ang+'_x'], row[mag]*row[ang+'_y'], head_width=0.5, head_length=0.7, ec='purple')



   

    plt.legend(loc=4)

    plt.show()
lossess=np.mean(np.square(np.clip(np.cumsum(train_preds, axis=1),0,1) - np.clip(np.cumsum(Y_c, axis=1),0,1)), axis=1)

lossess.mean()
train_runn['loss']=lossess
train_runn.loss.quantile([.1,.25,.5,.75,.8,.9,.95,1])
sns.lmplot('Yards','loss',data=train_runn)
from sklearn.metrics import r2_score
r2_score(train_runn['Yards'],train_runn['pred'])
big_preds=train_runn.sort_values('pred',ascending=False)[['PlayId','PossTeam','PossScore','DefTeam','DefScore','DisplayName','Quarter','GameClock','Down','Distance','Yards','pred','loss']].head(10)

big_preds
for i,row in big_preds.iterrows():

    print(row)

    show_play(row['PlayId'])

    fig, ax = plt.subplots(1, figsize=(12, 6.33))

    plt.bar(np.arange(-99,100),train_preds[i])

    plt.show()
small_preds=train_runn.sort_values('pred',ascending=True)[['PlayId','PossTeam','PossScore','DefTeam','DefScore','DisplayName','Quarter','GameClock','Down','Distance','Yards','pred','loss']].head(10)

small_preds
for i,row in small_preds.iterrows():

    print(row)

    show_play(row['PlayId'])

    fig, ax = plt.subplots(1, figsize=(12, 6.33))

    plt.bar(np.arange(-99,100),train_preds[i])

    plt.show()
iter_test=env.iter_test()
def clipper(yl,predsin):



    lind=-yl+99

    #print(lind)

    rind=100-yl+99

    #print(rind)

    #print(len(test[lind:rind]))

    #preds[0,lind:rind]=preds[0,lind:rind]/np.sum(preds[0,lind:rind])

    

    #preds[0,lind:rind]=softmax(preds[0,lind:rind])

    #predsin[:lind]=0

    predsin=np.clip(np.cumsum(predsin),0,1)

    predsin[:lind]=0

    predsin[rind:]=1

    return predsin


for (test_df, sample_prediction_df) in iter_test:

    

    #_,test_runn,test_conv,test_cont,tall_pipe,wide_pipe=feature_magic(test_df,lefty=False,training=False)

    testt,test_runn,test_conv,test_cont,test_cat,tall_pipe,cont_pipe,cat_pipe=feature_magic(test_df,lefty=False,training=False)

    

    

    test_pred=big_model.predict(test_cat+[test_cont,test_conv])

    

    #pred=np.clip(np.cumsum(clipper(test_run['YardsFromOwnGoal'].values[0],pred)),0,1)

    test_pred=clipper(test_runn.YardsFromOwnGoal.values[0],test_pred)

    pred_df=pd.DataFrame(data=[test_pred],columns=sample_prediction_df.columns)



    env.predict(pred_df)
env.write_submission_file()
# dummy_pred=np.zeros(199)

# pred_df=pd.DataFrame(data=[dummy_pred],columns=sample_prediction_df.columns)



# env.predict(pred_df)
