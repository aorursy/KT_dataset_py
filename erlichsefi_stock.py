# This is a kernel for a job interview.  
# geting two sigma env
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, GRU
from keras.losses import binary_crossentropy, mse
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import explained_variance_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#getting parameters 
explore_data=True;
savetime=True;
(market_train_df, news_train_df) = env.get_training_data()
if (explore_data):
    # the data set is to large, last look on a subset of it of exploring
    asserts_names=market_train_df.assetName.value_counts().sort_values(ascending=False).head(20).index.values
    market_train_df=market_train_df[market_train_df.assetName.isin(asserts_names)]
    news_train_df=news_train_df[news_train_df.assetName.isin(asserts_names)]
# let first look on the stock data
market_train_df.head()
# let's drop samples that tag as Unknown in assetName
# acording to the description thus assets as no samples in the news set.
market_train_df.loc[market_train_df.assetName=='Unknown']['assetCode'].describe()
# we will lose alot of samples.
# let's drop it to free some memory.
market_train_df.drop(columns=["assetName"],inplace=True)
print("Data size="+str(market_train_df.shape))
market_train_df.isnull().sum()
# for now lets drop the row containg NaN, it's seems that they are a small precetege of the samples.
# we can later try to optimize the estimator by replaceing any NAN value with mean values
# if we see that we need more samples to converge.

clean_market_train_df = market_train_df.loc[market_train_df.returnsClosePrevMktres1.notnull()]
print("After #1 filter= "+str(clean_market_train_df.shape))
clean_market_train_df = clean_market_train_df.loc[market_train_df.returnsOpenPrevMktres1.notnull()]
print("After #2 filter= "+str(clean_market_train_df.shape))
clean_market_train_df = clean_market_train_df.loc[market_train_df.returnsClosePrevMktres10.notnull()]
print("After #3 filter= "+str(clean_market_train_df.shape))

clean_market_train_df = clean_market_train_df.loc[market_train_df.returnsOpenPrevMktres10.notnull()]
print("After #4 filter= "+str(clean_market_train_df.shape))
clean_market_train_df.describe()
# it's seems the mean prediction is small positive (0.166107)
# - it's intersting to look on the example where the volume is 0, what that means for the prices? 
# - it's seems that overall the market want up between open to close
# - how come returnsOpenNextMktres10 reached 9761 ?!
# how do we explain this?
# is that means that one compny gained
clean_market_train_df[clean_market_train_df.returnsOpenNextMktres10>1000]
clean_market_train_df=clean_market_train_df[clean_market_train_df.returnsOpenNextMktres10<1000]
# some how the value of the stock drop even when volume is zero.
clean_market_train_df[(clean_market_train_df.volume==0) & (clean_market_train_df.open!=clean_market_train_df.close) ]
# we can drop the instance, it's can only be explain by a reverse split (but the news data wouldn't help as to see data)
# but we will fix all thus cases before sending to the classifer (change 0 to 2)
clean_market_train_df[~((clean_market_train_df.volume==0) & (clean_market_train_df.open!=clean_market_train_df.close))].volume=2;
# it's seems that the distribution of  returnsOpenNextMktres10 and returnsOpenPrevMktres10 are very close:
# very close STD and Mean, let's check corr()
clean_market_train_df.corr()
#so there isn't good corracltion between the two.

# Notice: There are good corracltion here but all of them are for the 'pure' return to MKtres returns
clean_market_train_df['month']=clean_market_train_df.time.dt.month
clean_market_train_df.groupby("month").returnsOpenNextMktres10.agg(['count','mean'])
# it's seems may and April is a good month of share holders :0.
clean_market_train_df['open_close_diff']=clean_market_train_df.close-clean_market_train_df.open
clean_market_train_df[['open_close_diff','returnsOpenNextMktres10','month']].plot.scatter(x='open_close_diff',y='returnsOpenNextMktres10',c='month')
# it's seems may is a good month of share holders :0.
def slide(to_sum):
    for i in range(9): 
        to_sum[i%len(to_sum)]=to_sum.mean()
    return to_sum.mean()
#clean_market_train_df['10day_moving_avg']=clean_market_train_df.apply(lambda x: clean_market_train_df[(clean_market_train_df.assetCode==x.assetCode) & (clean_market_train_df.time<x.time)].sort_values("time").head(10).returnsOpenNextMktres10,axis=1)
news_train_df.head()
news_train_df.assetCodes=news_train_df.assetCodes.apply(lambda x:str(list(eval(x))));
news_train_df['id'] = range(1, len(news_train_df) + 1)
#https://mikulskibartosz.name/how-to-split-a-list-inside-a-dataframe-cell-into-rows-in-pandas-9849d8ff2401
ingredients = []
cuisines = []
ids = []
for _, row in news_train_df.iterrows():
    assetCode = row.assetCodes
    identifier = row.id
    for ingredient in eval(row.assetCodes):
        cuisines.append(assetCode)
        ingredients.append(ingredient)
        ids.append(identifier)
        
ingredient_to_cuisine = pd.DataFrame({
    "id": ids,
    "single_assetCode": ingredients,
    "original_assetCode": cuisines
})
ingredient_to_cuisine.head()
ready_to_merge=news_train_df.merge(ingredient_to_cuisine, on='id').drop(columns=["id","assetCodes","original_assetCode"])
ready_to_merge.head(2)
# fix dates to join
clean_market_train_df['onlydate']=clean_market_train_df.time.dt.date
ready_to_merge['onlyDate']=ready_to_merge.firstCreated.dt.date

# join
new_df = pd.merge(ready_to_merge, clean_market_train_df,  how='inner', left_on=["single_assetCode","onlyDate"], right_on = ["assetCode","onlydate"])
new_df.shape
new_df.head()
# first features 
new_df['secondsToCreated']=(new_df.sourceTimestamp-new_df.firstCreated).dt.seconds;
new_df['secondsToFeed']=(new_df.time_x-new_df.firstCreated).dt.seconds;
new_df.groupby("sourceId").returnsOpenNextMktres10.mean().describe();
# we will not copy it to the  final
new_df.groupby("urgency").returnsOpenNextMktres10.mean()
# it's seems the urgency affect the price
new_df['urgency_reduce']=new_df.urgency.map({3:1,1:0})
new_df['CapitalLetters']=new_df.headline.apply(lambda x: len(x)-sum(map(str.islower, x)))
new_df['headlineLenth']=new_df.headline.apply(lambda x: len(x))
new_df[['CapitalLetters','headlineLenth','returnsOpenNextMktres10']].corr()
new_df.groupby("provider").provider.count().plot(kind="pie")
# we see that the provider of the sample if mostly RTRS, if so i would consider droping of now all other providers to decrease variance.
# we will leave if for optimitions, later. 

label = LabelEncoder()
label.fit(new_df.provider.drop_duplicates())
new_df['enc_provider'] = label.transform(new_df.provider) 
# source : https://www.kaggle.com/sergeykalutsky/lstm-model-on-market-data
def one_hot_encode(df, columns):
    categorical_df = pd.get_dummies(df[columns].astype(str))
    df.drop(columns=columns, inplace=True)
    df = pd.concat([df, categorical_df], axis=1)  
    # delete categorical
    del categorical_df
    
    return df
if not(savetime):
# if would like to check if 
#
    all_subjects=set()
    for s in new_df.subjects:
        all_subjects.update(eval(s))
    print("Done collecting "+str(len(all_subjects))+" subjects")


    ma={};
    i=0;
    for s in all_subjects:
        i=i+1
        mean=new_df[new_df.subjects.str.contains(s)].returnsOpenNextMktres10.mean()
        if (i%100==0):
            print(".",end='')
        ma[s]=mean
    

if not(savetime):
    plt.bar(ma.keys(), list(ma.values()), align='center')
    plt.xticks(range(len(ma)), list(ma.keys()))
    plt.show()
if not(savetime):
    subjects_features=[]
    for name, mean in ma.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if np.abs(mean) > 0.1:
            subjects_features.append(name)
        
    # i will had feature if one of thus subjects are in subject list
    print(subjects_features) # ['NM1', 'TJ', 'USASD', 'LPG1', 'ENDOCR']
    

# if would like to check if 
#
if not(savetime):
    all_audiences=set()
    for s in new_df.audiences:
        all_audiences.update(eval(s))
    print("Done collecting "+str(len(all_audiences))+" subjects")




    ma2={};
    i=0;
    for s in all_audiences:
        i=i+1
        mean=new_df[new_df.audiences.str.contains(s)].returnsOpenNextMktres10.mean()
        if (i%100==0):
            print(".",end='')
        ma2[s]=mean
    

if not(savetime):
    plt.bar(ma2.keys(), list(ma2.values()), align='center')
    plt.xticks(range(len(ma2)), list(ma2.keys()))
    plt.show()
if not(savetime):

    audiences_features=[]
    for name, mean in ma2.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if np.abs(mean) > 0.1:
            audiences_features.append(name)
        
    # i will had feature if one of thus audiences  are in subject list
    print(audiences_features) # ['FMW', 'FSC', 'PRL', 'FMO', 'FMA']
audiences_features=['FMW', 'FSC', 'PRL', 'FMO', 'FMA']
new_df.audiences=new_df.audiences.apply(lambda x: x if x in audiences_features else "Other")

subjects_features=['NM1', 'TJ', 'USASD', 'LPG1', 'ENDOCR']
new_df.subjects=new_df.subjects.apply(lambda x: x if x in subjects_features else "Other")

new_df = one_hot_encode(new_df, ['subjects','audiences'])
new_df.drop(columns=["headlineTag"],inplace=True)
new_df[['returnsOpenNextMktres10','marketCommentary']].groupby("marketCommentary").mean()

new_df.marketCommentary=new_df.marketCommentary.map({False:0,True:1})
# droping not used no numric values
clean_df=new_df.drop(columns=["onlydate","assetCode","onlyDate","time_y","single_assetCode","assetName","provider","headline","sourceId","firstCreated","sourceTimestamp"])
clean_df.head(2)
ready=clean_df.sort_values("time_x")
# let's check all numeric
ready.info()
X=ready.drop(columns=["returnsOpenNextMktres10","time_x","universe"]).values
universe = ready.universe.values
day = ready.time_x.dt.values
Y=ready["returnsOpenNextMktres10"].values


def split_sequence(sequence,label,universe,day, n_steps):
    X, y ,u, d = list(),list(),list(),list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y= label[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        u.append(universe[end_ix])
        d.append(day[end_ix])
    return np.array(X), np.array(y),np.array(u),np.array(d)
X_train, X_test, y_train, y_test, u_train, u_test, d_train, d_test = train_test_split(X, Y,universe,day,test_size=0.25)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_train)

#y_scaler = MinMaxScaler(feature_range=(0, 1))
#y_scaler = scaler.fit(y_train.reshape(-1, 1))


# normalize the features
y_train_normalized = y_train;#y_scaler.transform(y_train.reshape(-1, 1))
y_test_normalized = y_test;#y_scaler.transform(y_test.reshape(-1, 1))

# normalize the dataset and print
X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)
interval=4; # try others by cv

seq_n_X_train, seq_n_y_train,seq_n_u_train, seq_n_d_train=split_sequence(X_train_normalized,y_train_normalized,u_train,d_train,interval)
seq_n_X_test, seq_n_y_test,seq_n_u_test, seq_n_d_test=split_sequence(X_test_normalized,y_test_normalized,u_test,d_test,interval)

#https://www.kaggle.com/sergeykalutsky/lstm-model-on-market-data

model = Sequential()
model.add(LSTM(15, input_shape=(interval,X.shape[1]), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss='mean_squared_error', metrics=['categorical_accuracy'])

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=8,verbose=True)

model.fit(seq_n_X_train, seq_n_y_train,
        epochs= 10,
        batch_size = 500,
        validation_data = (seq_n_X_test, seq_n_y_test),
        callbacks=[early_stop,check_point])

model.load_weights('model.hdf5')
confidence_valid = model.predict(seq_n_X_test)
print(mean_squared_error(confidence_valid, seq_n_y_test))
most_data_asset_name=new_df.single_assetCode.value_counts().index[0];
change=new_df[new_df.single_assetCode==most_data_asset_name].sort_values("time_x").returnsOpenNextMktres10.values
temp_x=new_df[new_df.single_assetCode==most_data_asset_name].sort_values("time_x").drop(columns=["onlydate","assetCode","onlyDate","time_y","single_assetCode","assetName","provider","time_x","universe","headline","sourceId","firstCreated","sourceTimestamp","returnsOpenNextMktres10"])

sx,_,_,_=split_sequence(scaler.transform(temp_x.values),np.random.randint(5, size=temp_x.shape[0]),np.random.randint(5, size= temp_x.shape[0]),np.random.randint(5, size=temp_x.shape[0]),interval)
plt.figure(figsize=(25,25))
plt.plot(change)
plt.title(str(most_data_asset_name)+" True change")
plt.show()
 
plt.figure(figsize=(25,25))
plt.plot(model.predict(sx))
plt.title(str(most_data_asset_name)+" Pred change")
plt.show()
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'max_depth': [8,14,20,25],
#     'n_estimators': [100, 200, 300]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)
# grid_search.fit(rX_train, ry_train)
