import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Reduce the usage of memory

# Ref: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    '''

    iterate through all the columns of a dataframe and modify the data type

    to reduce memory usage.        

    '''

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
develop_mode = False

if develop_mode:

    df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv', nrows=5000))

    df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv'))

else:

    df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv'))

    df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv'))
print('The sizes of the datasets are:')

print('Training Dataset: ', df_train.shape)

print('Testing Dataset: ', df_test.shape)
# Get Sample Data

df_train.head(10)
def visualize(col_name, num_bin=10):

    '''

    Function for visualization

    '''

    title_name = col_name[0].upper() + col_name[1:]

    f, ax = plt.subplots()

    plt.xlabel(title_name)

    plt.ylabel('log Count')

    ax.set_yscale('log')

    df_train.hist(column=col_name,ax=ax,bins=num_bin)

    plt.title('Histogram of ' + title_name)

    tmp = df_train[col_name].value_counts().sort_values(ascending=False)



    print('Min value of ' + title_name + ' is: ',min(tmp.index))

    print('Max value of ' + title_name + ' is: ',max(tmp.index))
group_tmp = df_train[df_train['matchId']=='df014fbee741c6']['groupId'].value_counts().sort_values(ascending=False)
plt.figure()

plt.bar(group_tmp.index,group_tmp.values)

plt.xlabel('GroupId')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.title('Number of Group Members in One Match')

plt.show()



print('Min number of group members is: ',min(group_tmp.values))

print('Max number of group members is: ',max(group_tmp.values))
visualize('assists')
visualize('roadKills')
visualize('killStreaks')
visualize('teamKills')
visualize('longestKill',num_bin=100)
visualize('weaponsAcquired',num_bin=30)
visualize('headshotKills',num_bin=30)
visualize('DBNOs',num_bin=50)
visualize('boosts',num_bin=30)
visualize('heals',num_bin=80)
visualize('damageDealt',num_bin=1000)
visualize('revives',num_bin=40)
visualize('walkDistance',num_bin=260)
visualize('rideDistance',num_bin=400)
visualize('swimDistance',num_bin=100)
visualize('vehicleDestroys',num_bin=5)
def MissValueAnalysis():

    miss_total = df_train.isnull().sum().sort_values(ascending=False)

    miss_percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([miss_total, miss_percent], axis=1, keys=['total', 'percent'])



    percent_data = miss_percent.head(20)

    percent_data.plot(kind="bar")

    plt.xlabel("Columns")

    plt.ylabel("Percentage")

    plt.title("Total Missing Value (%) in Training Data")

    plt.show()



    miss_total = df_test.isnull().sum().sort_values(ascending=False)

    miss_percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([miss_total, miss_percent], axis=1, keys=['total', 'percent'])



    percent_data = miss_percent.head(20)

    percent_data.plot(kind="bar")

    plt.xlabel("Columns")

    plt.ylabel("Percentage")

    plt.title("Total Missing Value (%) in Training Data")

    plt.show()

    

MissValueAnalysis()
def CorrelationAnalysis():

    corr = df_train.corr()

    f, ax = plt.subplots(figsize=(15, 15))

    heatmap = sns.heatmap(corr,cbar=True, annot=True, 

                          square=True, fmt='.2f', 

                          cmap='YlGnBu')

    

CorrelationAnalysis()
df_train.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize=(8,6), title='walkDistance vs winPlacePerc')
df_train.plot(x="heals",y="winPlacePerc", kind="scatter", figsize=(8,6), title='heals vs winPlacePerc')
def HealsVSwinPlacePerc():

    f, ax = plt.subplots(figsize=(8, 6))

    fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)

    plt.title('heals vs winPlacePerc box plot')

    fig.axis(ymin=0, ymax=1)

    

HealsVSwinPlacePerc()
df_train.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6), title='longestKill vs winPlacePerc')
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import gc, sys



def BuildFeature(is_train=True):

    '''

    Function for feature engineering

    is_train incicates whether the train set or the test set is processed

    '''

    y = None

    test_idx = None

    

    if is_train: 

        print("Reading train.csv")

        df = pd.read_csv('../input/train_V2.csv')           

        df = df[df['maxPlace'] > 1]

    else:

        print("Reading test.csv")

        df = pd.read_csv('../input/test_V2.csv')

        test_idx = df.Id

    

    # Reduce the memory usage

    df = reduce_mem_usage(df)

    

    print("Delete Unuseful Columns")

    target = 'winPlacePerc'

    features = list(df.columns)

    features.remove("Id")

    features.remove("matchId")

    features.remove("groupId")

    features.remove("matchType")  

    

    if is_train: 

        print("Read Labels")

        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)

        features.remove(target)



    print("Read Group mean features")

    agg = df.groupby(['matchId','groupId'])[features].agg('mean')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train:

        df_out = agg.reset_index()[['matchId','groupId']]

    else:

        df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])



    print("Read Group max features")

    agg = df.groupby(['matchId','groupId'])[features].agg('max')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    

    print("Read Group min features")

    agg = df.groupby(['matchId','groupId'])[features].agg('min')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    

    print("Read Group size features")

    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')

    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    

    print("Read Match mean features")

    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    

    print("Read Match size features")

    agg = df.groupby(['matchId']).size().reset_index(name='match_size')

    df_out = df_out.merge(agg, how='left', on=['matchId'])

    

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out

    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank

    gc.collect()



    return X, y, feature_names, test_idx
X_train, y_train, train_columns, _ = BuildFeature(is_train=True)

X_test, _, _ , test_idx = BuildFeature(is_train=False)
X_train =reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression(n_jobs=4, normalize=True)

LR_model.fit(X_train,y_train)
LR_model.score(X_train,y_train)
y_pred_train = LR_model.predict(X_train)

y_pred_test = LR_model.predict(X_test)
y_pred_train[y_pred_train>1] = 1

y_pred_train[y_pred_train<0] = 0



f, ax = plt.subplots(figsize=(10,10))

plt.scatter(y_train, y_pred_train)

plt.xlabel("y")

plt.ylabel("y_pred_train")

plt.xlim([0,1])

plt.ylim([0,1])

plt.show()
y_pred_test[y_pred_test>1] = 1

y_pred_test[y_pred_test<0] = 0
df_test['winPlacePerc'] = y_pred_test

submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)