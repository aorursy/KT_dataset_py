# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import math





from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings  

warnings.filterwarnings('ignore')
TARGET_DATE=["2015/12/21","2015/12/22","2015/12/23","2015/12/24","2015/12/28","2015/12/29","2015/12/30"]
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)





# Any results you write to the current directory are saved as output.
raw_price_train_l = []

for i in range(1,9,1):

    filename = "/kaggle/input/tweetpredictstock/raw_price_train/"+str(i)+"_r_price_train.csv"

    print(filename)

    raw_price_file = pd.read_csv(filename)

    price_train = pd.DataFrame(raw_price_file)

    raw_price_train_l.append(price_train)
tweet_train_output_l = []

for i in range(1,9,1):

    filename = "/kaggle/input/tweet-test-output-p1-new/sentiment_output_train_"+str(i)+".csv"

    print(filename)

    tweet_output = pd.read_csv(filename)

    tweet_train_output_l.append(tweet_output)
tweet_test_output_l = []

for i in range(1,9,1):

    filename = "/kaggle/input/tweet-test-output-p1-new/sentiment_output_test_"+str(i)+".csv"

    print(filename)

    tweet_output = pd.read_csv(filename)

    tweet_test_output_l.append(tweet_output)
def formatDate(x):

    sts = x.split("-")

    year  = sts[0]

    month = sts[1]

    day = sts[2]

    if(month[0]=='0'):

        month = month[1:]

    if(day[0]=='0'):

        day = day[1:]

    res= year+"/"+month+"/"+day

    return res

    

df_merged_l= []



for i in range(0,8,1):

    raw_price = raw_price_train_l[i]

    tweet_sentiment= tweet_train_output_l[i]

    tweet_sentiment=tweet_sentiment.rename(columns={"Unnamed: 0": "Date"}, errors="raise")

    tweet_sentiment["Date"]= tweet_sentiment["Date"].apply(lambda x: formatDate(x))

    raw_price = raw_price.sort_values(by='Date')

    raw_mereged=raw_price.join(tweet_sentiment.set_index('Date'), on='Date')

    

    #fill nan for sentiment tweet data

    raw_mereged["no_of_positive_tweet"].fillna(0,inplace=True)

    raw_mereged["no_of_negative_tweet"].fillna(0,inplace=True)

    raw_mereged["no_of_tweet"].fillna(0,inplace=True)

    raw_mereged["score"].fillna(0.5,inplace=True)

    

    raw_mereged=raw_mereged.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    raw_mereged = raw_mereged.sort_values(by='Date')

    raw_mereged=raw_mereged.reset_index()

    df_merged_l.append(raw_mereged)



    if(i<1):

        print("raw price size:"+str(raw_price.shape[0]))

        print("tweet sentiment size:"+str(tweet_sentiment.shape[0]))

        print("merged df size:"+str(raw_mereged.shape[0]))

        print("tweet df:"+str(tweet_sentiment.ix[240:410, :20]))

        print("Merged df:"+str(raw_mereged.ix[250:410, :20]))
df_merged_l[0]
df_test_merged_l=[];



for i in range(0,8,1):

    tweet_sentiment= tweet_test_output_l[i]

    tweet_sentiment=tweet_sentiment.rename(columns={"Unnamed: 0": "Date"}, errors="raise")

    tweet_sentiment["Date"]= tweet_sentiment["Date"].apply(lambda x: formatDate(x))

    

    #fill nan for sentiment tweet data

    tweet_sentiment["no_of_positive_tweet"].fillna(0,inplace=True)

    tweet_sentiment["no_of_negative_tweet"].fillna(0,inplace=True)

    tweet_sentiment["no_of_tweet"].fillna(0,inplace=True)

    tweet_sentiment["score"].fillna(0.5,inplace=True)

    

#     raw_mereged=raw_mereged.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

#     raw_mereged = raw_mereged.sort_values(by='Date')

    tweet_sentiment=tweet_sentiment[tweet_sentiment["Date"].isin(TARGET_DATE)]

    df_test_merged_l.append(tweet_sentiment)

    

    if(i<1):    

        print("tweet sentiment size:"+str(tweet_sentiment.shape[0]))

        print("tweet df:"+str(tweet_sentiment))
df_test_merged_l[0]
#model 1 = Linear Regression

#model 2 = Multilayer Perceptron



#preprocess

NUM_OF_DAY_PAST =df_merged_l[0].shape[0]



def splitXY(input_df):

    #split to x, y dataframe

    train_df_x=input_df[["Date","no_of_positive_tweet", "no_of_negative_tweet", "no_of_tweet", "score"]]

    train_df_y=input_df[["Adj Close"]]

    return train_df_x, train_df_y



def extractDate(input_df):

    input_df['Year']=[d.split('/')[0] for d in input_df.Date]

    input_df['Month']=[d.split('/')[1] for d in input_df.Date]

    input_df['Day']=[d.split('/')[2] for d in input_df.Date]

    input_df=input_df.drop(columns=['Date'])

    return input_df



def normalize(input_df, input_df_2):

    #normalization     

    splittingIdx = input_df.shape[0]

    merged_df = pd.concat([input_df,input_df_2]).reset_index(drop=True)

    merged_df['nof day after first day']=merged_df.index

    

    scaler = MinMaxScaler(feature_range=(0, 1))

    merged_scaler=scaler.fit_transform(merged_df)

    scal_1 = merged_scaler[:splittingIdx,:].copy()

    scal_2 = merged_scaler[splittingIdx:,:].copy()

#     print(scal_1)

#     print(scal_2)

    

    return scal_1,scal_2



#plot result

def plot_result(train_y,pred_raw_y,stock_idx, model_name):

    #visualizaion by plot graph

    fig1, ax1 = plt.subplots()

    ax1.plot(train_y[400:])

    olen = len(train_y)

    test_idx_l = []

    pred_y = []

    

    for x in pred_raw_y:

        pred_y.append(x[0])

    for i in range(0,len(pred_y),1):

        test_idx_l.append((olen+i))

        

    test_series = pd.Series(pred_y, index=test_idx_l)        

    ax1.plot(test_series)

    

    ax1.set_title("Prediction of Stock "+str(stock_idx)+" using "+model_name)

    ax1.set_xlabel("No. of days after 2012/9/4")



def linearRegression(train_df_x, train_df_y, test_df):

    model = LinearRegression()

    model.fit(train_df_x,train_df_y)

    pred_y = model.predict(test_df) 

    return pred_y



def neuralNetworkRegressor(train_df_x, train_df_y, test_df):

    model_mlp = MLPRegressor(

        hidden_layer_sizes=(100, 100 ),  activation='logistic', solver='adam', alpha=0.01, batch_size='auto',

        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,

        random_state=4, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

        early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model_mlp.fit(train_df_x, train_df_y.values.ravel())

    pred_y = model_mlp.predict(test_df)

    pred_y=pred_y.reshape(-1,1)

    return pred_y
#model 1 = Linear Regression

#model 2 = Multilayer Perceptron



def normalize_train(pre_df_x):

    scaler = MinMaxScaler(feature_range=(0, 1))

    pre_df_x=scaler.fit_transform(pre_df_x)

    return pre_df_x



def plot_train_result(train_y,test_y,pred_y,stock_idx, model_name):

    #visualizaion by plot graph

    fig1, ax1 = plt.subplots()

    ax1.plot(train_y)

    plot_df=test_y.copy()

    plot_df['Predict'] = pred_y

    ax1.plot(plot_df[['Adj Close', 'Predict']])

    ax1.set_title("Prediction of Stock "+str(stock_idx)+" using "+model_name)

    ax1.set_xlabel("No. of days after 2012/9/4")

    

def evaluate_error(test_y, pred_y, model_name):

    mean_square=np.mean(np.power((np.array(test_y)-np.array(pred_y)),2))

    root_mean_square=np.sqrt(mean_square)

    percentage_error=np.mean(np.divide(np.absolute(np.array(test_y)-np.array(pred_y)), np.array(test_y)))

    print("["+model_name+"] Mean square error="+str(mean_square))

    #print(" Root mean square error="+str(root_mean_square))

    print("["+model_name+"] Percentage error="+str(percentage_error*100)+"%")    

    return percentage_error

percentage_error_l_1=[]

percentage_error_l_2=[]

percentage_error_l_3=[]



def big_process(raw_df, stock_idx):

    #preprocessing

    pre_df_x, pre_df_y =splitXY(raw_df)

    pre_df_x=extractDate(pre_df_x)

    pre_df_x=normalize_train(pre_df_x)



    #cross validation

    train_x, test_x, train_y, test_y = train_test_split(pre_df_x, pre_df_y, test_size=0.05, shuffle=False)

    

    pred_linear_y = linearRegression(train_x, train_y , test_x)

    plot_train_result(train_y,test_y,pred_linear_y,stock_idx, "Linear Regression")

    pred_mlp_y = neuralNetworkRegressor(train_x, train_y , test_x)

    plot_train_result(train_y,test_y,pred_mlp_y,stock_idx, "MLP Regression")

    ensembled_y = np.add(pred_linear_y, pred_mlp_y)/2

    plot_train_result(train_y,test_y,ensembled_y,stock_idx, "Ensembled Linear Regression + MLP Regression")

    

    #error analysis

    percentage_error_1= evaluate_error(test_y, pred_linear_y, "Linear Regression")

    percentage_error_l_1.append(percentage_error_1)

    percentage_error_2= evaluate_error(test_y, pred_mlp_y,"MLP Regression")

    percentage_error_l_2.append(percentage_error_2)

    percentage_error_3 =evaluate_error(test_y, ensembled_y, "Ensembled Linear Regression + MLP Regression")

    percentage_error_l_3.append(percentage_error_3)
for i in range(0, len(raw_price_train_l), 1):

    print("-------------------------------------------------TRAINING MODEL "+str(i)+"-------------------------------------")

    raw_df=df_merged_l[i]

    print(raw_df.iloc[:1,:])

    big_process(raw_df, (i+1))
def avg_percent_err(err_l, model_name):    

    avg= sum(err_l)/len(err_l)

    print("[%s] Average percentage error=%d" % (model_name,(avg*100)) +"%")



avg_percent_err(percentage_error_l_1, "Linear Regression")

avg_percent_err(percentage_error_l_2, "MLP Regression")

avg_percent_err(percentage_error_l_3, "Ensembled Linear Regression + MLP Regression")
def convert_to_requirment_l(l, ensembled_pred_y):

    print(len(ensembled_pred_y))

    for x in ensembled_pred_y:

        l.append(x[0])

        

def test_big_process(train_df, test_df, stock_idx):

    #Preprocessing

    train_df_x, train_df_y = splitXY(train_df)

    train_df_x=extractDate(train_df_x)

    test_df=extractDate(test_df)

    train_df_x,test_df =normalize(train_df_x,test_df)



    #Modelling

    linear_pred_y=linearRegression(train_df_x, train_df_y, test_df)    

    plot_result(train_df_y,linear_pred_y, stock_idx, "Linear Regression")

    mlp_pred_y=neuralNetworkRegressor(train_df_x, train_df_y, test_df)    

    plot_result(train_df_y, mlp_pred_y,stock_idx, "MLP Regression")

    ensembled_pred_y =(np.add(linear_pred_y,mlp_pred_y))/2

    convert_to_requirment_l(pred_result_l, mlp_pred_y)

    plot_result(train_df_y, ensembled_pred_y,stock_idx, "Ensembling Linear Regression+MLP Regression")
pred_result_l=[]



for i in range(0, len(raw_price_train_l), 1):

    print("-------------------------------------------------TRAINING MODEL "+str(i)+"-------------------------------------")

    rain_df=df_merged_l[i]

    test_df=df_test_merged_l[i]

    test_big_process(rain_df, test_df, (i+1))
print(len(pred_result_l))

print(pred_result_l)
import pickle



fw = open("result.pkl", 'wb')

pickle.dump(pred_result_l, fw)

fw.close()