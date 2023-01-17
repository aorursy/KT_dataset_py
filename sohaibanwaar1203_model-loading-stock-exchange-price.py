# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/market"))



# Any results you write to the current directory are saved as output.
coloum=['ID','symbol','last_price','change','change_percentage','bid_size','bid','offer','offer_size','turnOver','high','low','open','last_volume','total_trades','last_trade','time_stamp']

df=pd.read_csv("../input/market/Stock_Data_06-11.csv", names=coloum,header=None)

sample_df=pd.read_csv("../input/market/sample_solution.csv")

df.head()

df.drop("ID",axis=1,inplace=True)
df.head()
df_test=df.iloc[1500000:2442515]

df=df.iloc[0:1800000]

print(len(df))

print(len(df_test))
from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_AKBL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_AKBL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

AKBL_df_test=df_test[df_test["symbol"]=="AKBL"]

sample_df_test_AKBL=sample_df[sample_df["symbol"]=="AKBL"]



print(len(AKBL_df_test))

print(len(sample_df_test_AKBL))





#taking Last price coloum

Stock_price_AKBL =AKBL_df_test.iloc[:,1:2].values

plt.plot(Stock_price_AKBL[1:100])

plt.show()











import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_AKBL)



X=AKBL_df_test.drop("last_price",axis=1)

y=AKBL_df_test["last_price"]







x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    

for i in range(60,len(sample_df_test_AKBL)):

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])





x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)

yS_pred=loaded_model.predict(x_test)

from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print("Test Score",str(rnn_score))

plt.plot(y_pred)

plt.plot(y_train)

plt.show()





rnn_score = r2_score(y_test,yS_pred)

print("Test Score",str(rnn_score))

plt.plot(yS_pred)

plt.plot(y_test)

plt.show()





y_pred=sc.inverse_transform(yS_pred)

AKBL_prediction=y_pred

sub = pd.DataFrame()

sub['id'] = sample_df_test_AKBL.id

sub['last_price'] = pd.Series(AKBL_prediction.ravel())

sub["symbol"]="AKBL"

sub["timestamp"]=sample_df_test_AKBL.timestamp

sub.to_csv("AKBL_pred")

sub.head()
from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_INDU.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_INDU.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

INDU_df_test=df_test[df_test["symbol"]=="INDU"]

print(len(INDU_df_test))





#taking Last price coloum

Stock_price_INDU =INDU_df_test.iloc[:,2:3].values

plt.plot(Stock_price_INDU[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_INDU)



X=INDU_df_test.drop("last_price",axis=1)

y=INDU_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_hbl.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_hbl.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

HBL_df_test=df_test[df_test["symbol"]=="HBL"]

print(len(HBL_df_test))





#taking Last price coloum

Stock_price_HBL =HBL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_HBL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HBL)



X=HBL_df_test.drop("last_price",axis=1)

y=HBL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_ANL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_ANL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

ANL_df_test=df_test[df_test["symbol"]=="ANL"]

print(len(ANL_df_test))





#taking Last price coloum

Stock_price_ANL =ANL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_ANL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ANL)



X=ANL_df_test.drop("last_price",axis=1)

y=ANL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_BAFL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_BAFL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

BAFL_df_test=df_test[df_test["symbol"]=="BAFL"]

print(len(BAFL_df_test))





#taking Last price coloum

Stock_price_BAFL =BAFL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_BAFL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BAFL)



X=BAFL_df_test.drop("last_price",axis=1)

y=BAFL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_BOP.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_BOP.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

BOP_df_test=df_test[df_test["symbol"]=="BOP"]

sample_df_test_BOP=sample_df[sample_df["symbol"]=="BOP"]



print(len(BOP_df_test))

print(len(sample_df_test_BOP))





#taking Last price coloum

Stock_price_BOP =BOP_df_test.iloc[:,2:3].values

plt.plot(Stock_price_BOP[1:100])

plt.show()











import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BOP)



X=BOP_df_test.drop("last_price",axis=1)

y=BOP_df_test["last_price"]







x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    

for i in range(60,len(sample_df_test_BOP)):

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])





x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)

yS_pred=loaded_model.predict(x_test)

from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print("Test Score",str(rnn_score))

plt.plot(y_pred)

plt.plot(y_train)

plt.show()





rnn_score = r2_score(y_test,yS_pred)

print("Test Score",str(rnn_score))

plt.plot(yS_pred)

plt.plot(y_test)

plt.show()





y_pred=sc.inverse_transform(yS_pred)

BOP_prediction=y_pred

sub = pd.DataFrame()

sub['id'] = sample_df_test_BOP.id

sub['last_price'] = pd.Series(BOP_prediction.ravel())

sub["symbol"]="BOP"

sub["timestamp"]=sample_df_test_BOP.timestamp

sub.to_csv("BOP_pred")

sub.head()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_BYCO.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_BYCO.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

BYCO_df_test=df_test[df_test["symbol"]=="BYCO"]

print(len(BYCO_df_test))





#taking Last price coloum

Stock_price_BYCO =BYCO_df_test.iloc[:,2:3].values

plt.plot(Stock_price_BYCO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BYCO)



X=BYCO_df_test.drop("last_price",axis=1)

y=BYCO_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_DCL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_DCL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

DCL_df_test=df_test[df_test["symbol"]=="DCL"]

print(len(DCL_df_test))





#taking Last price coloum

Stock_price_DCL =DCL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_DCL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DCL)



X=DCL_df_test.drop("last_price",axis=1)

y=DCL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_DFML.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_DFML.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

DFML_df_test=df_test[df_test["symbol"]=="DFML"]

print(len(DFML_df_test))





#taking Last price coloum

Stock_price_DFML =DFML_df_test.iloc[:,2:3].values

plt.plot(Stock_price_DFML[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DFML)



X=DFML_df_test.drop("last_price",axis=1)

y=DFML_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_DGKC.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_DGKC.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

DGKC_df_test=df_test[df_test["symbol"]=="DGKC"]

print(len(DGKC_df_test))





#taking Last price coloum

Stock_price_DGKC =DGKC_df_test.iloc[:,2:3].values

plt.plot(Stock_price_DGKC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DGKC)



X=DGKC_df_test.drop("last_price",axis=1)

y=DGKC_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_ENGRO.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_ENGRO.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

ENGRO_df_test=df_test[df_test["symbol"]=="ENGRO"]

print(len(ENGRO_df_test))





#taking Last price coloum

Stock_price_ENGRO =ENGRO_df_test.iloc[:,2:3].values

plt.plot(Stock_price_ENGRO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ENGRO)



X=ENGRO_df_test.drop("last_price",axis=1)

y=ENGRO_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_FABL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_FABL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

FABL_df_test=df_test[df_test["symbol"]=="FABL"]

print(len(FABL_df_test))





#taking Last price coloum

Stock_price_FABL =FABL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_FABL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FABL)



X=FABL_df_test.drop("last_price",axis=1)

y=FABL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_FEROZ.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_FEROZ.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

FEROZ_df_test=df_test[df_test["symbol"]=="FEROZ"]

print(len(FEROZ_df_test))





#taking Last price coloum

Stock_price_FEROZ =FEROZ_df_test.iloc[:,2:3].values

plt.plot(Stock_price_FEROZ[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FEROZ)



X=FEROZ_df_test.drop("last_price",axis=1)

y=FEROZ_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_FFBL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_FFBL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

FFBL_df_test=df_test[df_test["symbol"]=="FFBL"]

print(len(FFBL_df_test))





#taking Last price coloum

Stock_price_FFBL =FFBL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_FFBL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FFBL)



X=FFBL_df_test.drop("last_price",axis=1)

y=FFBL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_GHNL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_GHNL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

GHNL_df_test=df_test[df_test["symbol"]=="GHNL"]

print(len(GHNL_df_test))





#taking Last price coloum

Stock_price_GHNL =GHNL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_GHNL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_GHNL)



X=GHNL_df_test.drop("last_price",axis=1)

y=GHNL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_HASCOL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_HASCOL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

HASCOL_df_test=df_test[df_test["symbol"]=="HASCOL"]

print(len(HASCOL_df_test))





#taking Last price coloum

Stock_price_HASCOL =HASCOL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_HASCOL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HASCOL)



X=HASCOL_df_test.drop("last_price",axis=1)

y=HASCOL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_HCAR.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_HCAR.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

HCAR_df_test=df_test[df_test["symbol"]=="HCAR"]

print(len(HCAR_df_test))





#taking Last price coloum

Stock_price_HCAR =HCAR_df_test.iloc[:,2:3].values

plt.plot(Stock_price_HCAR[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HCAR)



X=HCAR_df_test.drop("last_price",axis=1)

y=HCAR_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_HUBC.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_HUBC.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

HUBC_df_test=df_test[df_test["symbol"]=="HUBC"]

print(len(HUBC_df_test))





#taking Last price coloum

Stock_price_HUBC =HUBC_df_test.iloc[:,2:3].values

plt.plot(Stock_price_HUBC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HUBC)



X=HUBC_df_test.drop("last_price",axis=1)

y=HUBC_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

rnn_score

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_ISL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_ISL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

ISL_df_test=df_test[df_test["symbol"]=="ISL"]

print(len(ISL_df_test))





#taking Last price coloum

Stock_price_ISL =ISL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_ISL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ISL)



X=ISL_df_test.drop("last_price",axis=1)

y=ISL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print("LOSS ",rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_KAPCO.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_KAPCO.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

KAPCO_df_test=df_test[df_test["symbol"]=="KAPCO"]

print(len(KAPCO_df_test))





#taking Last price coloum

Stock_price_KAPCO =KAPCO_df_test.iloc[:,2:3].values

plt.plot(Stock_price_KAPCO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_KAPCO)



X=KAPCO_df_test.drop("last_price",axis=1)

y=KAPCO_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_KEL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_KEL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

KEL_df_test=df_test[df_test["symbol"]=="KEL"]

print(len(KEL_df_test))





#taking Last price coloum

Stock_price_KEL =KEL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_KEL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_KEL)



X=KEL_df_test.drop("last_price",axis=1)

y=KEL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_LUCK.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_LUCK.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

LUCK_df_test=df_test[df_test["symbol"]=="LUCK"]

print(len(LUCK_df_test))





#taking Last price coloum

Stock_price_LUCK =LUCK_df_test.iloc[:,2:3].values

plt.plot(Stock_price_LUCK[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_LUCK)



X=LUCK_df_test.drop("last_price",axis=1)

y=LUCK_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_MARI.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_MARI.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

MARI_df_test=df_test[df_test["symbol"]=="MARI"]

print(len(MARI_df_test))





#taking Last price coloum

Stock_price_MARI =MARI_df_test.iloc[:,2:3].values

plt.plot(Stock_price_MARI[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_MARI)



X=MARI_df_test.drop("last_price",axis=1)

y=MARI_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_NBP.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_NBP.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

NBP_df_test=df_test[df_test["symbol"]=="NBP"]

print(len(NBP_df_test))





#taking Last price coloum

Stock_price_NBP =NBP_df_test.iloc[:,2:3].values

plt.plot(Stock_price_NBP[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_NBP)



X=NBP_df_test.drop("last_price",axis=1)

y=NBP_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_OGDC.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_OGDC.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

OGDC_df_test=df_test[df_test["symbol"]=="OGDC"]

print(len(OGDC_df_test))





#taking Last price coloum

Stock_price_OGDC =OGDC_df_test.iloc[:,2:3].values

plt.plot(Stock_price_OGDC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_OGDC)



X=OGDC_df_test.drop("last_price",axis=1)

y=OGDC_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_PASL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_PASL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

PASL_df_test=df_test[df_test["symbol"]=="PASL"]

print(len(PASL_df_test))





#taking Last price coloum

Stock_price_PASL =PASL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_PASL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PASL)



X=PASL_df_test.drop("last_price",axis=1)

y=PASL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_PIAA.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_PIAA.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

PIAA_df_test=df_test[df_test["symbol"]=="PIAA"]

print(len(PIAA_df_test))





#taking Last price coloum

Stock_price_PIAA =PIAA_df_test.iloc[:,2:3].values

plt.plot(Stock_price_PIAA[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PIAA)



X=PIAA_df_test.drop("last_price",axis=1)

y=PIAA_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_POL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_POL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

POL_df_test=df_test[df_test["symbol"]=="POL"]

print(len(POL_df_test))





#taking Last price coloum

Stock_price_POL =POL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_POL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_POL)



X=POL_df_test.drop("last_price",axis=1)

y=POL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/model-results/results/model_SNGP.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/model-results/results/model_SNGP.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

SNGP_df_test=df_test[df_test["symbol"]=="SNGP"]

print(len(SNGP_df_test))





#taking Last price coloum

Stock_price_SNGP =SNGP_df_test.iloc[:,2:3].values

plt.plot(Stock_price_SNGP[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_SNGP)



X=SNGP_df_test.drop("last_price",axis=1)

y=SNGP_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_ATLH.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_ATLH.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

ATLH_df_test=df_test[df_test["symbol"]=="ATLH"]

print(len(ATLH_df_test))





#taking Last price coloum

Stock_price_ATLH =ATLH_df_test.iloc[:,2:3].values

plt.plot(Stock_price_ATLH[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ATLH)



X=ATLH_df_test.drop("last_price",axis=1)

y=ATLH_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_DAWH.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_DAWH.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

DAWH_df_test=df_test[df_test["symbol"]=="DAWH"]

print(len(DAWH_df_test))





#taking Last price coloum

Stock_price_DAWH =DAWH_df_test.iloc[:,2:3].values

plt.plot(Stock_price_DAWH[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DAWH)



X=DAWH_df_test.drop("last_price",axis=1)

y=DAWH_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_GHNI.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_GHNI.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

GHNI_df_test=df_test[df_test["symbol"]=="GHNI"]

print(len(GHNI_df_test))





#taking Last price coloum

Stock_price_GHNI =GHNI_df_test.iloc[:,2:3].values

plt.plot(Stock_price_GHNI[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_GHNI)



X=GHNI_df_test.drop("last_price",axis=1)

y=GHNI_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_INIL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_INIL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

INIL_df_test=df_test[df_test["symbol"]=="INIL"]

print(len(INIL_df_test))





#taking Last price coloum

Stock_price_INIL =INIL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_INIL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_INIL)



X=INIL_df_test.drop("last_price",axis=1)

y=INIL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/models/model_JBSL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/models/model_JBSL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

JBSL_df_test=df_test[df_test["symbol"]=="JBSL"]

print(len(JBSL_df_test))





#taking Last price coloum

Stock_price_JBSL =JBSL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_JBSL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_JBSL)



X=JBSL_df_test.drop("last_price",axis=1)

y=JBSL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_MTL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_MTL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

MTL_df_test=df_test[df_test["symbol"]=="MTL"]

print(len(MTL_df_test))





#taking Last price coloum

Stock_price_MTL =MTL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_MTL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_MTL)



X=MTL_df_test.drop("last_price",axis=1)

y=MTL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_PAEL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_PAEL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

PAEL_df_test=df_test[df_test["symbol"]=="PAEL"]

print(len(PAEL_df_test))





#taking Last price coloum

Stock_price_PAEL =PAEL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_PAEL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PAEL)



X=PAEL_df_test.drop("last_price",axis=1)

y=PAEL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_POWER.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_POWER.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

POWER_df_test=df_test[df_test["symbol"]=="POWER"]

print(len(POWER_df_test))





#taking Last price coloum

Stock_price_POWER =POWER_df_test.iloc[:,2:3].values

plt.plot(Stock_price_POWER[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_POWER)



X=POWER_df_test.drop("last_price",axis=1)

y=POWER_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_PPL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_PPL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

PPL_df_test=df_test[df_test["symbol"]=="PPL"]

print(len(PPL_df_test))





#taking Last price coloum

Stock_price_PPL =PPL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_PPL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PPL)



X=PPL_df_test.drop("last_price",axis=1)

y=PPL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_PSX.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_PSX.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

PSX_df_test=df_test[df_test["symbol"]=="PSX"]

print(len(PSX_df_test))





#taking Last price coloum

Stock_price_PSX =PSX_df_test.iloc[:,2:3].values

plt.plot(Stock_price_PSX[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PSX)



X=PSX_df_test.drop("last_price",axis=1)

y=PSX_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()

from keras.models import model_from_json

# load json and create model

json_file = open('../input/muazz12/muazz/model_SEARL.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights('../input/muazz12/muazz/model_SEARL.h5')

print("Loaded model from disk")



import matplotlib.pyplot as plt

SEARL_df_test=df_test[df_test["symbol"]=="SEARL"]

print(len(SEARL_df_test))





#taking Last price coloum

Stock_price_SEARL =SEARL_df_test.iloc[:,2:3].values

plt.plot(Stock_price_SEARL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_SEARL)



X=SEARL_df_test.drop("last_price",axis=1)

y=SEARL_df_test["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)



y_pred=loaded_model.predict(x_train)



from sklearn.metrics import r2_score

rnn_score = r2_score(y_train,y_pred)

print(rnn_score)

plt.plot(y_pred)

plt.plot(y_train)

plt.show()
