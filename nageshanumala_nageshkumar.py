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
#1.2 Keras libraries

from keras.layers import Input, Dense

from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten,Embedding, GRU

from keras.layers.merge import concatenate

from keras.models import Model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import  plot_model



# 1.4 sklearn libraries

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# 1.4 For model plotting

import matplotlib.pyplot as plt

import pydot

from skimage import io



# 1.5 Misc

import os

import time



# 2.2

data = pd.read_csv("../input/mimic3d.csv",

	               compression='infer',

                   encoding="ISO-8859-1"      # 'utf-8' gives error, hence the choice

                  )

data.head(3)




         

data.columns.values   # Our target column is LOSdays

data.tail(3)
data.dtypes  
data.drop(['hadm_id'], axis = 'columns' , inplace = True)





# 2.5 Any missing value?

data.isnull().values.sum()        # 10611

data.columns[data.isnull().sum()  > 0]    # Three: Index(['AdmitDiagnosis', 'religion', 'marital_status'], dtype='object')

data.AdmitDiagnosis = data.AdmitDiagnosis.fillna("missing")

data.religion = data.religion.fillna("missing")

data.marital_status = data.marital_status.fillna("missing")

data.isnull().values.sum()        # 0

dtrain,  dtest = train_test_split(data, test_size=0.33)



# 4. Which columns are 'object'

obj_columns = data.select_dtypes(include = ['object']).columns.values

obj_columns

# 4.1 Which columns have numeric data

num = data.select_dtypes(include = ['int64', 'float64']).columns.values

num





# 4.2 Among object, columns, let us check levels of each

#     column



for i in obj_columns:

	print(i,len(data[i].value_counts()))



num = ['age', 'NumCallouts', 'NumDiagnosis', 'NumProcs',

       'NumCPTevents', 'NumInput', 'NumLabs', 'NumMicroLabs', 'NumNotes',

       'NumOutput', 'NumRx', 'NumProcEvents', 'NumTransfers',

       'NumChartEvents', 'TotalNumInteract']





# 4.4 Final seven obj_columns for One Hot Encoding

obj_cols = ["gender", "admit_type", "admit_location", "insurance" ,"marital_status", 'ExpiredHospital', 'LOSgroupNum']

ohe = OneHotEncoder()

# 4.4.1 Traing on dtrain

ohe = ohe.fit(dtrain[obj_cols])

# 4.4.2 Transform train (dtrain) and test (dtest) data

dtrain_ohe = ohe.transform(dtrain[obj_cols])

dtest_ohe = ohe.transform(dtest[obj_cols])

# 4.4.3

dtrain_ohe.shape       # (39513, 34)

dtest_ohe.shape        # (19463, 34)





# 5.0 Label encode relegion and ethnicity

# 5.1 First 'religion'

le = LabelEncoder()

le.fit(dtrain["religion"])

dtrain["re"] = le.transform(dtrain['religion'])    # Create new column in dtrain

dtest["re"] = le.transform(dtest['religion'])      #   and in dtest



# 5.2 Now 'ethnicity'

le = LabelEncoder()

le.fit(dtrain["ethnicity"])

dtrain["eth"]= le.transform(dtrain['ethnicity'])   # Create new column in dtrain

dtest["eth"]= le.transform(dtest['ethnicity'])     #   and in dtest

# 6. Finally transform two obj_columns for tokenization

te_ad = Tokenizer()

# 6.1 Train tokenizer on train data ie 'dtrain'

te_ad.fit_on_texts(data.AdmitDiagnosis.values)

# 6.2 Transform both dtrain and dtest and create new columns

dtrain["ad"] = te_ad.texts_to_sequences(dtrain.AdmitDiagnosis)

dtest["ad"] = te_ad.texts_to_sequences(dtest.AdmitDiagnosis)



dtrain.head(3)

dtest.head(3)

# 6.3 Similarly for column: AdmitProcedure

te_ap = Tokenizer(oov_token='<unk>')

te_ap.fit_on_texts(data.AdmitProcedure.values)

dtrain["ap"] = te_ap.texts_to_sequences(dtrain.AdmitProcedure)

dtest["ap"] = te_ap.texts_to_sequences(dtest.AdmitProcedure)



dtrain.head(3)

dtest.head(3)



# 7. Standardize numerical data

se = StandardScaler()

# 7.1 Train om dtrain

se.fit(dtrain.loc[:,num])

# 7.2 Then transform both dtrain and dtest

dtrain[num] = se.transform(dtrain[num])

dtest[num] = se.transform(dtest[num])

dtest.loc[:,num].head(3)



# 8. Get max length of the sequences

#    in dtrain["ad"], dtest["ad"]

maxlen_ad = 0

for i in dtrain["ad"]:

	if maxlen_ad < len(i):

		maxlen_ad = len(i)



for i in dtest["ad"]:

	if maxlen_ad < len(i):

		maxlen_ad = len(i)



maxlen_ad     # 24

# 8.1 Get max length of the sequences

#    in dtrain["ap"], dtest["ap"]



maxlen_ap = 0

for i in dtrain["ap"]:

	if maxlen_ap < len(i):

		maxlen_ap = len(i)



maxlen_ap      # 7



for i in dtest["ap"]:

	if maxlen_ap < len(i):

		maxlen_ap = len(i)



maxlen_ap     # 7



# 9. Get max vocabulary size ie value of highest

#    integer in dtrain["ad"] and in dtest["ad"]



one = np.max([np.max(i) for i in dtrain["ad"].tolist() ])

two = np.max([np.max(i) for i in dtest["ad"].tolist() ])

MAX_VOCAB_AD = np.max([one,two])

# 9.1 Get max vocabulary size ie value of highest

#     integer in dtrain["ap"] and in dtest["ap"]



one = np.max([np.max(i) for i in dtrain["ap"].tolist() ])

two = np.max([np.max(i) for i in dtest["ap"].tolist() ])

MAX_VOCAB_AP = np.max([one,two])



# 9.2

MAX_VOCAB_RE = len(dtrain.religion.value_counts())

MAX_VOCAB_ETH = len(dtrain.ethnicity.value_counts())

# 10. Let us put our data in a dictionary form

#     Required when we have multiple inputs

#     to Deep Neural network. Each Input layer

#     should also have the corresponding 'key'

#     name



# 10.1 Training data

Xtr = {

	"num" : dtrain[num].values,          # Note the name 'num'

	"ohe" : dtrain_ohe.toarray(),        # Note the name 'ohe'

	"re"  : dtrain["re"].values,

	"eth" : dtrain["eth"].values,

	"ad"  : pad_sequences(dtrain.ad, maxlen=maxlen_ad),

	"ap"  : pad_sequences(dtrain.ap, maxlen=maxlen_ap )

      }

# 10.2 Test data

Xte = {

	"num" : dtest[num].values,

	"ohe" : dtest_ohe.toarray(),

	"re"  : dtest["re"].values,

	"eth" : dtest["eth"].values,

	"ad"  : pad_sequences(dtest.ad, maxlen=maxlen_ad ),

	"ap"  : pad_sequences(dtest.ap, maxlen=maxlen_ap )

      }



# 10.3 Just check shapes

Xtr["num"].shape         # (39513, 15)

Xtr["ohe"].shape         # (39513, 34)

Xtr["ad"].shape          # (39513, 24)

Xtr["ap"].shape          # (39513, 7)

Xtr["re"].shape          # (39513,)  1D

Xtr["eth"].shape         # (39513,)  1D

# 11. Design a simple model now



dr_level = 0.1

# 11.1

num = Input(

                      shape= (Xtr["num"].shape[1], ),

					  name = "num"            # Name 'num' should be a key in the dictionary for numpy array input

					                          #    That is, this name should be the same as that of key in the dictionary

					  )



# 11.2

ohe =   Input(

                      shape= (Xtr["ohe"].shape[1], ),

					  name = "ohe"

					  )



# 11.3

re =   Input(

                      shape= [1],  # 1D shape or one feature

					  name = "re"

					  )

# 11.4

eth =   Input(

                      shape= [1],  # 1D shape or one feature

					  name = "eth"

					  )

# 11.5

ad =   Input(

                      shape= (Xtr["ad"].shape[1], ),

					  name = "ad"

					  )

# 11.6

ap =   Input(

                      shape= (Xtr["ap"].shape[1],),

					  name = "ap"

					  )



# 12. Embedding layers for each of the two of the columns with sequence data

#     Why add 1 to vocabulary?

#     See: https://stackoverflow.com/questions/52968865/invalidargumenterror-indices127-7-43-is-not-in-0-43-in-keras-r



emb_ad  =      Embedding(MAX_VOCAB_AD+ 1 ,      32  )(ad )

emb_ap  =      Embedding(MAX_VOCAB_AP+ 1 ,      32  )(ap)

# 12.1 Embedding layers for the two categorical variables

emb_re  =      Embedding(MAX_VOCAB_RE+ 1 ,      32  )(re)

emb_eth =      Embedding(MAX_VOCAB_ETH+ 1 ,      32  )(eth)



# 12.2 RNN layers for sequences

rnn_ad = GRU(16) (emb_ad)          # Output of GRU is a vector of size 8

rnn_ap = GRU(16) (emb_ap)







# 12.3 Interim model summary.

#      For 'output' we have all the existing (unterminated) outputs

model = Model([num, ohe, re, eth, ad,ap], [rnn_ad, rnn_ap, emb_re, emb_eth, num, ohe])

model.summary()

# 12.4 Concatenate all outputs

class_l = concatenate([

                      rnn_ad,        # GRU output is already 1D

                      rnn_ap,

                      

                      num,                # 1D output. No need to flatten. See model summary

					  ohe,           # 1D output

					  Flatten()(emb_re),   # Why flatten? See model summary above

					  Flatten()(emb_eth)

                      ]

                     )





# 12.5 Add classification layer

class_l = Dense(64) (class_l)

class_l = Dropout(0.1)(class_l)

class_l = Dense(32) (class_l)

class_l = Dropout(0.1) (class_l)



# 12.6 Output neuron. Activation is linear

#      as our output is continous

output = Dense(1, activation="linear") (class_l)



# 12.7 Formulate Model now

model = Model(

              inputs= [num, ohe, re, eth, ad, ap],

              outputs= output

             )



# 12.8

model.summary()



# 12.9 Model plot uisng keras plot_model()

plt.figure(figsize = (14,14))

plot_model(model, to_file = "model.png")

io.imshow("model.png")

# 13. Compile model

model.compile(loss="mse",

              optimizer="adam",

              metrics=["mae"]

			  )
# 13.1

BATCH_SIZE = 5000

epochs = 20



# 13.2

start = time.time()

history= model.fit(Xtr,

                   dtrain.LOSdays,

                   epochs=epochs,

                   batch_size=BATCH_SIZE,

				   validation_data=(Xte, dtest.LOSdays),

				   verbose = 1

                  )

end = time.time()

print((end-start)/60)
