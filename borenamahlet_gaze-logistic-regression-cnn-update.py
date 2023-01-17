import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os 

import glob

from tqdm import tqdm # for  well-established ProgressBar

import seaborn as sb

from random import shuffle #only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.

import os

print (os.listdir("../input/gaze-position/data/Data"))
data_dir = '../input/gaze-position/data/Data'

train_dir = os.path.join(data_dir, 'TrainData')

test_dir = os.path.join(data_dir, 'TestData')

CATEGORIES = ['ASD', 'Normal']
def create_train_dataframe():

    train = []

    for category_id, category in enumerate(CATEGORIES):

        for csvfile in tqdm(os.listdir(os.path.join(train_dir, category))):

            label=label_data_singleValue(category)

            path=os.path.join(train_dir,category,csvfile)

            traincsv = pd.read_csv(path,header=0,index_col=None)

            traincsv['ASD']=label

            if 'Data/TrainData/Normal/.DS_Store' in path :

                continue

            else :

                train.append(traincsv)

    frame = pd.concat(train, axis=0, ignore_index=True)

    return  frame
asd_df = pd.read_csv("../input/gaze-position/data/Data/TrainData/ASD/log.csv")

asd_df.describe()
asd_df = pd.read_csv("../input/gaze-position/data/Data/TrainData/Normal/log1.csv")

asd_df.describe()
def label_data_singleValue(word_label):

    if word_label == 'ASD': return 1

    elif word_label == 'Normal': return 0
frame=create_train_dataframe()

trainData=frame
frame.describe()
X = trainData.iloc[:,0:16]

Y = trainData.iloc[:,16]
from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

import numpy as np



# split the data into training (67%) and testing (33%)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=16)





# create the model

model = Sequential()

model.add(Dense(64, input_dim=16, kernel_initializer='uniform', activation='relu'))

model.add(Dense(32, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=16, verbose=0)

# evaluate the model

scores = model.evaluate(X_test, Y_test)

print("Accuracy: %.2f%%" % (scores[1]*100))
def decode(datum):

    if(np.argmax(datum)==0):  return 'Normal'

    elif(np.argmax(datum)==1): return 'ASD'

    else: return 'Unknow'
test_df = pd.read_csv("../input/gaze-position/data/Data/TestData/log1.csv")

test_df.describe()
test_df = pd.read_csv("../input/gazedata/data1/Data1/MpaiNewResult.csv" )

test_df.groupby(['centerLX','centerLY']).size()
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_15-54-09.csv" )

test_df.head()
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_15-54-09.csv" )

test_df.sample(100)
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_14-47-04.csv" )

groupbyvalue=test_df.groupby('lookingat').size()

groupbyvalue.plot.bar()
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_14-47-04.csv" )

groupbyvalue=test_df.groupby('lookingat').mean()

groupbyvalue.plot.bar()
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_15-54-09.csv" )

#test_df.groupby('lookingat').mean()

test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_15-54-09.csv" )

groupbyvalue=test_df.groupby(['lookingat']).size()

groupbyvalue.plot.bar()
test_df = pd.read_csv("../input/gazedata/data1/Data1/19-05-16_15-54-09.csv" )

groupbyvalue=test_df.groupby(['blink']).size()

groupbyvalue.plot.bar()
test_df = pd.read_csv("../input/gazedata/data1/Data1/MpaiNewResult.csv")

test_df.describe()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error



y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)



plt.plot(y_pred)

plt.title('Prediction')
test_df = pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidLeftResult.csv")

test_df.describe()

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error

y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)

plt.plot(y_pred)

plt.title('Prediction')
test_df = pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidRightResult.csv")

test_df.describe()

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error



y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)



plt.plot(y_pred)

plt.title('Prediction')
test_df = pd.read_csv("../input/gazedata/data1/Data1/TestData/HellenLeftResult.csv")

test_df.describe()



from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error



y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)





plt.plot(y_pred)

plt.title('Prediction')
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error



y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)





plt.plot(y_pred)

plt.title('Prediction')
import pandas as pd

df =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidLeftResult.csv")

df.describe()
import pandas as pd

df =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidLeftResult.csv")

df.describe()

df = df.iloc[:100,:]

df

#THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



import pandas as pd

import pandas as pd

df =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidLeftResult.csv")

df.describe()

df = df.iloc[:100,:]





# Creating trace1

trace1 = go.Scatter(

                    x = df.centerLX,

                    y = df.centerLY,

                    mode = "lines",

                    name = "Gaze position",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.angleLX)

# Creating trace2



data = [trace1]

layout = dict(title = 'gaze position',

              xaxis= dict(title= 'gaze position',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

py.offline.iplot(fig)

#iplot(fig)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



import pandas as pd

dfNormal =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidLeftResult.csv")

dfASD =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidRightResult.csv")



#df = df.iloc[:100,:]





# Creating trace1

trace1 = go.Scatter(

                    x = dfNormal.centerLX,

                    y = dfNormal.centerLY,

                    mode = "markers",

                    name = "Normal Toddler",

                    marker = dict(size=10,color = 'rgba(16, 112, 2, 0.8)',

                    line = dict(width = 2,)

                    ),

    

                    )

# Creating trace2



# Creating trace1

trace2 = go.Scatter(

                    x = dfASD.centerLX,

                    y = dfASD.centerLY,

                    mode = "markers",

                    name = "ASD Toddler",

                    marker = dict(size=10, color = 'rgba(255, 182, 193, .9)',

                    line = dict(width = 2,)

                                 )

                    )

# Creating trace2

data = [trace1, trace2]



layout = dict(title = 'Gaze Position',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False)

             )



fig = go.Figure(data=data,layout=layout)



py.offline.iplot(fig)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



import pandas as pd



dfNormal =  pd.read_csv("../input/gazedata/data1/Data1/TestData/HellenLeftResult.csv")

#dfASD =  pd.read_csv("../input/gazedata/data1/Data1/TestData/test/MpaidRightResult.csv")

dfASD = pd.read_csv("../input/gazedata/data1/Data1/TestData/HellenRightResult.csv")







dfNormal = dfNormal.iloc[:1000,:]

dfASD = dfASD.iloc[:1000,:]







trace1 = go.Scatter3d(

    x=dfNormal.coordLX,

    y=dfNormal.coordLY,

    z=dfNormal.coordLZ,

    mode='markers',

    name='Normal Toddler',

    marker=dict(

        size=10,

        color='rgba(16, 112, 2, 0.8)',                # set color to an array/list of desired values      

    )

)

trace2 = go.Scatter3d(

    x=dfASD.coordLX,

    y=dfASD.coordLY,

    z=dfASD.coordLZ,

    mode='markers',

    name='Toddler with ASD',

    marker=dict(

        size=10,

        color='rgb(255,182,193,.9)',  

    )

  

)



 

                    

data = [trace1,trace2]



layout = dict(title = 'gaze position',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False),

              

             )



fig = go.Figure(data=data,layout=layout)



py.offline.iplot(fig)






