import matplotlib.pyplot as plt

import os

import pandas as pd

import seaborn as sns

import numpy as np

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.svm import OneClassSVM

from numpy.random import seed

from keras.layers import Input, Dropout

from keras.layers.core import Dense 

from keras.models import Model, Sequential, load_model

from keras import regularizers

from keras.models import model_from_json

from scipy.special import softmax
main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

main_df.describe()
#heatmap of correlations from -1 to 1

sns.heatmap(main_df.corr(), vmin= -1, vmax = 1)
main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

#Dropping nonsense columns for this proposal                                 (axis=1) = columns

main_df = main_df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp'], axis=1)

#Flipping column values

main_df['pCut::Motor_Torque'] = main_df['pCut::Motor_Torque'] *-1

#Heatmap

sns.heatmap(main_df.corr(), vmin= -1, vmax = 1)

def handle_non_numeric(df):

    # Values in each column for each column

    columns = df.columns.values

    

    for column in columns:

        

        # Dictionary with each numerical value for each text

        text_digit_vals = {}

        

        # Receives text to convert to a number

        def convert_to_int (val):

            

            # Returns respective numerical value for class

            return text_digit_vals[val]

        

        # If values in columns are not float or int

        if df[column].dtype !=np.int64 and df[column].dtype != np.float64:

            

            # Gets values form current column

            column_contents = df[column].values.tolist()

            

            # Gets unique values from current column

            unique_elements = set(column_contents)

            

            # Classification starts at 0

            x=0

            

            for unique in unique_elements:

                

                # Adds the class value for the text in dictionary, if it's not there

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x+=1

            

            # Maps the numerical values to the text values in columns 

            df[column] = list(map(convert_to_int, df[column]))

    

    return df
#Grabbing the entire dataset

main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

#Dropping columns with unwanted/irrelevant info for the algorithm

main_df = main_df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp'], axis=1)

#Transforming modes into classified data

main_df = handle_non_numeric(main_df)



#Passing our dataframe as our features

X = main_df



#Defining preprocessor for the data

scaler = preprocessing.MinMaxScaler()

#Preprocessing

X = pd.DataFrame(scaler.fit_transform(X), 

                              columns=X.columns, 

                              index=X.index)





#Scaling

X = preprocessing.scale(X)

#Splitting the feature data for training data. First 200.000 rows.

X_train = X[:200000]





#Creating a fitting OneClass SVM

ocsvm = OneClassSVM(nu=0.25, gamma=0.05)

ocsvm.fit(X_train)




df=main_df.copy()

df['anomaly'] = pd.Series(ocsvm.predict(X))



#Saving Dataframe.

df.to_csv('Labled_df.csv')
#Reading into dataframe

df = pd.read_csv('../input/created/Labled_df.csv', index_col=0)

df.head()
#Getting labled groups

scat_1 = df.groupby('anomaly').get_group(1)

scat_0 = df.groupby('anomaly').get_group(-1)



# Plot size

plt.subplots(figsize=(15,7))



# Plot group 1 -labeled, color green, point size 1

plt.plot(scat_1.index,scat_1['pCut::Motor_Torque'], 'g.', markersize=1)



# Plot group -1 -labeled, color red, point size 1

plt.plot(scat_0.index, scat_0['pCut::Motor_Torque'],'r.', markersize=1)

#Creating a dataframe for the score of each data sample

score = pd.DataFrame()

#Returning scores for the dataset

score['score'] = ocsvm.score_samples(X)



#Plot size

plt.subplots(figsize=(15,7))

#Plotting

score['score'].plot()

#Saving score dataframe

score.to_csv('SVM_Score.csv')
fig, ax = plt.subplots(figsize=(15,7))





((score['score'].rolling(20000).mean())*-1).plot(ax=ax)
plt.subplots(figsize=(15,7))

plt.plot(score.index, score['score'],'r.', markersize=1)
#------ Preparing features for training and future prediction -----

main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

main_df = main_df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp'], axis=1)

main_df = handle_non_numeric(main_df)

X = main_df



scaler = preprocessing.MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), 

                              columns=X.columns, 

                              index=X.index)







X = preprocessing.scale(X)

#-------------------------------------------------------------------





#Percentage of the data that will be considered healthy condition

train_percentage = 0.15

#Integer value for the slice that will be considered healthy condition

train_size = int(len(main_df.index)*train_percentage)

#Grabbing slice for training data

X_train = X[:train_size]





#Defining KMeans with 1 cluster

kmeans = KMeans(n_clusters=1)

#Fitting the algorithm

kmeans.fit(X_train)



#Creating a copy of the main dataset

k_anomaly = main_df.copy()



#Dataframe now will receive the distance of each data sample from the cluster

k_anomaly = pd.DataFrame(kmeans.transform(X))



#Saving cluster distane into csv file

k_anomaly.to_csv('KM_Distance.csv')



#Plot

plt.subplots(figsize=(15,7))



plt.plot(k_anomaly.index, k_anomaly[0], 'g', markersize=1)
#------------------------- Preparing data for training --------------------------- 

main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

main_df = main_df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp'], axis=1)

main_df = handle_non_numeric(main_df)

X = main_df



scaler = preprocessing.MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), 

                              columns=X.columns, 

                              index=X.index)







X = preprocessing.scale(X)





train_percentage = 0.15

train_size = int(len(main_df.index)*train_percentage)



X_train = X[:train_size]

#----------------------------------------------------------------------------------







#Seed for random batch validation and training

seed(10)





#Elu activatoin function

act_func = 'elu'



# Input layer

model=Sequential()



# First hidden layer, connected to input vector X. 

model.add(Dense(50,activation=act_func,

                kernel_initializer='glorot_uniform',

                kernel_regularizer=regularizers.l2(0.0),

                input_shape=(X_train.shape[1],)

               )

         )

# Second hidden layer

model.add(Dense(10,activation=act_func,

                kernel_initializer='glorot_uniform'))

# Thrid hidden layer

model.add(Dense(50,activation=act_func,

                kernel_initializer='glorot_uniform'))



# Input layer

model.add(Dense(X_train.shape[1],

                kernel_initializer='glorot_uniform'))



# Loss function and Optimizer choice

model.compile(loss='mse',optimizer='adam')



# Train model for 50 epochs, batch size of 200 

NUM_EPOCHS=50

BATCH_SIZE=200



#Grabbing validation and training loss over epochs

history=model.fit(np.array(X_train),np.array(X_train),

                  batch_size=BATCH_SIZE, 

                  epochs=NUM_EPOCHS,

                  validation_split=0.1,

                  verbose = 1)

plt.subplots(figsize=(15,7))



plt.plot(history.history['loss'],'b',label='Training loss')

plt.plot(history.history['val_loss'],'r',label='Validation loss')

plt.legend(loc='upper right')

plt.xlabel('Epochs')

plt.ylabel('Loss, [mse]')



plt.show()
#Reconstructing train data

X_pred = model.predict(np.array(X_train))



#Creating dataframe for reconstructed data

X_pred = pd.DataFrame(X_pred,columns=main_df.columns)

X_pred.index = pd.DataFrame(X_train).index



#Dataframe to get the difference of predicted data and real data. 

scored = pd.DataFrame(index=pd.DataFrame(X_train).index)

#Returning the mean of the loss for each column

scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)



#plot

plt.subplots(figsize=(15,7))

sns.distplot(scored['Loss_mae'],

             bins = 15, 

             kde= True,

            color = 'blue');









#Reconstructing full data

X_pred = model.predict(np.array(X))

X_pred = pd.DataFrame(X_pred,columns=main_df.columns)

X_pred.index = pd.DataFrame(X).index



#Returning mean of the losses for each column and putting it in a dataframe

scored = pd.DataFrame(index=pd.DataFrame(X).index)

scored['Loss_mae'] = np.mean(np.abs(X_pred-X), axis = 1)



#Plot size

plt.subplots(figsize=(15,7))





#Saving dataframe

scored.to_csv('AutoEncoder_loss.csv')



#Plot

plt.plot(scored['Loss_mae'],'b',label='Prediction Loss')



plt.legend(loc='upper right')

plt.xlabel('Sample')

plt.ylabel('Loss, [mse]')
#Plot size

plt.subplots(figsize=(15,7))

#Reading loss csv file

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')

#Plot

plt.plot(enc_loss.index,enc_loss['Loss_mae'], 'g.', markersize=1,label="AutoEncoder Loss")

#Labels and legends

plt.legend(loc='upper right')

plt.xlabel('Sample')

#Show plot

plt.show()



plt.subplots(figsize=(15,7))

k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

plt.plot(k_anomaly.index,k_anomaly['0'], 'g.', markersize=1,label="KM cluster Distance")

plt.legend(loc='upper right')

plt.xlabel('Sample')

plt.show()



plt.subplots(figsize=(15,7))

score = pd.read_csv('../input/created/SVM_Score.csv')

plt.plot(score.index,score['score'], 'g.', markersize=1,label="OCSVM score")

plt.legend(loc='upper right')

plt.xlabel('Sample')

plt.show()

#Plot size

plt.subplots(figsize=(15,7))



#Reading each socring csv file

k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



#Scaling data for vizualization

k_distance = k_anomaly/k_anomaly.max()

svm_score = (score/score.max())*-1



plt.plot(enc_loss.index,enc_loss['Loss_mae'], label="AutoEncoder Loss")

plt.plot(svm_score.index, svm_score['score'],label="OCSVM score")

plt.plot(k_distance.index,k_distance['0'], label="Kmeans Euclidean Dist")







plt.gca().legend(('AutoEncoder Loss','OCSVM score * -1','Kmeans Euclidean Dist'))

#Reading score files

k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



# Dataframe to see correlation

corr = pd.DataFrame()



#Passing score data to corr dataframe 

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']



#Seeing correlation

corr.corr()
#---- Reading data and passing it to dataframe again ----- 



k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']

#---------------------------------------------------------





#Plot size

plt.subplots(figsize=(15,7))



#Scatter plot of SVM score

plt.plot(corr.index, corr['SVM_score'], 'g.', markersize=1, label = 'OCSVM_score')

#Plotting moving mean of 1000 data points

plt.plot(corr.index, corr['SVM_score'].rolling(1000).mean(), 'r', markersize=1, label = 'Moving Mean')

#Legend

plt.legend(loc='upper right')

#Show

plt.show()





plt.subplots(figsize=(15,7))

  

plt.plot(corr.index, corr['KM_cluster_distance'], 'g.', markersize=1, label = 'KM_cluster_distance')

plt.plot(corr.index, corr['KM_cluster_distance'].rolling(1000).mean(), 'r', markersize=1, label = 'Moving Mean')



plt.legend(loc='upper right')

plt.show()











plt.subplots(figsize=(15,7))

  

plt.plot(corr.index, corr['AutoEnc_loss'], 'g.', markersize=1, label = 'AutoEnc_loss')

plt.plot(corr.index, corr['AutoEnc_loss'].rolling(1000).mean(), 'r', markersize=1, label = 'Moving Mean')





plt.legend(loc='upper right')

plt.show()



k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']







#Plot size

plt.subplots(figsize=(10,7))

#Hist plot of first 160.000 rows, 15 bins

sns.distplot(corr['SVM_score'].head(160000), bins=15)

#Show

plt.show()



plt.subplots(figsize=(10,7))

sns.distplot(corr['KM_cluster_distance'].head(160000),bins=15)

plt.show()



plt.subplots(figsize=(10,7))

sns.distplot(corr['AutoEnc_loss'].head(160000),bins=15)

plt.show()
k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']











plt.subplots(figsize=(10,7))

sns.distplot(corr['SVM_score'], bins=15)

plt.show()



plt.subplots(figsize=(10,7))

sns.distplot(corr['KM_cluster_distance'],bins=15)

plt.show()



plt.subplots(figsize=(10,7))

sns.distplot(corr['AutoEnc_loss'],bins=15)

plt.show()
k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']







#Creating an array for the thresholds to be plotted over the entire dataset

lower_threshold = np.full((corr['SVM_score'].size, 1), 0)

upper_threshold = np.full((corr['SVM_score'].size, 1), 18000)

high_density_threshold = np.full((corr['SVM_score'].size, 1), 13250)



#Plot size

plt.subplots(figsize=(15,7))



#Score Plot

plt.plot(corr.index, corr['SVM_score'], 'k', markersize=1, label = 'OCSVM_score')

#Moving mean plot

plt.plot(corr.index, corr['SVM_score'].rolling(100).mean(), 'r', markersize=1, label = 'Moving Mean')

#Threshold plots

plt.plot(corr.index, lower_threshold, label='Lower Threshold')

plt.plot(corr.index, upper_threshold, label = 'Upper Threshold')

plt.plot(corr.index, high_density_threshold, label = 'Highest Density')

plt.legend(loc='upper right')

#Show

plt.show()





lower_threshold = np.full((corr['KM_cluster_distance'].size, 1), 1.2)

upper_threshold = np.full((corr['KM_cluster_distance'].size, 1), 17.5)

high_density_threshold = np.full((corr['KM_cluster_distance'].size, 1), 2.5)



plt.subplots(figsize=(15,7))

  

plt.plot(corr.index, corr['KM_cluster_distance'], 'k', markersize=1, label = 'KM_cluster_distance')

plt.plot(corr.index, corr['KM_cluster_distance'].rolling(100).mean(), 'r', markersize=1, label = 'Moving Mean')

plt.plot(corr.index, lower_threshold, label='Lower Threshold')

plt.plot(corr.index, upper_threshold, label = 'Upper Threshold')

plt.plot(corr.index, high_density_threshold, label = 'Highest Density')

plt.legend(loc='upper right')

plt.show()







lower_threshold = np.full((corr['AutoEnc_loss'].size, 1), 0)

upper_threshold = np.full((corr['AutoEnc_loss'].size, 1), 0.1)

high_density_threshold = np.full((corr['AutoEnc_loss'].size, 1), 0.05)



plt.subplots(figsize=(15,7))

  

plt.plot(corr.index, corr['AutoEnc_loss'], 'k', markersize=1, label = 'AutoEnc_loss')

plt.plot(corr.index, corr['AutoEnc_loss'].rolling(100).mean(), 'r', markersize=1, label = 'Moving Mean')

plt.plot(corr.index, lower_threshold, label='Lower Threshold')

plt.plot(corr.index, upper_threshold, label = 'Upper Threshold')

plt.plot(corr.index, high_density_threshold, label = 'Highest Density')

plt.legend(loc='upper right')

plt.show()

k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']









plt.subplots(figsize=(15,7))

  

plt.plot(corr['KM_cluster_distance'],corr['SVM_score'],'b.',markersize=1 )

plt.xlabel('KM')

plt.ylabel('SVM')

plt.show()





plt.subplots(figsize=(15,7))

  

plt.plot(corr['AutoEnc_loss'],corr['SVM_score'],'b.' ,markersize=1 )

plt.xlabel('Encoder')

plt.ylabel('SVM')

plt.show()



plt.subplots(figsize=(15,7))

  

plt.plot(corr['AutoEnc_loss'],corr['KM_cluster_distance'],'b.' ,markersize=1 )

plt.xlabel('Encoder')

plt.ylabel('KM')

plt.show()
k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']



main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')







#Passing encoder loss to main dataframe, to make it easier to separate by month

main_df['AutoEnc_loss'] = corr['AutoEnc_loss']



#Getting list of months

months = main_df['month'].dropna().unique()



#Looping through every month

for month in months:

    #Grabbing the slice of the dataframe for each month 

    month_df = main_df.groupby('month').get_group(month)

    

    

    # Array Thresholds

    upper_threshold = np.full((month_df['AutoEnc_loss'].size, 1), 0.1)

    high_density_threshold = np.full((month_df['AutoEnc_loss'].size, 1), 0.05)



    #Plot

    plt.subplots(figsize=(15,7))

    plt.plot(month_df.index, month_df['AutoEnc_loss'], label=f'AutoEnc_loss month_{month}')

    plt.plot(month_df.index, upper_threshold, label = 'Upper Threshold')

    plt.plot(month_df.index, high_density_threshold, label = 'Highest Density')

    plt.legend(loc='upper right')

    plt.ylim(0,1.3)

    

    plt.show()

    

    
k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']



main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')





main_df['AutoEnc_loss'] = corr['AutoEnc_loss']











months = main_df['month'].dropna().unique()



for month in months:

    month_df = main_df.groupby('month').get_group(month)

    

    

    

    plt.subplots(figsize=(15,7))

    sns.distplot((month_df['AutoEnc_loss']), bins=15).set_title(f'Month {month} Loss Distribution')

    #X axis limits

    plt.xlim([-1.2,1.2])

    plt.show()



    

    

    

    
k_anomaly = pd.read_csv('../input/created/KM_Distance.csv')

score = pd.read_csv('../input/created/SVM_Score.csv')

enc_loss = pd.read_csv('../input/created/AutoEncoder_loss.csv')



corr = pd.DataFrame()

corr['SVM_score'] = score['score']

corr['KM_cluster_distance'] = k_anomaly['0']

corr['AutoEnc_loss'] = enc_loss['Loss_mae']



main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')





main_df['AutoEnc_loss'] = corr['AutoEnc_loss']





months = main_df['month'].dropna().unique()



for month in months:

    month_df = main_df.groupby('month').get_group(month)

    kurt = (month_df['AutoEnc_loss']).kurtosis()

    print(f'Month {month} kurtosis = {kurt}')
main_df = pd.read_csv('../input/datasetsone-year-compiledcsv/One_year_compiled.csv')

main_df = main_df.drop(['day', 'hour', 'sample_Number', 'month', 'timestamp'], axis=1)

main_df = handle_non_numeric(main_df)

X = main_df



scaler = preprocessing.MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), 

                              columns=X.columns, 

                              index=X.index)







X = preprocessing.scale(X)





train_percentage = 0.15

train_size = int(len(main_df.index)*train_percentage)



X_train = X[:train_size]



seed(10)



act_func = 'elu'



# Input layer:

model=Sequential()

# First hidden layer, connected to input vector X. 

model.add(Dense(50,activation=act_func,

                kernel_initializer='glorot_uniform',

                kernel_regularizer=regularizers.l2(0.0),

                input_shape=(X_train.shape[1],)

               )

         )



model.add(Dense(10,activation=act_func,

                kernel_initializer='glorot_uniform'))



model.add(Dense(50,activation=act_func,

                kernel_initializer='glorot_uniform'))



model.add(Dense(X_train.shape[1],

                kernel_initializer='glorot_uniform'))



model.compile(loss='mse',optimizer='adam')



# Train model for 100 epochs, batch size of 10: 

NUM_EPOCHS=50

BATCH_SIZE=200



history=model.fit(np.array(X_train),np.array(X_train),

                  batch_size=BATCH_SIZE, 

                  epochs=NUM_EPOCHS,

                  validation_split=0.1,

                  verbose = 1)
#Predicting and passing prediction to dataframe

X_pred = model.predict(np.array(X))

X_pred = pd.DataFrame(X_pred,columns=main_df.columns)

X_pred.index = pd.DataFrame(main_df).index



#Passing X from an array to a dataframe

X = pd.DataFrame(X,columns=main_df.columns)

X.index = pd.DataFrame(main_df).index



#Dataframe where all the loss per columns will go

loss_df = pd.DataFrame()



#Dropping mode as it can't logically contribute to degredation

main_df.drop('mode',axis=1, inplace=True)



#Iterating through columns

for column in main_df.columns:

    #Getting the loss of the prediction for that column

    loss_df[f'{column}'] = (X_pred[f'{column}'] - X[f'{column}']).abs()

     

    #Plotting the loss

    plt.subplots(figsize=(15,7))

    plt.plot(loss_df.index, loss_df[f'{column}'], label=f'{column} loss')

    plt.legend(loc='upper right')

    

    plt.show()



#Saving loss Dataframe

loss_df.to_csv('AutoEncoder_loss_p_column.csv')
sftmax_df = pd.read_csv('../input/created/AutoEncoder_loss_p_column.csv', index_col=0)

sftmax_df = softmax(sftmax_df, axis=1)

sftmax_df.describe()
for column in sftmax_df.columns:

    



    plt.subplots(figsize=(15,7))

    plt.plot(sftmax_df.index, sftmax_df[f'{column}'], label=f'{column} loss')

    plt.legend(loc='upper right')

    

    plt.show()
plt.subplots(figsize=(15,7))



#Labels for stackbar plot

df_label = ['Torque', 'Cut lag','Cut speed','Cut position','Film position','Film speed','Film lag','VAX']



#Stackbar plot

plt.stackplot(sftmax_df.index, sftmax_df['pCut::Motor_Torque'],

             sftmax_df['pCut::CTRL_Position_controller::Lag_error'],

             sftmax_df['pCut::CTRL_Position_controller::Actual_speed'],

              sftmax_df['pCut::CTRL_Position_controller::Actual_position'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Actual_position'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Actual_speed'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Lag_error'],

             sftmax_df['pSpintor::VAX_speed'],

             labels = df_label)



plt.legend(loc='upper center', ncol=8)



plt.ylim(0,1)
plt.subplots(figsize=(15,7))



df_label = ['Torque', 'Cut lag','Cut speed','Cut position','Film position','Film speed','Film lag','VAX']



#Grabbing the slice where the larger anomaly is

sftmax_df = sftmax_df[400000:600000]



plt.stackplot(sftmax_df.index, sftmax_df['pCut::Motor_Torque'],

             sftmax_df['pCut::CTRL_Position_controller::Lag_error'],

             sftmax_df['pCut::CTRL_Position_controller::Actual_speed'],

              sftmax_df['pCut::CTRL_Position_controller::Actual_position'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Actual_position'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Actual_speed'],

             sftmax_df['pSvolFilm::CTRL_Position_controller::Lag_error'],

             sftmax_df['pSpintor::VAX_speed'],

             labels = df_label)



plt.legend(loc='upper center', ncol=8)



plt.ylim(0,1)
for column in sftmax_df.columns:

    



    plt.subplots(figsize=(15,7))

    

    sns.distplot(( sftmax_df[f'{column}']), bins=15).set_title(f'Contribution Distribution')

    plt.xlim(0,1)

    

    plt.show()