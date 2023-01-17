# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import the dependencies

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import SMOTE



# Import Stochastic Gradient Descent in order to improve the learning rate of the neural network

from keras.optimizers import SGD



from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

from matplotlib import pyplot



from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cleaned_data_df = pd.read_csv("../input/cleaned-data-no-draws/cleaned_data.csv")

new_and_old_features_df = pd.read_csv("../input/ufc-model-evaluation-input-datasets/new_and_old_features_df.csv")

new_and_old_features_dropped_aggregated_df = pd.read_csv("../input/ufc-model-evaluation-input-datasets/new_and_old_features_dropped_aggregated_df.csv")

pre_game_df = pd.read_csv("../input/ufc-model-evaluation-input-datasets/pre_game_df.csv")
#Size of each dataframe

cleaned_data_df.shape
new_and_old_features_df.shape
new_and_old_features_dropped_aggregated_df.shape
pre_game_df.shape
#Method for standardising dataframes as some models work better with standardisation

def standardiser(dfName):



    df = dfName

    #Prepare the training set

    df_base = df.copy(deep=True)

    

    #Bringing one-hot encoded columns to the scaled_df

    scaled_df = pd.DataFrame()

    scaled_df["gender"] = df["gender"]

    scaled_df["B_Stance"] = df["B_Stance"]

    scaled_df["R_Stance"] = df["R_Stance"]

    scaled_df["Winner"] = df["Winner"]

    scaled_df["title_bout"] = df["title_bout"]

    scaled_df["weight_class"] = df["weight_class"]

    #scaled_df["no_of_rounds"] = df["no_of_rounds"]

    #scaled_df["B_age"] = df["B_age"]

    #scaled_df["R_age"] = df["R_age"]

    

    #Removing one-hot encoded cols from the df to be scaled

    df.drop(["gender","B_Stance","R_Stance","Winner","title_bout","weight_class"], axis=1)

    

    #storing col names

    column_names = []

    for col in df.columns:

        column_names.append(col)

    

    #Normalising the values in df that are not one-hot encoded

    x_2 = df.values #returns a numpy array

    min_max_scaler = preprocessing.MinMaxScaler()

    x_2_scaled = min_max_scaler.fit_transform(x_2)

    df = pd.DataFrame(x_2_scaled)

    

    counter = 0

    for col in column_names:

        scaled_df[col] = df[counter]

        counter = counter+1

        

    return scaled_df
#obtaining scaled versions of each dataset

scaled_cleaned_data_df = standardiser(cleaned_data_df)

scaled_new_and_old_features_df = standardiser(new_and_old_features_df)

scaled_new_and_old_features_dropped_aggregated_df = standardiser(new_and_old_features_dropped_aggregated_df)

scaled_pre_game_df = standardiser(pre_game_df)
#Checking head

scaled_cleaned_data_df.head(10)
def gaussianNB(dfName,argNum):

    df = dfName

    #Prepare the training set



    # x = feature values, all the columns except the Winner

    x = df.loc[:, df.columns != 'Winner']



    # y = target values, Winner column

    y = df.loc[:, df.columns == 'Winner']

    

    #Split the data into 80% training and 20% testing

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    

    #If argNum = 0, don't change fit. If arg = 1, use near miss and if arg = 2, use SMOTE

    

    if (argNum == 1):

        nr = NearMiss()

        X_train, y_train = nr.fit_sample(X_train, y_train)

    elif (argNum == 2):

        smt = SMOTE()

        X_train, y_train = smt.fit_sample(X_train, y_train)



    #Train the model

    model = GaussianNB()

    model.fit(X_train, y_train.values.ravel()) #Training the model

    

    #Test the model

    predictions = model.predict(X_test)

    #print(predictions)# printing predictions



    print()# Printing new line



    #Check precision, recall, f1-score

    print( classification_report(y_test, predictions) )
#Output of Naive Bayes Model without changing the fit for cleaned_data

gaussianNB(scaled_cleaned_data_df,0)
#Output of Naive Bayes Model with NearMiss undersampling for cleaned_data

gaussianNB(scaled_cleaned_data_df,1)
#Output of Naive Bayes Model with SMOTE oversampling for cleaned_data

gaussianNB(scaled_cleaned_data_df,2)
#Output of Naive Bayes Model without changing the fit for new_and_old_features_df

gaussianNB(scaled_new_and_old_features_df,0)
#Output of Naive Bayes Model with NearMiss undersampling for new_and_old_features_df

gaussianNB(scaled_new_and_old_features_df,1)
#Output of Naive Bayes Model with SMOTE oversampling for new_and_old_features_df

gaussianNB(scaled_new_and_old_features_df,2)
#Output of Naive Bayes Model without changing the fit for new_and_old_features_dropped_aggregated_df

gaussianNB(scaled_new_and_old_features_dropped_aggregated_df,0)
#Output of Naive Bayes Model with NearMiss undersampling for new_and_old_features_dropped_aggregated_df

gaussianNB(scaled_new_and_old_features_dropped_aggregated_df,1)
#Output of Naive Bayes Model with SMOTE oversampling for new_and_old_features_dropped_aggregated_df

gaussianNB(scaled_new_and_old_features_dropped_aggregated_df,2)
#Output of Naive Bayes Model without changing the fit for pre_game_df

gaussianNB(scaled_pre_game_df,0)
#Output of Naive Bayes Model with NearMiss undersampling for pre_game_df

gaussianNB(scaled_pre_game_df,1)
#Output of Naive Bayes Model with SMOTE oversampling for pre_game_df

gaussianNB(scaled_pre_game_df,2)
def logisticsRegression(pdName,argNum):

    df = pdName

    #Prepare the training set



    # x = feature values, all the columns except the Winner

    x = df.loc[:, df.columns != 'Winner']



    # y = target values, Winner column

    y = df.loc[:, df.columns == 'Winner']

    

    #Split the data into 80% training and 20% testing

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    

    #If argNum = 0, don't change fit. If arg = 1, use near miss and if arg = 2, use SMOTE

    

    if (argNum == 1):

        nr = NearMiss()

        X_train, y_train = nr.fit_sample(X_train, y_train)

    elif (argNum == 2):

        smt = SMOTE()

        X_train, y_train = smt.fit_sample(X_train, y_train)



    #Train the model

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train.values.ravel()) #Training the model

    

    #Test the model

    predictions = model.predict(X_test)

    #print(predictions)# printing predictions



    print()# Printing new line



    #Check precision, recall, f1-score

    print( classification_report(y_test, predictions) )
#Output of Logistic Regression Model without changing the fit for cleaned_data

logisticsRegression(scaled_cleaned_data_df,0)
#Output of Logistic Regression Model with NearMiss undersampling for cleaned_data

logisticsRegression(scaled_cleaned_data_df,1)
#Output of Logistic Regression Model with SMOTE oversampling for cleaned_data

logisticsRegression(scaled_cleaned_data_df,2)
#Output of Logistic Regression Model without changing the fit for new_and_old_features_df

logisticsRegression(scaled_new_and_old_features_df,0)
#Output of Logistic Regression Model with NearMiss undersampling for new_and_old_features_df

logisticsRegression(scaled_new_and_old_features_df,1)
#Output of Logistic Regression Model with SMOTE oversampling for new_and_old_features_df

logisticsRegression(scaled_new_and_old_features_df,2)
#Output of Logistic Regression Model without changing the fit for new_and_old_features_dropped_aggregated_df

logisticsRegression(scaled_new_and_old_features_dropped_aggregated_df,0)
#Output of Logistic Regression Model with NearMiss undersampling for new_and_old_features_dropped_aggregated_df

logisticsRegression(scaled_new_and_old_features_dropped_aggregated_df,1)
#Output of Logistic Regression Model with SMOTE oversampling for new_and_old_features_dropped_aggregated_df

logisticsRegression(scaled_new_and_old_features_dropped_aggregated_df,2)
#Output of Logistic Regression Model without changing the fit for pre_game_df

logisticsRegression(scaled_pre_game_df,0)
#Output of Logistic Regression Model with NearMiss undersampling for pre_game_df

logisticsRegression(scaled_pre_game_df,1)
#Output of Logistic Regression Model with SMOTE oversampling for pre_game_df

logisticsRegression(scaled_pre_game_df,2)
def neuralNetwork(dfName,argNum):

    df = dfName

    #Prepare the training set



    # x = feature values, all the columns except the Winner

    x = df.loc[:, df.columns != 'Winner']



    # y = target values, Winner column

    y = df.loc[:, df.columns == 'Winner']

    

    #Split the data into 80% training and 20% testing

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    

    #If argNum = 0, don't change fit. If arg = 1, use near miss and if arg = 2, use SMOTE

    

    if (argNum == 1):

        nr = NearMiss()

        X_train, y_train = nr.fit_sample(X_train, y_train)

    elif (argNum == 2):

        smt = SMOTE()

        X_train, y_train = smt.fit_sample(X_train, y_train)

    

    # create an optimizer with a learning rate of 0.01 and a momentum of 0.9

    # by default there is no momentum used

    # Momentum [1] or SGD with momentum is method which helps accelerate gradients vectors ...

    # in the right directions, thus leading to faster converging. 

    opt = SGD(lr=0.01, momentum=0.9, decay=0.01)

    

    model = Sequential()

    model.add(Dense(x.shape[1], input_dim=x.shape[1], activation='relu')) # input layer

    # model.add(Dense(16, activation='relu'))

    model.add(Dense(32, activation='relu'))

    # model.add(Dense(50, activation='relu'))

    # model.add(Dense(25, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    #an epoch refers to one cycle through the full training dataset

    #loss_settings=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=50)

    

    # to check accuracy

    #model.evaluate(x_test, y_test)[1]

    

    # predict probabilities for test set

    yhat_probs = model.predict(X_test)

    # predict crisp classes for test set

    yhat_classes = model.predict_classes(X_test)

    # reduce to 1d array

    yhat_probs = yhat_probs[:, 0]

    yhat_classes = yhat_classes[:, 0]



    # accuracy: (tp + tn) / (p + n)

    accuracy = accuracy_score(y_test, yhat_classes)

    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)

    precision = precision_score(y_test, yhat_classes)

    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)

    recall = recall_score(y_test, yhat_classes)

    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)

    f1 = f1_score(y_test, yhat_classes)

    print('F1 score: %f' % f1)
#Output of Neural Networks without changing the fit for cleaned_data

neuralNetwork(scaled_cleaned_data_df,0)
#Output of Neural Networks with NearMiss undersampling for cleaned_data_df

neuralNetwork(scaled_cleaned_data_df,1)
#Output of Neural Networks with with SMOTE oversampling for cleaned_data_df

neuralNetwork(scaled_cleaned_data_df,2)
#Output of Neural Networks without changing the fit for new_and_old_features_df

neuralNetwork(scaled_new_and_old_features_df,0)
#Output of Neural Networks with NearMiss undersampling for new_and_old_features_df

neuralNetwork(scaled_new_and_old_features_df,1)
#Output of Neural Networks with with SMOTE oversampling for new_and_old_features_df

neuralNetwork(scaled_new_and_old_features_df,2)
#Output of Neural Networks without changing the fit for new_and_old_features_dropped_aggregated_df

neuralNetwork(scaled_new_and_old_features_dropped_aggregated_df,0)
#Output of Neural Networks with NearMiss undersampling for new_and_old_features_dropped_aggregated_df

neuralNetwork(scaled_new_and_old_features_dropped_aggregated_df,1)
#Output of Neural Networks with with SMOTE oversampling for new_and_old_features_dropped_aggregated_df

neuralNetwork(scaled_new_and_old_features_dropped_aggregated_df,2)
#Output of Neural Networks without changing the fit for pre_game_df

neuralNetwork(scaled_pre_game_df,0)
#Output of Neural Networks with NearMiss undersampling for pre_game_df

neuralNetwork(scaled_pre_game_df,1)
#Output of Neural Networks with with SMOTE oversampling for pre_game_df

neuralNetwork(scaled_pre_game_df,2)
def knn(X, y):

    # this splits the dataset into 80% train data and 20% test data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



    # training and predictions

    

    classifier = KNeighborsClassifier(n_neighbors=10)

    classifier.fit(X_train, y_train)



    # make the prediction

    y_pred = classifier.predict(X_test)



    # evaluating the algorithm

    

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))



    # comparing error rate with the K value

    error = []



    # calculating error for K values between 1 and 40

    for i in range(1, 40):

        knn = KNeighborsClassifier(n_neighbors=i)

        knn.fit(X_train, y_train)

        pred_i = knn.predict(X_test)

        error.append(np.mean(pred_i != y_test))



    plt.figure(figsize=(12,6))

    plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)

    plt.title('Error Rate K Value')

    plt.xlabel('K Value')

    plt.ylabel('Mean Error')
#Output of KNN without changing the fit for cleaned_data



knn_cleaned_data_df = scaled_cleaned_data_df.copy(deep=False)



df1 = knn_cleaned_data_df.iloc[:, :3] 

df2 = knn_cleaned_data_df.iloc[:, 4:]

X = pd.concat([df1, df2], axis=1) # this is all the columns except the class column

y = knn_cleaned_data_df.iloc[:, 3] # only the fourth column (winner) is selected



knn(X, y)
#Output of KNN without changing the fit for new_and_old_features_df



knn_new_and_old_features_df = scaled_new_and_old_features_df



df1 = knn_new_and_old_features_df.iloc[:, :3] 

df2 = knn_new_and_old_features_df.iloc[:, 4:]

X = pd.concat([df1, df2], axis=1) # this is all the columns except the class column

y = knn_new_and_old_features_df.iloc[:, 3] # only the fourth column (winner) is selected





knn(X, y)
#Output of KNN without changing the fit for new_and_old_features_dropped_aggregated_df



knn_new_and_old_features_dropped_aggregated_df = scaled_new_and_old_features_dropped_aggregated_df

df1 = knn_new_and_old_features_dropped_aggregated_df.iloc[:, :3] 

df2 = knn_new_and_old_features_dropped_aggregated_df.iloc[:, 4:]

X = pd.concat([df1, df2], axis=1) # this is all the columns except the class column

y = knn_new_and_old_features_dropped_aggregated_df.iloc[:, 3] # only the fourth column (winner) is selected



knn(X, y)
#Output of KNN without changing the fit for pre_game_df

knn_pre_game_df = scaled_pre_game_df

df1 = knn_pre_game_df.iloc[:, :3] 

df2 = knn_pre_game_df.iloc[:, 4:]

X = pd.concat([df1, df2], axis=1) # this is all the columns except the class column

y = knn_pre_game_df.iloc[:, 3] # only the fourth column (winner) is selected



knn(X, y)
def RandomForest(dfName,argNum):

    df = dfName

    #Prepare the training set



    # x = feature values, all the columns except the Winner

    x = df.loc[:, df.columns != 'Winner']



    # y = target values, Winner column

    y = df.loc[:, df.columns == 'Winner']

    

    #Split the data into 80% training and 20% testing

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    

    #If argNum = 0, don't change fit. If arg = 1, use near miss and if arg = 2, use SMOTE

    

    if (argNum == 1):

        nr = NearMiss()

        X_train, y_train = nr.fit_sample(X_train, y_train)

    elif (argNum == 2):

        smt = SMOTE()

        X_train, y_train = smt.fit_sample(X_train, y_train)



    #Train the model

    

    # initialise Decision Tree

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=50, random_state=424, max_features = "sqrt")

    

    # train model

    clf.fit(X_train,y_train.values.ravel())

    

    #Test the model

    predictions = clf.predict(X_test)

    #print(predictions)# printing predictions



    print()# Printing new line



    #Check precision, recall, f1-score

    print( classification_report(y_test, predictions) )

#Output of Random Forest Model without changing the fit for cleaned_data

RandomForest(scaled_cleaned_data_df,0)
#Output of Random Forest Model with NearMiss undersampling for cleaned_data

RandomForest(scaled_cleaned_data_df,1)
#Output of Random Forest Model with SMOTE oversampling for cleaned_data

RandomForest(scaled_cleaned_data_df,2)
#Output of Random Forest Model without changing the fit for new_and_old_features_df

RandomForest(scaled_new_and_old_features_df,0)
#Output of Random Forest Model with NearMiss undersampling for new_and_old_features_df

RandomForest(scaled_new_and_old_features_df,1)
#Output of Random Forest Model with SMOTE oversampling for new_and_old_features_df

RandomForest(scaled_new_and_old_features_df,2)
#Output of Random Forest Model without changing the fit for new_and_old_features_dropped_aggregated_df

RandomForest(scaled_new_and_old_features_dropped_aggregated_df,0)
#Output of Random Forest Model with NearMiss undersampling for new_and_old_features_dropped_aggregated_df

RandomForest(scaled_new_and_old_features_dropped_aggregated_df,1)
#Output ofRandom Forest Model with SMOTE oversampling for new_and_old_features_dropped_aggregated_df

RandomForest(scaled_new_and_old_features_dropped_aggregated_df,2)
#Output of Random Forest Model without changing the fit for pre_game_df

RandomForest(scaled_pre_game_df,0)
#Output of Random Forest Model with NearMiss undersampling for pre_game_df

RandomForest(scaled_pre_game_df,1)
#Output of Random Forest Model with SMOTE oversampling for pre_game_df

RandomForest(scaled_pre_game_df,2)