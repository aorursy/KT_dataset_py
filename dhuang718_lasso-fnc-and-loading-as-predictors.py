import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import statistics as s 
#import dataset and specify X (independent variables) and y (dependent variable)

df_icn = pd.read_csv('/kaggle/input/trends-assessment-prediction/ICN_numbers.csv')

df_fnc = pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')

df_loading = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')

df_reveal = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv')

df_sample = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')

df_train = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv')

#impute missing training values

from sklearn.impute import KNNImputer



#separate Id column and attributes

df_train2 = df_train[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']]



#impute

imputer = KNNImputer(n_neighbors = 3, weights="uniform")

df_train2 = imputer.fit_transform(df_train2)



#convert the 2d array back to the dataframe and add back the Id column

df_train2 = pd.DataFrame(df_train2, columns = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'])

df_train2 = pd.concat([df_train['Id'], df_train2], axis =1)



df_train_imputed = df_train2.copy()



df_train_imputed
#specify the datasets to be tested in the model, 



#option 1

df_combine_fnc = df_train_imputed.join(df_fnc.set_index('Id'), on = 'Id')



#option 2

df_combine_loading = df_train_imputed.join(df_loading.set_index('Id'), on = 'Id')



#option 3

df_combine = df_train_imputed.join(df_fnc.set_index('Id'), on = 'Id')

df_combine = df_combine.join(df_loading.set_index('Id'), on = 'Id')



#current testing

df = df_combine

from termcolor import colored

text = colored('Currently testing ICs and FNCs as predictors',  'red', attrs=['reverse', 'blink'])

print(text)
#X for df_combine_fnc

X = df.iloc[:, 6:]



#specify the y's

y = df.iloc[:, 0:6]

y_age = df.iloc[:, 1]

y_1_1 = df.iloc[:, 2].round(2)

y_1_2 = df.iloc[:, 3]

y_2_1 = df.iloc[:, 4]

y_2_2 = df.iloc[:, 5]
#less importance to fnc dataframe

FNC_SCALE = 1/300



X = df.iloc[:, 6:]

X.iloc[:,:1378] *= FNC_SCALE

X
#specify features to predict

targets = [y_age, y_1_1, y_1_2, y_2_1, y_2_2]



#weights from TRENDS

weights = [.3, .175, .175, .175, .175]
#framework to be able to run different models on each feature

from sklearn.model_selection import train_test_split



#select models 

from sklearn.linear_model import Lasso

from sklearn.linear_model import BayesianRidge





#score holding

scores_storage = []



for i in targets:

    #split the data

    X_train, X_test, y_train, y_test = train_test_split(X, i, test_size=0.3, random_state=0)

    

    #run the model

    if i.equals(y_age):

        model = Lasso(alpha=0.1)

    elif i.equals(y_1_1):

        model = Lasso(alpha=0.1)

    elif i.equals(y_1_2):

        model = Lasso(alpha=0.1)

    elif i.equals(y_2_1):

        model = Lasso(alpha=0.1)

    elif i.equals(y_2_2):

        model = Lasso(alpha=0.1)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    #get scores of each target

    unweighted_score = (abs(y_test-y_pred)).sum()/y_pred.sum()

    scores_storage.append(unweighted_score)



#multiple scores with weights and sum for final score

grand_score = [scores_storage[i] * weights[i] for i in range(len(scores_storage))]

sum(grand_score)
#get the ID's of the submission

df_sub_id = df_sample.copy()



df_sub_id['Id'] = df_sub_id['Id'].str.slice(0,5,1)

df_sub_id = df_sub_id['Id'].unique()



df_sub_id = pd.DataFrame({'Id' : df_sub_id , 'hold' : np.zeros(len(df_sub_id))})

df_sub_id['Id'] = df_sub_id['Id'].astype(int)
#create the features dataframe for submission Ids

df_sub_combine = df_fnc.join(df_loading.set_index('Id'), on = 'Id')

df_sub_combine = df_sub_combine.join(df_sub_id.set_index('Id'), on = 'Id')

df_sub_combine = df_sub_combine.dropna()



#create the features only dataframe for submission Ids

df_sub_test = df_sub_combine.drop(columns = ['Id', 'hold'])



#less importance to fnc portion

FNC_SCALE = 1/300

df_sub_test.iloc[:,:1378] *= FNC_SCALE

df_sub_test
#model

targets = [y_age, y_1_1, y_1_2, y_2_1, y_2_2]

targets_names = ['y_age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

target_models = [Lasso(alpha=0.1),

                 Lasso(alpha=0.1),

                 Lasso(alpha=0.1),

                 Lasso(alpha=0.1),

                 Lasso(alpha=0.1)]



int_index = [0, 1, 2, 3, 4]



#create empty submission dataframe

submission = pd.DataFrame()



#create submission 

for i in int_index:

    #for i in targets:



    #train the model on the training set

    model = target_models[i]

    model.fit(X, targets[i])



    #predict 

    y_pred = model.predict(df_sub_test)

    

    #add predictions by column

    submission[targets_names[i]] = y_pred

    

submission
#turn dataframe prediction values into one long series

predicted = pd.Series([], dtype = 'float')

for i in range(submission.shape[0]):

    row_values = pd.Series(submission.iloc[i].values)

    predicted = predicted.append(row_values, ignore_index= True)



#add the series to the submission file

df_submission_ensemble = df_sample.copy()

df_submission_ensemble['Predicted'] = predicted

df_submission_ensemble.to_csv('submission_ensemble_linear.csv', index = False)

df_submission_ensemble