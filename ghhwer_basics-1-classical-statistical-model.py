import numpy as np # Numpy for numbers 

import pandas as pd # Pandas for data stuff

import os



# PyTorch for modeling

import torch 

from torch.autograd import Variable



#Sklearn for ease of life

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from scipy import stats
# Load dataset

df = pd.read_csv('/kaggle/input/toy-dataset/toy_dataset.csv').drop('Number', axis = 1)



# Remove any obvious outliers

df = df[np.abs(df.Income-df.Income.mean()) <= (3*df.Income.std())]



# Lets just randomize the set

rand_df = df.sample(df.count()[0])



# Separating into train and test

train_test_ratio = 0.7

cut_at = int(rand_df.count()[0]*train_test_ratio)

train, test = rand_df[:cut_at], rand_df[cut_at:]

train
cat = ['City', 'Gender', 'Illness'] # Categorical Fields

ctn = ['Age'] # Continuous Fields

aggs = ['mean','std', 'min', 'max'] # Aggregate Fields
def expands_variables(df, cats, conts, aggs=['mean', 'std', 'max', 'min'], powers=4, put_dummies=True):

    """

    This functions expands categorical fields by aggregating them and appending data as new rows

      this method also puts raises the continuous variables to powers if powers is not None

      It also creates dummies if you ask it to!

    """

    use_powers = (not powers==None) # Should use powers?

    groups = cats + list(set([tuple(sorted([x,y]))  for x in cats for y in cats if x!=y ])) # Permutes categorical to create combinations 

    cont_filters = {key:aggs for key in conts} # Agg dict

    temp = df.copy() # Create a copy from DataFrame

    

    # For each group do groupby and merge DataFrame

    for idx in range(len(groups)):

        g = groups[idx]

        # To get rid of annoying warnnings 

        if type(g) == tuple:

            g = list(g)

        gb = temp.groupby(g).agg(cont_filters) # GroupBy

        gb.columns = ["_".join([x[0],x[1],str(idx)]) for x in list(gb.columns)] # Rename columns so they don't overlap

        temp = pd.merge(temp, gb, on=g, how="left") # Merge DataFrame

    

    if use_powers: # If you desire to use powers

        for x in conts:

            for pw in range(powers-1):

                temp[x+'_pow_'+str(pw+2)] = temp[x]**(pw+2) # Raise them powers and put to their respective names

    if put_dummies:# If you desire to use dummies

        for x in cats: 

            temp = temp.join(pd.get_dummies(temp[x], dtype=float), how='left') # Get them dummies

    return temp # return transformed DataFrame
feature_expanded_train = expands_variables(train, cat, ctn) # Expand Variables

cat_remove_train = feature_expanded_train.drop(cat ,1) # Remove Categorical Values

feature_expanded_test = expands_variables(test, cat, ctn) # Expand Variables

cat_remove_test = feature_expanded_test.drop(cat ,1) # Remove Categorical Values

print('From these columns:')

print(train.columns)

print('\nWe derived these columns:')

print(cat_remove_train.columns) # Let us see the new 
X_train = cat_remove_train.drop(['Income'],1).values # Get X Values for Training 

Y_train = cat_remove_train[['Income']].values # Get True Values for Training 

X_test = cat_remove_test.drop(['Income'],1).values # Get X Values for Training 

Y_test = cat_remove_test[['Income']].values # Get True Values for Training 
X_train = cat_remove_train.drop(['Income'],1).values # Get X Values for Training 

Y_train = cat_remove_train[['Income']].values # Get True Values for Training 

clf = linear_model.Lasso(alpha=0.89)

clf.fit(X_train, Y_train)
def test_model(X_test, Y_test, predict_func, use_batch=False, batch_size=100):

    #predict

    if(use_batch):

        predict = []

        batch_idx = int(X_test.shape[0]/batch_size)

        for x in range(batch_idx):

            batchX = X_test[batch_size*x:batch_size*(x+1)]

            predict.append(predict_func(X_test))

    else:

        predict = predict_func(X_test)

    test_results = test.copy() # Create new DataFrame

    test_results['Model'] = predict # Push Prediction

    test_results['Error'] = test_results.Model/test_results.Income # Push Error from Prediction

    return test_results
sklearnEval = test_model(X_test, Y_test, clf.predict)



import matplotlib.pyplot as plt



fig = plt.gcf()

fig.set_size_inches(10, 10)

ax = plt.subplot(2, 1, 2)

plt.title('Sklearn Model Distribution')

sklearnEval.Income.hist(label='Actual Value')

sklearnEval.Model.hist(label='Prediction')

ax.legend()
on_range = sklearnEval.loc[(sklearnEval.Error >= 0.9) & (sklearnEval.Error <= 1.1)].count()[0]

total = sklearnEval.count()[0]

print('Sklearn 10% Accuracy:'+' %.2f'%((on_range/total)*100)+'%')



on_range = sklearnEval.loc[(sklearnEval.Error >= 0.8) & (sklearnEval.Error <= 1.2)].count()[0]

total = sklearnEval.count()[0]

print('Sklearn 20% Accuracy:'+' %.2f'%((on_range/total)*100)+'%')

print(clf.coef_)

print(clf.intercept_)
i = 0

drop_colinear = []

for x in clf.coef_:

    if(x>-0.001 and x<0.001):

        drop_colinear.append(cat_remove_train.drop(['Income'],1).columns[i])

    i += 1



print("We can drop",len(drop_colinear),"colinear features")
# Filtered Train Set

feature_expanded_train = expands_variables(train, cat, ctn) # Expand Variables

cat_remove_train = feature_expanded_train.drop(cat ,1) # Remove Categorical Values

flltered_expansion_train = cat_remove_train.drop(drop_colinear, 1) # Drop Colinear Variables

# Flitered Test Set

feature_expanded_test = expands_variables(test, cat, ctn) # Expand Variables

cat_remove_test = feature_expanded_test.drop(cat ,1) # Remove Categorical Values

flltered_expansion_test = cat_remove_test.drop(drop_colinear, 1) # Drop Colinear Variables

flltered_expansion_train.head() # Let's take a look!