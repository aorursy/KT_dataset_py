import pandas as pd
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
def gen_features():
    a_levels  = 4     # number of levels per feature
    a_len     = 16000 # number of instances per feature
    A = [ i % a_levels for i in range(a_len)]
    B = [ i % a_levels for i in range(a_len)]
    C = [ i % a_levels for i in range(a_len)]
    D = [ i % a_levels for i in range(a_len)]
    shuffle(A)
    shuffle(B)
    shuffle(C)
    shuffle(D)
    return A, B, C, D
def gen_df():
    A, B, C, D = gen_features()
    df = pd.DataFrame({'A':A,'B':B,'C':C,'D':D})
    df['CLASS'] = df['A'] + df['B']*df['C'] + df['A']*df['D']
    return df

df = gen_df()
df.head(10)
df['A'][0:100].plot(title='A', style='.')
def add_errors(df):
    # shuffle the first shuffle_feature rows for each feature 20%
    shuffle_features = int(df.shape[0] * 0.20)
    shuffle(df['A'][0:shuffle_features])
    shuffle(df['B'][0:shuffle_features])
    shuffle(df['C'][0:shuffle_features])
    shuffle(df['D'][0:shuffle_features])
    return df

df = add_errors(df)

def add_marker(df):
    # create a marker for tracking rows that no longer follows the non-linear relation
    df['GOOD'] = (df['A'] + df['B']*df['C'] + df['A']*df['D']) == df['CLASS'] 
    df['GOOD'].replace([True,False],[1,0],inplace = True)
    return df

df = add_marker(df)
df.head(10)    
df = df.sample(frac=1).reset_index(drop=True)
df['CLASS'][0:200].plot(title='CLASS',style='.')
print('Whole set true mean classification accuracy {0:2.4f}'.format(df['GOOD'].mean()))
from sklearn.model_selection import train_test_split

# reserving emulated unseen data. df2 is what's visible for the "data scientist"
df2, unseen = train_test_split(df, test_size=0.8)

# let's leave 10% data out for hold-out
train, hold_out = train_test_split(df2,test_size=0.10)
print(train.shape)
print(hold_out.shape)
print('hold out true mean classification accuracy {0:2.4f}'.format(hold_out['GOOD'].mean()))
print('unseen   true mean classification accuracy {0:2.4f}'.format(unseen['GOOD'].mean()))
print('train (cv) true mean classification accuracy {0:2.4f}'.format(train['GOOD'].mean()))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'minkowski')
kn_param_grid={'n_neighbors':[3,5,7,9,11], 'p':[1,1.5,2]}
gs = GridSearchCV(knclass, kn_param_grid, cv = 5, return_train_score = True, n_jobs=4)
gs.fit(X=np.array(train[['A','B','C','D']]), y = np.array(train['CLASS']))
gs.best_params_
unseen_data_score = gs.score(unseen[['A','B','C','D']],unseen['CLASS'])
print('accuracy of the model on unseen data {0:2.4f}'.format(unseen_data_score))
gs.cv_results_['mean_train_score']
# cross validation score
cv_best_score = gs.best_score_
gs.cv_results_['mean_test_score']
# hold out score
hold_out_score = gs.score(hold_out[['A','B','C','D']], hold_out['CLASS'])
print('Accuracy by model on hold-out {0:2.4f}; accuracy by model on unseen {1:2.4f}'.format(hold_out_score, unseen_data_score))
print('Accuracy by model on CV       {0:2.4f}; accuracy by model on unseen {1:2.4}'.format(cv_best_score, unseen_data_score))
def one_run():
    # create base data frame
    df = gen_df()
    df = add_errors(df)
    df = add_marker(df)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # make partitions
    df2,   unseen   = train_test_split(df, test_size= 0.8)
    train, hold_out = train_test_split(df2,test_size= 0.1)
    
    # grid search and fit
    gs.fit(X=np.array(train[['A','B','C','D']]), y = np.array(train['CLASS']))
    
    # compute hold out score
    hold_out_score = gs.score(hold_out[['A','B','C','D']], hold_out['CLASS'])
    
    # compute cv score
    cv_best_score = gs.best_score_
    
    # compute unseen score
    unseen_score = gs.score(unseen[['A','B','C','D']], unseen['CLASS'])
    
    return unseen_score, cv_best_score, hold_out_score

n_loops = 30
result = np.empty([n_loops,3],dtype=float)

for i in range(n_loops):
    result[i,:] = np.array(one_run())
    print('round {3:2}   unseen {0:2.4f}  CV {1:2.3f} hold-out {2:2.3f}'.format(result[i,0],result[i,1],result[i,2],i))
    
plt.plot(result[:,0]-result[:,1],label='unseen-cv')
plt.plot(result[:,0]-result[:,2],label='unseen-hold-out')
plt.xlabel('round')
plt.ylabel('delta')
plt.legend()
print('cv                      mean error:', np.mean(result[:,0]-result[:,1]))
print('cv        error standard deviation:', np.std(result[:,0]-result[:,1]))
print('hold out                mean error:', np.mean(result[:,0]-result[:,2]))
print('hold out  error standard deviation:', np.std(result[:,0]-result[:,2]))