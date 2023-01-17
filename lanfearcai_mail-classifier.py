

"""

This script can be used as skelton code to read the challenge train and test

csvs, to train a trivial model, and write data to the submission file.

"""

import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict



#%%



from sklearn.metrics import accuracy_score



## Read csvs

train_df = pd.read_csv('train.csv', index_col=0)

test_df = pd.read_csv('test.csv', index_col=0)



#%%



## Handle missing values

train_df.fillna('NA', inplace=True)

test_df.fillna('NA', inplace=True)



#%%



## Filtering column "mail_type"

train_x = train_df[['org','tld','mail_type']]

train_y = train_df[['label']]



test_x = test_df[['org','tld','mail_type']]



#sns.distplot(int(train_df['mail_type']), kde=False)





#%%



## PCA

data_train = np.array(train_df.iloc[:, [3,4,6,7,8,9,10,11]])

data_test = np.array(test_df.iloc[:, [3,4,6,7,8,9,10,11]])

print(len(data_test))

data = np.vstack((data_train, data_test))#0:10744是test



M = np.mean(data, 0) # compute the mean

Var = np.var(data,0)

# C = data-M

C = (data - M)*1/Var

W = np.dot(C.T, C) # compute covariance matrix

eigval, eigvec = np.linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix

idx = eigval.argsort()[::-1] # Sort eigenvalues

eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues

print(eigval/sum(eigval))

newData2 = np.dot(C,np.real(eigvec[:,:2])) # Project the data to the new space (2-D)

newData3 = np.dot(C,np.real(eigvec[:,:5])) # Project the data to the new space (3-D)



newData2_train = newData2[0:25066]

newData2_test = newData2[25066:35811]



newData3_train = newData3[0:25066]

newData3_test = newData3[25066:35811]





print(newData3_train)

len(newData2_train)

print(len(newData3_test))

# newData3 = np.dot(C,np.real(eigvec[:,:3])) # Project the data to the new space (3-D)









#%%



## Do one hot encoding of categorical feature

feat_enc = OneHotEncoder()

feat_enc.fit(np.vstack((train_x, test_x)))

train_x_featurized = feat_enc.transform(train_x)

test_x_featurized = feat_enc.transform(test_x)



train_type_array = train_x_featurized.A

test_type_array = test_x_featurized.A



data_type = np.vstack((train_type_array, test_type_array))



M0 = np.mean(data_type, 0) # compute the mean

Var0 = np.var(data_type,0)

C0 = (data_type - M0)*1/Var0



train_type_array = C0[0:25066]

test_type_array = C0[25066:35811]





newData3_train_fin = np.hstack((newData3_train, train_type_array))

newData3_test_fin = np.hstack((newData3_test, test_type_array))

print(newData3_test_fin)

newData3_train_fin

type(test_type_array)

#type(train_type_array = train_x_featurized.A)



#%%



## Train a simple KNN classifier using featurized data

neigh = KNeighborsClassifier(n_neighbors=150)

neigh.fit(newData3_train, train_y)

pred_y = neigh.predict(newData3_test)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("knn_submission.csv", index=True, index_label='Id')





#%%



##Train a SVM classifier

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(newData2_train, train_y)

pred_y = clf.predict(newData2_test)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("svm_submission.csv", index=True, index_label='Id')





#%%



##Train a NNC classifier

clf = MLPClassifier(batch_size=80)

clf.fit(newData3_train_fin, train_y)



pred_y = clf.predict(newData3_test_fin)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("nn_submission.csv", index=True, index_label='Id')





#%%



clf_rdf = RandomForestClassifier()

clf_rdf.fit(newData3_train_fin,train_y)



pred_y = clf_rdf.predict(newData3_test_fin)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("rdf_submission.csv", index=True, index_label='Id')



#%%



clf_GB = GradientBoostingClassifier()

clf_GB.fit(newData3_train_fin,train_y)



pred_y = clf_GB.predict(newData3_test_fin)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("GB_submission.csv", index=True, index_label='Id')





#%%



batch_size = [10, 20, 40, 60, 80, 100, 120, 140] 

epochs = [10, 50, 100, 150, 200] 

param_grid = dict(batch_size=batch_size) 



grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1) 



grid_result = grid.fit(newData3_train_fin, train_y) 



# summarize results 



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 



for params, mean_score, scores in grid_result.grid_scores_: 



    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))



pred_y = grid_result.predict(newData3_test_fin)



pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("nn_GS_submission.csv", index=True, index_label='Id')



#%%



## Save results to submission file

pred_df = pd.DataFrame(pred_y, columns=['label'])

pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')


