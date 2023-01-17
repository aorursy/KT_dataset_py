# load the packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



%pylab inline
# load data



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

target = train["label"]

train = train.drop("label",1)
train.shape
# plot some of the numbers



figure(figsize(5,5))

for digit_num in range(0,64):

    subplot(8,8,digit_num+1)

    grid_data = train.iloc[digit_num].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

    xticks([])

    yticks([])
# train the model by applying different n_emulators value



def trainByN_emulator():

    # loading training data

    print('Loading training data')

    X_tr = train

    y_tr = target



    scores = list()

    scores_std = list()



    print('Start learning...')

    n_trees = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 250, 300]

    for n_tree in n_trees:

        print(n_tree)

        recognizer = RandomForestClassifier(n_tree)

        score = cross_val_score(recognizer, X_tr, y_tr)

        scores.append(np.mean(score))

        scores_std.append(np.std(score))

    sc_array = np.array(scores)

    std_array = np.array(scores_std)

    print('Score: ', sc_array)

    print('Std  : ', std_array)



    plt.plot(n_trees, scores)

    plt.ylabel('CV score')

    plt.xlabel('number of trees')

    plt.savefig('cv_trees.png')

trainByN_emulator()
# The performance of model with default parameters



rf0 = RandomForestClassifier(oob_score=True, random_state=10)

rf0.fit(train,target)

score = cross_val_score(rf0, train,target)
np.mean(score)
# Grid search on n_estimators in range(200, 221, 5), which is from experiment above



param_test1 = {'n_estimators':range(200,221,5)}

gsearch1 = GridSearchCV(estimator = RandomForestClassifier(oob_score=True ,random_state=10), 

                       param_grid = param_test1,cv=5)

gsearch1.fit(train,target)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
# Grid search on max_samples_leaf and max_samples_split in range(1,5,2) and range(2, 10, 2)



param_test2 = dict(    

    min_samples_split = [n for n in range(2, 10, 2)], 

    min_samples_leaf = [n for n in range(1, 5, 2)], 

)

gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=210,random_state=10, oob_score=True), 

                       param_grid = param_test2,cv=5)

gsearch2.fit(train,target)

gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
# Grid search on max_features in range(20,81,10)



param_test3 = {'max_features':range(20,81,10)}

gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=210,random_state=10,min_samples_leaf = 1, 

                                                           min_samples_split = 2, oob_score=True),param_grid = param_test3,cv=5)

gsearch3.fit(train,target)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
# Grid search on max_depth in range(13,54,10)



param_test4 = {'max_depth':range(13,54,10)}

gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=210, max_features = 30,random_state=10, 

                                                           min_samples_leaf = 1, min_samples_split = 2, oob_score=True),param_grid = param_test4,cv=5)

gsearch4.fit(train,target)

gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_