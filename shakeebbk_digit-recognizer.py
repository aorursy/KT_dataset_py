import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
# configuration

SVM_kernel = 'rbf'

cache_size = 8000

svm_params_c = [0.001, 0.1, 100, 10e5]

svm_params_gamma = [10,1,0.1,0.01]

cv_folds = 5

grid_cv_samples = 4000



model_from_files = False



grid_model_file = f'grid_{SVM_kernel}.pkl'

trained_model_file = f'model_{SVM_kernel}.pkl'



submission_file = f'submission_{SVM_kernel}.csv'
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

print(train_df.shape)

test_df = pd.read_csv('../input/digit-recognizer/test.csv')

print(test_df.shape)
train_df.columns[0:10]
# lets checkout some images

plt.figure(figsize=(12, 10))

for i in range(1, 5):

    plt.subplot(1, 4, i)

    plt.imshow(train_df.iloc[i-1, 1:].values.reshape(28, 28), cmap='gray')

    plt.title(f'plot for {train_df.iloc[i-1, 0]}')

plt.show()
sns.countplot(train_df['label'])

plt.show()
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV



from sklearn.decomposition import PCA



import time

import pickle
if not model_from_files:

    steps = [('scaler', MinMaxScaler()), ('PCA', PCA()), ('SVM', SVC(kernel=SVM_kernel, cache_size=cache_size))]

    pipeline = Pipeline(steps)



    params = {'PCA__n_components': [10, 20, 25, 30, 50],

              'SVM__C':svm_params_c, 'SVM__gamma':svm_params_gamma}

    grid = GridSearchCV(pipeline,

                        param_grid = params,

                        cv = cv_folds)



    t_start = time.time()



    grid.fit(train_df.iloc[:grid_cv_samples, 1:], train_df.iloc[:grid_cv_samples, 0])



    print("Elapsed Time", time.time()-t_start)



    # without PCA - for 4000 samples, time = 1276 secs

    # save the model

    # pickle.dump(grid, open(grid_model_file, 'wb'))
if model_from_files:

    grid = pickle.load(open(grid_model_file, 'rb'))
print(f"Best score = {grid.best_score_}")

print(f"Best params = {grid.best_params_}")
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
if not model_from_files:

    steps = [('scaler', MinMaxScaler()), 

             ('PCA', PCA(n_components=grid.best_params_['PCA__n_components'])), 

             ('SVM', SVC(kernel=SVM_kernel, cache_size=cache_size, 

                                                     gamma=grid.best_params_['SVM__gamma'], 

                                                     C=grid.best_params_['SVM__C']))]

    pipeline = Pipeline(steps)



    t_start = time.time()



    # train on entire train data set

    pipeline.fit(train_df.iloc[:, 1:], train_df.iloc[:, 0])



    print("Elapsed Time", time.time()-t_start)



    # without PCA time - 205 secs

    

    # save the model

    pickle.dump(pipeline, open(trained_model_file, 'wb'))
if model_from_files:

    pipeline = pickle.load(open(trained_model_file, 'rb'))
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
submission = pd.DataFrame()

submission['ImageId'] = test_df.index + 1

submission.head()
submission['Label'] = pipeline.predict(test_df)
# submission.to_csv(submission_file, index=False)

submission.to_csv('submission_pca_svm', index=False)