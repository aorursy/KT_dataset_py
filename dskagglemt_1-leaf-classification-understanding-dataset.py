import pandas as pd

import numpy as np
df = pd.read_csv('../input/leaf-classification/train.csv.zip')
df.head()
df.shape
df.info()
df.describe()
# types

df.dtypes
set((df.dtypes).to_list())
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



# Image manipulation.

import PIL.Image

from IPython.display import display
len(df.id)
# from zipfile import ZipFile

# image_dir = '../input/leaf-classification/images.zip'

# image_folder = ZipFile(image_dir, 'r')

# image_folder.namelist()[0:5]
# image_folder.namelist()[1:2]
# image_folder
# # importing required modules 

# from zipfile import ZipFile 

  

# # specifying the zip file name 

# images_zip = '../input/leaf-classification/images.zip'

  

# # opening the zip file in READ mode 

# with ZipFile(images_zip, 'r') as zip: 

#     # printing all the contents of the zip file 

# #     zip.printdir() 

  

#     # extracting all the files 

#     print('Extracting all the files now...') 

#     zip.extractall() 

#     print('Done!') 
image_dir = './images'
# img = image_dir + '/' + str(100) + '.jpg'

# img
# img = mpimg.imread(img)

# imgplot = plt.imshow(img)

# plt.show()



# img_lbl = df['species'].loc[100]

# print(img_lbl)
# type(imgplot)

# # imgplot.shape



# type(img), img.shape
# img = img.resize((160, 240), mpimg.ANTIALIAS)



# type(img), img.shape
# randTrainInd = np.random.randint(len(df.id))

# print(randTrainInd)



# randomID = df.iloc[randTrainInd].id

# print(randomID)



# # df.loc[randTrainInd]

# df[df['id']==randomID]['species']

# # df.loc[(df[id] == randTrainInd)]



# # df[df['id']==randomID]['species']
# # show some random images

# plt.figure(figsize=(12,12))



# for k in range(28):

#     randTrainInd = np.random.randint(len(df.id))

    

#     randomID = df.iloc[randTrainInd].id

    

#     imageFilename = image_dir + '/' + str(randomID) + '.jpg' 

    

#     plt.subplot(4,7,k+1); 

    

#     plt.imshow(mpimg.imread(imageFilename), cmap='gray')

    

# #     plt.title(df['species'].loc[randTrainInd], fontsize=8); 

#     plt.title(df['species'].loc[randTrainInd] + '; ' + str(randomID), fontsize=8); 

#     plt.axis('off')
# X = df.drop(['id', 'species'], axis=1).values

# X
X = df.drop(['id', 'species'], axis=1)

y = df['species']



print(X.shape, y.shape)
from sklearn.preprocessing import LabelEncoder
# df['species'].value_counts()

df['species'].unique()
classEncoder = LabelEncoder()

trainLabels = classEncoder.fit_transform(df.loc[:,'species'])
trainLabels[:5] 
classEncoder.classes_[trainLabels[:5]]
y.head()
y = classEncoder.fit_transform(y)

y[:5]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled[:5]
# Split the data into training and validation

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 99)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 99)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
# Lets define the parameters for GridSearchCV.

params = {'C':[1, 10, 50, 100, 500, 1000, 2000],                   # Inverse of regularization strength; must be a positive float.

          'tol': [0.001, 0.0001, 0.005]                           # olerance for stopping criteria.

#           'penalty' : ['l1', 'l2', 'elasticnet', 'none'] ,          # Used to specify the norm used in the penalization.

#           'solver' : ['lbfgs','newton-cg','sag','saga']            # Algorithm to use in the optimization problem.

         }
# Initiate the Logistic Regression Model.

lr = LogisticRegression(solver='newton-cg', multi_class='multinomial')



# lr = LogisticRegression(multi_class='multinomial')



# Here we are taking solver as `newton-cg` we can also have other solvers such as `lbfgs`.
# For evluation we have to use log_loss, and for this we have to make a callable from GridSearchCV

from sklearn.metrics import log_loss, make_scorer

LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
# lr1 = LogisticRegression(multi_class='multinomial')
clf = GridSearchCV(lr, params, scoring = LogLoss, refit = 'True', n_jobs = 1, cv = 5)

# Refer https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html to get more details.
lr.fit(X_train, y_train)
clf_fit = clf.fit(X_scaled, y)
print('Logistic Regression Score {}'.format( lr.score(X_test, y_test)))
print('Grid Search Score {}'.format( clf_fit.score(X_scaled, y)))
print("best params: " + str(clf.best_params_))



print("best estimator: " + str(clf.best_estimator_))



clf.cv_results_['mean_test_score']



# for i in ['mean_test_score', 'std_test_score', 'param_n_estimators']:

#     print(i," : ",grid.cv_results_[i])

        

# for params, mean_score, scores in clf.cv_results_:

#     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

#     print(scores)
test = pd.read_csv('../input/leaf-classification/test.csv.zip')



test_ids = test.pop('id')



x_test = test.values



scaler = StandardScaler()



x_test = scaler.fit_transform(x_test)



y_pred = clf.predict_proba(x_test)
y_pred
sample_sub = pd.read_csv('../input/leaf-classification/sample_submission.csv.zip', nrows = 5)

sample_sub
subm = pd.DataFrame(y_pred, index=test_ids, columns = classEncoder.classes_)

subm.to_csv('Submission_LogisticReg_v1.csv')
subm.head()