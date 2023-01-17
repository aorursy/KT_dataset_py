# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

ONE_HOT_ENCODING_OUTPUT=True
CROSS_VALIDATION=True
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('../input/train.csv')
#print(data.head)


#######################################################################
 #               DATA PREPROCESSING
#######################################################################
y = data.label
X = data.drop(['label'], axis=1)

#Save columns in case of post-processing. 
columns=X.columns
#print(columns)

#Are there missing values?
#print(X.info())
#print(X.describe())

#Scaling the value of the pixel using sklearn StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled=pd.DataFrame(data=scaler.transform(X), columns=columns)

#The output of the model is a number 0-9. To avoid that the model learns that 6 is after 5, then the value is encoded as an array. 
#The array of zeros has only a 1 in the position of the corrispondent value. 
#E.g. 3=[0,0,0,1,0,0,0,0,0,0]
if ONE_HOT_ENCODING_OUTPUT==True:
    y=pd.get_dummies(y)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

hidden_layers=[(512),(512,256,64,32)]
MAX_ITER=2

PATH = 'mlp_model.pkl'

if os.path.exists(PATH):
    print('Loading model from file.')
    clf = joblib.load(PATH).best_estimator_
else:

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
    
    del data
    del X
    del y
    del X_scaled

    #Divide the training set in train and dev set. 
    #The model can use train_test_split or the crossing validation, according to the initial setting.
    if CROSS_VALIDATION==False:
        f, axarr = plt.subplots(len(hidden_layers))
       
        for par in range(len(hidden_layers)):

            print('-------------------------------------------------------------------')
            print('-------------------------------------------------------------------')
            print("Starting the training...")
            print("Training with layout:",hidden_layers[par])

            clf = MLPClassifier(solver='sgd', alpha=0, learning_rate_init=1e-1, learning_rate='adaptive', hidden_layer_sizes=hidden_layers[par], random_state=1,  verbose=True, max_iter=MAX_ITER, activation='relu')
            #solver='lbfgs' has no attribute n_iter e cost_
            clf.fit(X_train, y_train)

            print('Loss:',clf.loss_ )
            y_predict=clf.predict(X_test)

            print('Score:',accuracy_score(y_test, y_predict))
            print('-------------------------------------------------------------------')

            #Draw the loss curve        
            axarr[par].plot(range(1,clf.n_iter_+1),clf.loss_curve_)
            axarr[par].set_title('Hidden Layer Layout:'+str(hidden_layers[par]))      

        

    else:
        
        print('-------------------------------------------------------------------')
        print('------------------CROSS VALIDATION----------------------')
        print("Starting the training...")
        params = {'hidden_layer_sizes': hidden_layers}
        mlp = MLPClassifier(solver='sgd', verbose=10, learning_rate='adaptive', max_iter=MAX_ITER)
        
        #scores = cross_val_score(clf, X_scaled, y, cv=5)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)
        
        print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
        print('Best params appeared to be', clf.best_params_)
        #To save permanently the classificator in the model.pkl
        #joblib.dump(clf, PATH)
        clf = clf.best_estimator_

        print('Test accuracy:', clf.score(X_test, y_test))
     
    
    plt.show()

X_submission = pd.read_csv('../input/test.csv')
scaler.transform(X_submission)

y_predict_submission=clf.predict(X_submission)
print(y_predict_submission)
#print(len(y_predict_submission))

if ONE_HOT_ENCODING_OUTPUT==True:
    y_predict_val=np.zeros((len(y_predict_submission)), dtype=int)

    for i, item1 in enumerate(y_predict_submission):
        for index, item in enumerate(y_predict_submission[i]):
            if item ==1:
               y_predict_val[i]=index
else:
    y_predict_val=y_predict_submission

    
#print(y_predict_val)  

my_submission = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': y_predict_val})
print(my_submission.head())
my_submission.to_csv('submission.csv', index=False)