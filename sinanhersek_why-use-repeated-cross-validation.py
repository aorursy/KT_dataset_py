import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import random
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

wine_dataset = pd.read_csv('../input/winequality-red.csv')

#form feature matrix and target variable vector
X = wine_dataset[['fixed acidity', 'volatile acidity', 'citric acid',
       'chlorides',  'total sulfur dioxide', 'density',
        'sulphates', 'alcohol']].values #feature matrix

y = (wine_dataset['quality']>5).values.astype(int) #target variable, considers wines with quality score>5 good wines


def perform_repeated_cv(X, y , model):
    #set random seed for repeartability
    random.seed(1)

    #set the number of repetitions
    n_reps = 50

    # perform repeated cross validation
    accuracy_scores = np.zeros(n_reps)
    precision_scores=  np.zeros(n_reps)
    recall_scores =  np.zeros(n_reps)

    for u in range(n_reps):

        #randomly shuffle the dataset
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices] #dataset has been randomly shuffled

        #initialize vector to keep predictions from all folds of the cross-validation
        y_predicted = np.zeros(y.shape)

        #perform 10-fold cross validation
        kf = KFold(n_splits=5 , random_state=142)
        for train, test in kf.split(X):

            #split the dataset into training and testing
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]

            #standardization
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #train model
            clf = model
            clf.fit(X_train, y_train)

            #make predictions on the testing set
            y_predicted[test] = clf.predict(X_test)

        #record scores
        accuracy_scores[u] = accuracy_score(y, y_predicted)
        precision_scores[u] = precision_score(y, y_predicted)
        recall_scores[u]  = recall_score(y, y_predicted)

    #return all scores
    return accuracy_scores, precision_scores, recall_scores


#perform repeted CV with logistic regression
accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y ,  RandomForestClassifier(n_estimators=100) )

#plot results from the 50 repetitions
fig, axes = plt.subplots(3, 1)

axes[0].plot(100*accuracy_scores , color = 'xkcd:cherry' , marker = 'o')
axes[0].set_xlabel('Repetition')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_facecolor((1,1,1))
axes[0].spines['left'].set_color('black')
axes[0].spines['right'].set_color('black')
axes[0].spines['top'].set_color('black')
axes[0].spines['bottom'].set_color('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].spines['right'].set_linewidth(0.5)
axes[0].spines['top'].set_linewidth(0.5)
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

axes[1].plot(100*precision_scores , color = 'xkcd:royal blue' , marker = 'o')
axes[1].set_xlabel('Repetition')
axes[1].set_ylabel('Precision(%)')
axes[1].set_facecolor((1,1,1))
axes[1].spines['left'].set_color('black')
axes[1].spines['right'].set_color('black')
axes[1].spines['top'].set_color('black')
axes[1].spines['bottom'].set_color('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].spines['right'].set_linewidth(0.5)
axes[1].spines['top'].set_linewidth(0.5)
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

axes[2].plot(100*precision_scores , color = 'xkcd:emerald' , marker = 'o')
axes[2].set_xlabel('Repetition')
axes[2].set_ylabel('Recall (%)')
axes[2].set_facecolor((1,1,1))
axes[2].spines['left'].set_color('black')
axes[2].spines['right'].set_color('black')
axes[2].spines['top'].set_color('black')
axes[2].spines['bottom'].set_color('black')
axes[2].spines['left'].set_linewidth(0.5)
axes[2].spines['right'].set_linewidth(0.5)
axes[2].spines['top'].set_linewidth(0.5)
axes[2].spines['bottom'].set_linewidth(0.5)
axes[2].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

plt.grid(True)
plt.tight_layout()

from sklearn.linear_model import LogisticRegression

#set up the parameter sweep
c_sweep =  np.power(10, np.linspace(-4,4,50))

#perform repeated cross-validation by sweeping the parameter
accuracy_parameter_sweep = [] # keep scores here
std_parameter_sweep = [] #keep parameters in here
for c in c_sweep:

    #perform repeated cross-validation
    accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y ,  LogisticRegression(C=c) )

    ##append scores
    accuracy_parameter_sweep.append(np.mean(100*accuracy_scores))
    std_parameter_sweep.append(np.std(100*accuracy_scores))


#plot C vs. accuracy
plt.fill_between(c_sweep , np.array(accuracy_parameter_sweep) - np.array(std_parameter_sweep) ,
                 np.array(accuracy_parameter_sweep) + np.array(std_parameter_sweep) , facecolor = 'xkcd:light pink', alpha=0.7)
plt.semilogx(c_sweep,accuracy_parameter_sweep , color= 'xkcd:red' , linewidth=4)
plt.xlabel('C')
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Accuracy vs. Hyper-parameter C')
plt.grid(True, which='both')
plt.tight_layout()