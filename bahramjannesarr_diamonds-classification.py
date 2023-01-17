import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler , MinMaxScaler

from sklearn.svm import SVC

from sklearn.metrics import classification_report , confusion_matrix , roc_curve , roc_auc_score 

from sklearn import metrics
diamonds = pd.read_csv('../input/diamonds/diamonds.csv' , index_col = 0)
diamonds.head()
diamonds.info()
clarity = pd.get_dummies(diamonds['clarity'])

color = pd.get_dummies(diamonds['color'])
diamonds = pd.concat([diamonds , clarity , color] , axis = 1)
diamonds.drop(['color' , 'clarity'] , axis = 1 , inplace=True)
diamonds.head()
diamonds.info()
plt.figure(figsize=(20, 20))

df_corr = diamonds.corr()

sns.heatmap(df_corr, cmap=sns.diverging_palette(220, 20, n=12), annot=True)

plt.title("Diamonds")

plt.show()
diamonds['cut'].value_counts()
for i in range(len(diamonds)):

    if diamonds['cut'].iloc[i] == 'Ideal':

        diamonds['cut'].iloc[i] = 1

    if diamonds['cut'].iloc[i] == 'Premium':

        diamonds['cut'].iloc[i] = 2

    if diamonds['cut'].iloc[i] == 'Very Good':

        diamonds['cut'].iloc[i] = 3

    if diamonds['cut'].iloc[i] == 'Good':

        diamonds['cut'].iloc[i] = 4

    if diamonds['cut'].iloc[i] == 'Fair':

        diamonds['cut'].iloc[i] = 5
X = diamonds.drop(['cut'], axis = 1 ).values

y = diamonds['cut'].values
scalar = StandardScaler()

X = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X , y  , test_size = 0.2 , random_state = 42)
y_train = y_train.astype('int')

y_test = y_test.astype('int')
def calculate_and_plot_k_neighbors(X_train, X_test, y_train, y_test):

    

    neighbors = np.arange(1, 10)

    train_accuracy = np.empty(len(neighbors))

    test_accuracy = np.empty(len(neighbors))

    

    for i, k in enumerate(neighbors):

        knn = KNeighborsClassifier(n_neighbors= k)

        knn.fit(X_train , y_train)

        train_accuracy[i] = knn.score(X_train, y_train)    

        test_accuracy[i] = knn.score(X_test, y_test)



    plt.figure(figsize=(10, 8))   

    plt.title('k in kNN analysis')

    plt.plot( neighbors , test_accuracy , label = 'Testing Accuracy')

    plt.plot(neighbors,train_accuracy ,label = 'Training Accuracy')

    plt.legend()

    plt.annotate('Best accuracy for this model with this k is {0:.2f} %'.format(max(test_accuracy) * 100), xy=(np.argmax(test_accuracy) + 1 , max(test_accuracy)), xytext=(5 , 0.80),

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="angle3,angleA=0,angleB=-90"));

    plt.xlabel('Number of Neighbors')

    plt.ylabel('Accuracy')

    plt.show()
def plot_confusion_matrix(cf_matrix , y_test , model_type , cf_size):

    if cf_size == '2x2':

        group_names = ['True Negative','False Positive','False Negative','True Positive']

        group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]

        labels = ['{}\n{}'.format(v1 ,v2) for v1, v2 in zip(group_names,group_counts)]

        labels = np.asarray(labels).reshape(2,2)

        plt.figure(figsize=(10, 8))

        sns.heatmap(

            cf_matrix,

            annot = labels,

            cmap=sns.cubehelix_palette(100, as_cmap=True, hue=1, dark=0.30),

            fmt='',

            linewidths=1.5,

            vmin=0,

            vmax=len(y_test),

        )

        plt.title(model_type)

        plt.show()

    else:

        plt.figure(figsize=(10, 8))

        sns.heatmap(

            cf_matrix / np.sum(cf_matrix) * 100,

            annot = True,

            cmap=sns.cubehelix_palette(100, as_cmap=True, hue=1, dark=0.30),

            fmt='.2f',

            linewidths=1.5,

            vmin=0,

            vmax=100,

        )

        plt.title(model_type)

        plt.show()
def kNN_algorithm(X_train , y_train , X_test , y_test , k):

    

    global y_pred_kNN

    global kNN_pipeline

    

    steps = [('impute' , SimpleImputer(missing_values = 0, strategy='mean')),

             ('sclaer', StandardScaler()),

             ('kNN', KNeighborsClassifier(n_neighbors = k))]

    

    kNN_pipeline = Pipeline(steps)

    

    kNN_pipeline.fit(X_train , y_train)

    

    y_pred_kNN = kNN_pipeline.predict(X_test)

    

    print(classification_report(y_test , y_pred_kNN))

    print('kNN algorithm acuracy is : {0:.2f} %'.format(kNN_pipeline.score(X_test , y_test) * 100))
def SVM_algorithm(X_train, X_test, y_train, y_test):

    

    global y_pred_SVM

    global SVM_pipeline

    global y_prob_SVM

    

    steps = [('scaler', StandardScaler()),

             ('SVM', SVC(probability=True))]

    

    SVM_pipeline = Pipeline(steps)

    

    parameters = {'SVM__C':[1, 10, 100 ],

                  'SVM__gamma':[0.1, 0.01]}

    

    cv = GridSearchCV(SVM_pipeline , cv = 5 , param_grid = parameters)

    

    cv.fit(X_train , y_train)

    

    y_pred_SVM = cv.predict(X_test)

    

    y_prob_SVM = cv.predict_proba(X_test)

    

    print("Accuracy: {0:.2f} %".format(cv.score(X_test, y_test) * 100))

    print(classification_report(y_test, y_pred_SVM))

    print("Tuned Model Parameters: {}".format(cv.best_params_))
def LogisticRegression_algorithm(X_train, X_test, y_train, y_test):

    

    global y_pred_LG

    global LG_pipeline

    global y_prob_LG

    

    steps = [('scaler', StandardScaler()),

             ('LogisticRegression', LogisticRegression(random_state = 0))]

    

    LG_pipeline = Pipeline(steps)



    

    LG_pipeline.fit(X_train , y_train)

    

    y_pred_LG = LG_pipeline.predict(X_test)

    

    y_prob_LG = LG_pipeline.predict_proba(X_test)

    

    print("Accuracy: {0:.2f} %".format(LG_pipeline.score(X_test, y_test) * 100))

    print(classification_report(y_test, y_pred_LG))
calculate_and_plot_k_neighbors(X_train, X_test, y_train, y_test)
kNN_algorithm(X_train , y_train , X_test , y_test , 8)
cf_matrix_knn = confusion_matrix(y_test, y_pred_kNN)

cf_matrix_knn = pd.DataFrame(cf_matrix_knn  , index = ['Ideal' ,'Premium','Very Good','Good','Fair'] , columns =['Ideal','Premium','Very Good','Good','Fair'])

plot_confusion_matrix(cf_matrix_knn , y_test , 'kNN Confusion Matrix in percent %' , '5x5')
SVM_algorithm(X_train, X_test, y_train, y_test)
cf_matrix_SVM = confusion_matrix(y_test, y_pred_SVM)

cf_matrix_SVM = pd.DataFrame(cf_matrix_SVM  , index = ['Ideal' ,'Premium','Very Good','Good','Fair'] , columns =['Ideal','Premium','Very Good','Good','Fair'])

plot_confusion_matrix(cf_matrix_SVM , y_test , 'SVM Confusion Matrix in percent' , '5x5')
LogisticRegression_algorithm(X_train, X_test, y_train, y_test)
cf_matrix_LG = confusion_matrix(y_test, y_pred_LG)

cf_matrix_LG = pd.DataFrame(cf_matrix_LG  , index = ['Ideal' ,'Premium','Very Good','Good','Fair'] , columns =['Ideal','Premium','Very Good','Good','Fair'])

plot_confusion_matrix(cf_matrix_LG , y_test , 'LogisticRegression Confusion Matrix in percent' , '5x5')