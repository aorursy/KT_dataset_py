import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, ShuffleSplit

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, confusion_matrix

from astropy.table import Table, Column

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.base import clone

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/mushrooms.csv")

print(df.shape)

df.head()
df['class'].value_counts()
df.isnull().sum().sum()
df_dum = pd.DataFrame()

for col in df.columns:

    dum = pd.get_dummies(df[col])

    for dcol in dum.columns:

        name = col +"_"+ dcol

        df_dum[name] = dum[dcol]

print(df_dum.shape)
df_dum.head()
df_int = pd.DataFrame()

le = LabelEncoder()

for col in df.columns:

    df_int[col] = le.fit_transform( df[col])

df_int.head()
X = df.iloc[:,1:]

y = df.iloc[:,0]

Xd = df_dum.iloc[:,2:]

yd = df_dum.iloc[:,0:2]

Xi = df_int.iloc[:,1:]

yi = df_int.iloc[:,0]

scalerd = StandardScaler()

Xds = scalerd.fit_transform(Xd)

scaleri = StandardScaler()

Xis = scaleri.fit_transform(Xi)
def d_method( X, y, model, random_state = 0, k = 5 ):

    # Fits a categorical model and outputs a cross-validation result of:

    # Accuracy, Recall, Precision, and the model thats fit last.

    # The data is train/test split and shuffled systematically

    kf = ShuffleSplit( n_splits = k )

    ac = np.zeros( k ); re = np.zeros( k ); pr = np.zeros( k )

    i = 0

    for train_index, test_index in kf.split(X):

        t_model = clone(model)

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        t_model.fit( X_train, y_train )

        y_pred = t_model.predict( X_test )

        ac[i] = accuracy_score( y_test, y_pred )

        re[i] = recall_score( y_test, y_pred )

        pr[i] = precision_score( y_test, y_pred )

        i = i+1

    return( ac, re, pr, t_model )
def d_conf( X_test, y_test, t_model, p=0.5 ):

    # Creates a confusion matrix

    # Using a test set of data and a trained model

    ypro = t_model.predict_proba(X_test)

    yp = ypro[:,1] >= p

    cm = confusion_matrix(y_test, yp)

    return(cm)
def d_conf_l( X_test, y_test, t_model):

    # The same as d_conf only removing the theshold of class selection

    yp = t_model.predict(X_test)

    cm = confusion_matrix(y_test, yp)

    return(cm)
def d_roc( X_test, y_test, t_model, points=100):

    # Generates a plot that describes the changes in

    #   Accuracy, recall, precision scores as the threshold

    #   For classification changes.

    #   Using a test set of data and a trained model

    ac_p=np.zeros(points); re_p=np.zeros(points); pr_p=np.zeros(points); 

    for i in np.arange(points):

        ypro = t_model.predict_proba(X_test)

        yp = ypro[:,1] >= (i/points)

        ac_p[i] = accuracy_score( y_test, yp )

        re_p[i] = recall_score( y_test, yp )

        pr_p[i] = precision_score( y_test, yp )

    t = np.arange(points)/points

    plt.plot(t, ac_p, label='accuracy score')

    plt.plot(t, re_p, label = 'recall score')

    plt.plot(t, pr_p, label = 'precision score')

    plt.legend()

    plt.show()
def d_summary(X, y, model, random_state = 0,k=5,X_test=None,y_test=None,p=0.5,points=100):

    # Runs the d_method, d_conf, and d_roc on a model and dataset.

    ac, re, pr, t_model = d_method( X, y, model )

    print( "Accuracy Score =  ", ac, " Mean = ", ac.mean() )

    print( "Recall Score =    ", re, " Mean = ", re.mean() )

    print( "Precision Score = ", pr, " Mean = ", pr.mean() )

    if X_test is None:

        X_test = X

    if y_test is None:

        y_test = y

    cm = d_conf( X_test, y_test, t_model, p)

    print("Confusion Matrix:")

    print(cm)

    d_roc( X_test, y_test, t_model, points)
d_summary( Xds, yi, LogisticRegression() )
d_summary(Xis, yi, LogisticRegression() )
X_train, X_test, y_train, y_test = train_test_split(Xis, yi, test_size=0.2)

d_summary( Xis, yi, LogisticRegression(),X_test= X_test,y_test=y_test)
X_train, X_test, y_train, y_test = train_test_split(Xds, yi, test_size=0.2)

d_summary( Xds, yi, LogisticRegression(),X_test= X_test,y_test=y_test)
def d_table( X, y, model, A, parameter):

    # Returns accuracy scores for models performed with specfied hyper-parameters.

    #   Take care to describe the parameter as a string

    print( 'alpha\t\tAccuracy\tRecall\t\tPrecision')

    for alpha in A:

        l_model = clone(model)

        eval("l_model.set_params(" + parameter + "=" + str(alpha) + ")")

        ac, re, pr, t_model = d_method( Xds, yi, l_model )

        print( alpha,'\t\t%0.4f\t\t%0.4f\t\t%0.4f' % (ac.mean(), re.mean(), pr.mean() ) )
reg_strength = [0.01, 0.05, 0.1, 0.5,1,5, 10,50,100]

print('Indicator Variables')

d_table( Xds, yi, LogisticRegression(penalty="l1"), reg_strength, 'C' )

print('Integer Variables')

d_table( Xis, yi, LogisticRegression(penalty="l1"), reg_strength, 'C' )
print("Indicator / Dummy")

d_summary( Xds, yi, DecisionTreeClassifier())

print("Integer / LabelEncoder")

d_summary( Xis, yi, DecisionTreeClassifier())
min_samples = np.arange(10) + 3

print("Indicator / Dummy")

d_table( Xds, yi, DecisionTreeClassifier(), min_samples, 'min_samples_split')

print("Integer / LabelEncoder")

d_table( Xis, yi, DecisionTreeClassifier(), min_samples, 'min_samples_split')
print( "Dummy / Indicator")

d_summary( Xds, yi, RandomForestClassifier())

print( "LabelEncoder / Integer")

d_summary( Xis, yi, RandomForestClassifier())
max_features = np.arange(10) + 3

print("Indicator / Dummy")

d_table( Xds, yi, DecisionTreeClassifier(), max_features, 'max_features')

print("Integer / LabelEncoder")

d_table( Xis, yi, DecisionTreeClassifier(), max_features, 'max_features')
print( "Dummy / Indicator")

ac, re, pr, t_model = d_method( Xds, yi, SVC())

cm = d_conf_l(Xds, yi, t_model)

print( "Accuracy Score =  ", ac, " Mean = ", ac.mean() )

print( "Recall Score =    ", re, " Mean = ", re.mean() )

print( "Precision Score = ", pr, " Mean = ", pr.mean() )

print(cm)

print( "LabelEncoder / Integer")

ac, re, pr, t_model = d_method( Xis, yi, SVC())

cm = d_conf_l(Xis, yi, t_model)

print( "Accuracy Score =  ", ac, " Mean = ", ac.mean() )

print( "Recall Score =    ", re, " Mean = ", re.mean() )

print( "Precision Score = ", pr, " Mean = ", pr.mean() )

print(cm)
print( "Dummy / Indicator")

ac, re, pr, t_model = d_method( Xds, yi, KNeighborsClassifier())

cm = d_conf_l(Xds, yi, t_model)

print( "Accuracy Score =  ", ac, " Mean = ", ac.mean() )

print( "Recall Score =    ", re, " Mean = ", re.mean() )

print( "Precision Score = ", pr, " Mean = ", pr.mean() )

print(cm)

print( "LabelEncoder / Integer")

ac, re, pr, t_model = d_method( Xis, yi, KNeighborsClassifier())

cm = d_conf_l(Xis, yi, t_model)

print( "Accuracy Score =  ", ac, " Mean = ", ac.mean() )

print( "Recall Score =    ", re, " Mean = ", re.mean() )

print( "Precision Score = ", pr, " Mean = ", pr.mean() )

print(cm)
print( "Dummy / Indicator")

d_summary( Xds, yi, GaussianNB())

print( "LabelEncoder / Integer")

d_summary( Xis, yi, GaussianNB())