import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.preprocessing import scale, OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.utils import shuffle
df = pd.read_csv("../input/HR_comma_sep.csv")

print( df.isnull().any() )

df.head()
print( df.left.value_counts() )

print( df.sales.value_counts() )

print( df.salary.value_counts() )

df.describe()
X = df.drop('left', axis=1)

y = df['left']

print( X.shape )

print( y.shape )
X = X.rename(columns={'sales': 'sales1'})

X_sales = pd.get_dummies(X['sales1'])

X_salary = pd.get_dummies( X['salary'] )

Xd = pd.concat( [X, X_sales, X_salary], axis=1 )

Xd = Xd.drop( ['sales1','salary'], axis=1)

print( Xd.shape )

Xd.head()
sc = StandardScaler().fit(Xd)

Xs = sc.transform(Xd)

Xs, y = shuffle( Xs, y, random_state = 10 )
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
ROS = RandomOverSampler()

Xo1, yo1 = ROS.fit_sample(Xs, y)

Xo1, yo1 = shuffle( Xo1, yo1 )

print( Xo1.shape )

print( yo1.shape )

print( yo1.mean() )
SMO = SMOTE()

Xo2, yo2 = SMO.fit_sample(Xs, y)

Xo2, yo2 = shuffle( Xo2, yo2 )

print( Xo2.shape )

print( yo2.shape )

print( yo2.mean() )
ADA = ADASYN()

Xo3, yo3 = ADA.fit_sample(Xs, y)

Xo3, yo3 = shuffle( Xo3, yo3 )

print( Xo3.shape )

print( yo3.shape )

print( yo3.mean() )
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score

import xgboost as xgb
# Xo1_train, Xo1_test, yo1_train, yo1_test = train_test_split( Xo1, yo1, test_size = 0.3 )

kf = KFold( n_splits = 5 )

ac = np.zeros( 5 ); re = np.zeros( 5 ); pr = np.zeros( 5 )

Xo1, yo1 = shuffle( Xo1, yo1, random_state = 0 )

i = 0

for train_index, test_index in kf.split(Xo1):

    Xo1_train, Xo1_test = Xo1[train_index], Xo1[test_index]

    yo1_train, yo1_test = yo1[train_index], yo1[test_index]

    LgR = LogisticRegression()

    LgR.fit( Xo1_train, yo1_train )

    yo1_pred = LgR.predict( Xo1_test )

    ac[i] = accuracy_score( yo1_test, yo1_pred )

    re[i] = recall_score( yo1_test, yo1_pred )

    pr[i] = precision_score( yo1_test, yo1_pred )

    i = i+1
def d_method( X, y, model, random_state = 0 ):

    kf = KFold( n_splits = 5 )

    ac = np.zeros( 5 ); re = np.zeros( 5 ); pr = np.zeros( 5 )

    X, y = shuffle( X, y, random_state = random_state )

    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model.fit( X_train, y_train )

        y_pred = model.predict( X_test )

        ac[i] = accuracy_score( y_test, y_pred )

        re[i] = recall_score( y_test, y_pred )

        pr[i] = precision_score( y_test, y_pred )

        i = i+1

    return( ac.mean(), re.mean(), pr.mean(), model )
ac, re, pr, model = d_method( Xo1, yo1, LogisticRegression())

print( "Accuracy Score =  ", ac )

print( "Recall Score =    ", re )

print( "Precision Score = ", pr )
from astropy.table import Table, Column
t = Table( names = ('C','Accuracy','Recall','Precision') )

reg_strength = [0.01, 0.1, 1, 10, 100]

for c in reg_strength:

    ac, re, pr, model = d_method( Xo1, yo1, LogisticRegression(penalty = 'l1', C=c) )

    t.add_row( (c, ac, re, pr) )

t
t = Table( names = ('C','Accuracy','Recall','Precision') )

reg_strength = [0.01, 0.1, 1, 10, 100]

for c in reg_strength:

    ac, re, pr, m = d_method( Xo2, yo2, LogisticRegression(penalty = 'l1', C=c) )

    t.add_row( (c, ac, re, pr) )

t
t = Table( names = ('C','Accuracy','Recall','Precision') )

reg_strength = [0.01, 0.1, 1, 10, 100]

for c in reg_strength:

    ac, re, pr, model = d_method( Xo3, yo3, LogisticRegression(penalty = 'l1', C=c) )

    t.add_row( (c, ac, re, pr) )

t
t = Table( names = ('Accuracy','Recall','Precision') )

ac, re, pr, model = d_method( Xo1, yo1, DecisionTreeClassifier() )

t.add_row( (ac, re, pr) )

t
t = Table( names = ('Accuracy','Recall','Precision') )

ac, re, pr, model = d_method( Xo1, yo1,RandomForestClassifier() )

t.add_row( (ac, re, pr) )

t
fi = pd.DataFrame()

fi['features'] = Xd.columns.values

fi['importance'] = model.feature_importances_

fi
Xc = df.drop('left', axis=1)

yc = df['left']

Xc['sales'] = Xc['sales'].astype('category').cat.codes

Xc['salary'] = Xc['salary'].astype('category').cat.codes

Xc = np.array(Xc)

yc = np.array(yc)

print(X.shape)

print(y.shape)
t = Table( names = ('Accuracy','Recall','Precision') )

ac, re, pr, model = d_method( Xc, yc, DecisionTreeClassifier(class_weight = "balanced") )

t.add_row( (ac, re, pr) )

t
fi = pd.DataFrame()

fi['features'] = X.columns.values

fi['importance'] = model.feature_importances_

fi
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.plotly as py

import plotly.graph_objs as go
df = pd.read_csv("../input/HR_comma_sep.csv")

df.head()
X = df.drop('left',axis = 1 )

y = df['left']

print(X.shape)

print(y.shape)
def d_hist(X,y,look,n_bins):

    X0 = X.loc[ y == 0 ]

    X1 = X.loc[ y == 1 ]

    d0 = X0[look]

    d1 = X1[look]

    n1, bins1, patches1 = plt.hist(d0, n_bins, facecolor='orange', alpha = 0.5)

    n0, bins0, patches0 = plt.hist(d1, n_bins, facecolor='blue', alpha = 0.5)

    plt.xlabel(look)

    plt.ylabel('Employees')

    plt.title(look + ' vs Retention')

    plt.legend(labels = ['retained','left'])

    plt.show()
d_hist( X,y, 'satisfaction_level',50)

d_hist( X,y, 'last_evaluation', 50)

d_hist( X,y, 'average_montly_hours', 50)
def d_bar(X,y,look):

    X0 = X.loc[ y == 0 ]

    X1 = X.loc[ y == 1 ]

    c0 = X0[look].value_counts(sort = False)

    c1 = X1[look].value_counts(sort = False)

    f0 = pd.DataFrame(c0)

    f1 = pd.DataFrame(c1)

    f = pd.concat([f0,f1],axis=1)

    f = f.fillna(0).astype(int)

    l = np.arange(f.shape[0])

    plt.bar(l,f.iloc[:,0]/sum(f.iloc[:,0]),facecolor = 'orange', alpha = 0.5)

    plt.bar(l,f.iloc[:,1]/sum(f.iloc[:,1]),facecolor = 'blue', alpha = 0.5)

    if isinstance(f.index.values[0],str):

        plt.xticks(l,f.index.values, rotation = 70)

    else:

        plt.xticks(l,f.index.values)

    plt.xlabel(look)

    plt.ylabel('Percent Employees')

    plt.title(look + ' vs Retention')

    plt.legend(labels = ['retained','left'])

    plt.show()
d_bar( X, y, 'number_project')

d_bar( X, y, 'time_spend_company')

d_bar( X, y, 'Work_accident')

d_bar( X, y, 'promotion_last_5years')

d_bar( X, y, 'sales')

d_bar( X, y, 'salary')
def d_dot( X, y, look1, look2):

    x1 = X[look1]; x2 = X[look2]

    x10 = x1[y==0]; x20 = x2[y==0]

    x11 = x1[y==1]; x21 = x2[y==1]

    plt.plot(x10,x20, marker = '.', linestyle = 'None', 

             color = 'orange', alpha = 0.5, label = 'Retained')

    plt.plot(x11,x21, marker = '.', linestyle = 'None',

             color = 'blue',alpha = 0.5, label = 'Left')

    plt.xlabel(look1)

    plt.ylabel(look2)

    plt.title(look1 + ' vs ' + look2)

    plt.legend()

    plt.show()
d_dot( X, y, 'satisfaction_level', 'last_evaluation')

d_dot( X, y, 'satisfaction_level', 'average_montly_hours')

d_dot( X, y, 'last_evaluation','average_montly_hours')