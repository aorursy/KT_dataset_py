import numpy as np                     # array goodnes

from pandas import DataFrame, read_csv # excel for python

from matplotlib import pyplot as plt   # plotting library

from pandas import DataFrame, read_csv # excel for python



%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')       # nice colors

plt.xkcd()

plt.rc('font',family='DejaVu Sans')

plt.rcParams['figure.figsize'] = (12, 8)





def plot_decision_surface(CLF, df, labels, axes, plot_step=1e-1):

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(labels.reshape(-1))

    fig, axs = plt.subplots(len(axes), len(axes), figsize=(12, 12))

    for idx in range(len(axes)):

        for idy in range(len(axes)):

            ax = axs[idy][idx]

            if idx == 0: ax.set_ylabel(axes[idy])

            if idy == len(axes)-1: ax.set_xlabel(axes[idx])

            if idx == idy: continue

            x_label, y_label = axes[idx], axes[idy]

            x, y = df[x_label], df[y_label]

            _clf = CLF()

            _clf.fit(np.column_stack([x, y]), labels)

            xx, yy = np.meshgrid(np.arange(np.min(x)-1, np.max(x)+1, plot_step),

                                 np.arange(np.min(y)-1, np.max(y)+1, plot_step))

            Z = _clf.predict(np.c_[xx.ravel(), yy.ravel()])

            Z = le.transform(Z)

            Z = Z.reshape(xx.shape)

            cs = ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

            l = labels.reshape(-1)

            for label in le.classes_:

                ax.scatter(x[l==label], y[l==label], label=label, 

                           cmap=plt.cm.RdYlBu, edgecolors='black')

    fig.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
df_iris = read_csv('../input/iris/Iris.csv')

X_iris = df_iris.drop(['Species', 'Id'], axis=1)

Y_iris = df_iris['Species'].reshape((-1, 1))
X_iris.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, Y_train, Y_test = train_test_split(X_iris, Y_iris, 

                                                    test_size=0.33, random_state=42)



clf = DecisionTreeClassifier()

clf.fit(X_train, Y_train)



clf.score(X_test, Y_test)
plot_decision_surface(DecisionTreeClassifier, X_iris, Y_iris, ['SepalLengthCm', 'SepalWidthCm', 

                                                               'PetalLengthCm', 'PetalWidthCm', 

                                                               ])
def plot_importance(clf, df):

    plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)

    plt.xticks(range(len(clf.feature_importances_)), df.columns)

    

plot_importance(clf, X_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



# read the data set

df_titanic = read_csv('../input/titanic/train.csv')



# handle nan entries. 

df_titanic = df_titanic.fillna(0)



# convert string data to numbers

for idx in ['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked']:

    df_titanic[idx] = le.fit_transform(df_titanic[idx].astype(str))

# create feature vector

X_titanic = df_titanic.drop(['Survived'], axis=1)

Y_titanic = df_titanic['Survived'].reshape((-1, 1))



# perform test / train split

X_train, X_test, Y_train, Y_test = train_test_split(X_titanic, Y_titanic, 

                                                    test_size=0.33, random_state=42)



# create the classifier

clf = RandomForestClassifier(n_estimators=100)



# train the classifier

clf.fit(X_train, Y_train)



# evalutate the classifier

clf.score(X_test, Y_test)