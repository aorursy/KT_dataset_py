###############################################################

# NB: shift + tab HOLD FOR 2 SECONDS!

###############################################################





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('\n ')

print('Getting traing dataset...')

data = pd.read_csv('../input/pokemon/Pokemon.csv')

print('Traing data set obtained. \n')
data.head(3)
def type_numbering(string) : 

    number = 0

    if string == 'Normal' :

        number = 1

    elif string == 'Fire' :

        number = 2

    elif string == 'Fighting' :

        number = 3

    elif string == 'Water' :

        number = 4

    elif string == 'Flying' :

        number = 5

    elif string == 'Grass' :

        number = 6

    elif string == 'Poison' :

        number = 7

    elif string == 'Electric' :

        number = 8

    elif string == 'Ground' :

        number = 9

    elif string == 'Psychic' :

        number = 10

    elif string == 'Rock' :

        number = 11

    elif string == 'Ice' :

        number = 12

    elif string == 'Bug' :

        number = 13

    elif string == 'Dragon' :

        number = 14

    elif string == 'Ghost' :

        number = 15

    elif string == 'Dark' :

        number = 16

    elif string == 'Steel' :

        number = 17

    elif string == 'Fairy' :

        number = 18

    else :

        number = 0

    

    return number;
def DT_RF_classifier(data, kind='DT', test_size=0.3, max_depth=None):

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report,confusion_matrix

    

    print('Splitting data...')

    df = data

    df['Type 1'] = data['Type 1'].apply(type_numbering)

    df['Type 2'] = data['Type 2'].apply(type_numbering)

    X = df.drop('Legendary',axis=1).drop('Name', axis=1)

    y = df['Legendary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print('Splitting done. \n')



    print('Initializing classifier...')

    if (kind == 'DT'):

        from sklearn.tree import DecisionTreeClassifier

        

        print('Classifier type: Decision Tree ')

        print('Fitting classifier...')

        clf = DecisionTreeClassifier(max_depth=max_depth)

        clf.fit(X_train,y_train)

        

        predictions = clf.predict(X_test)

        print('Fit done. \n')

    

    else :

        from sklearn.ensemble import RandomForestClassifier

        

        print('Classifier type: Random Forest ')

        print('Fitting classifier...')

        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=100)

        clf.fit(X_train,y_train)

        

        predictions = clf.predict(X_test)

        print('Fit done. \n')



    print('Evaluating the model...')

    

    print(classification_report(y_test,predictions))

    print('The score is: ', clf.score(X_test, y_test))

    print('\n')

    cm = confusion_matrix(y_test,predictions)

    print(cm)

    df_cm = pd.DataFrame(cm, index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])

    plt.figure(figsize = (7,7))

    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

    plt.xlabel("Predicted Class", fontsize=18)

    plt.ylabel("True Class", fontsize=18)

    

    if (kind == 'DT'):

        from sklearn.externals.six import StringIO  

        from IPython.display import Image  

        from sklearn.tree import export_graphviz

        

        

        dot_data = StringIO()

        

        export_graphviz(clf, out_file='tree.dot',

                        feature_names = X.columns.values, 

                        class_names = ['Non-Legendary', 'Legendary'], 

                        filled=True, rounded=True, proportion=False, precision=2)

        

    else : 

        from sklearn.externals.six import StringIO  

        from IPython.display import Image  

        from sklearn.tree import export_graphviz

        

        # Extract single tree

        estimator = clf.estimators_[5]

        

        # Export as dot file

        export_graphviz(estimator, out_file='tree.dot', 

                feature_names = X.columns.values,

                class_names = ['Non-Legendary', 'Legendary'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)

        

    # Convert to png

    from subprocess import call

    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

        

    # Display in python

    plt.figure(figsize = (14, 18))

    plt.imshow(plt.imread('tree.png'))

    plt.axis('off');

        

        

    plt.show()

    print('\n ')

    print('Process ended. ')

    

    return clf
DT_RF_classifier(data)
DT_RF_classifier(data, max_depth=5)
DT_RF_classifier(data, kind='RF', max_depth=5)
def LR_classifier(data, test_size=0.3):

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report,confusion_matrix

    

    print('Splitting data...')

    df = data

    df['Type 1'] = data['Type 1'].apply(type_numbering)

    df['Type 2'] = data['Type 2'].apply(type_numbering)

    X = df.drop('Legendary',axis=1).drop('Name', axis=1)

    y = df['Legendary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print('Splitting done. \n')



    print('Initializing classifier...')

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()

    clf.fit(X_train, y_train) # We fit the Logistic Regression Classifier

    predictions = clf.predict(X_test) # We compute the predictions

    

    print('Evaluating the model...')

    print(classification_report(y_test,predictions))

    print('The score is: ', clf.score(X_test, y_test))

    print('\n')

    cm = confusion_matrix(y_test,predictions)

    print(cm)

    df_cm = pd.DataFrame(cm, index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])

    plt.figure(figsize = (7,7))

    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

    plt.xlabel("Predicted Class", fontsize=18)

    plt.ylabel("True Class", fontsize=18)

    

    

    print('\n ')

    print('Process ended. ')

    

    return clf
LR_classifier(data)
def SVM_classifier(data, test_size=0.3):

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report,confusion_matrix

    

    print('Splitting data...')

    df = data

    df['Type 1'] = data['Type 1'].apply(type_numbering)

    df['Type 2'] = data['Type 2'].apply(type_numbering)

    X = df.drop('Legendary',axis=1).drop('Name', axis=1)

    y = df['Legendary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print('Splitting done. \n')



    print('Initializing classifier...')

    from sklearn.svm import SVC

    clf = SVC()

    clf.fit(X_train,y_train)

    

    # Starting the GridSearch

    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

    grid.fit(X_train,y_train)

    grid_predictions = grid.predict(X_test)

    print('\n ')

    print('\n ')

    print('Best set of parameters found by GridSearch: ', grid.best_params_)

    print('\n ')

    print('Initialization done. \n')

    

    

    

    print('Evaluating the model...')

    print(classification_report(y_test,grid_predictions))

    cm=confusion_matrix(y_test,grid_predictions)

    print(cm)

    df_cm = pd.DataFrame(cm, index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])

    plt.figure(figsize = (7,7))

    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

    plt.xlabel("Predicted Class", fontsize=18)

    plt.ylabel("True Class", fontsize=18)

    

    

    print('\n ')

    print('Process ended. ')

    

    return clf
SVM_classifier(data)