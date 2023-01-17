# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns



def get_var_category(series):

    unique_count = series.nunique(dropna=False)

    total_count = len(series)

    if pd.api.types.is_numeric_dtype(series):

        return 'Numerical'

    elif pd.api.types.is_datetime64_dtype(series):

        return 'Date'

    elif unique_count==total_count:

        return 'Text (Unique)'

    else:

        return 'Categorical'





def print_categories(df):

    for column_name in df.columns:

        print(column_name, ": ", get_var_category(df[column_name]))





def explore_dataframe(df):

    print(df.shape)

    print(df.axes)

    print(df.dtypes)

    print(df.head(20))

    for column in df:

        print(column)

        print(df[column].value_counts())

    print(df.describe())

    print_categories(df)

    print(df.apply(lambda x: sum(x.isnull()), axis=0))



#Math Course Data

mat_students = pd.read_csv('../input/student-mat.csv')

explore_dataframe(mat_students)

sns.pairplot(mat_students, hue="Walc")

plt.show()

sns.heatmap(mat_students.corr(), vmax=.8, square=True);

plt.show()

#Portuguese Course Data

por_students = pd.read_csv('../input/student-por.csv')

explore_dataframe(por_students)

sns.pairplot(por_students,  hue="Walc")

plt.show()

sns.heatmap(por_students.corr(), vmax=.8, square=True);

plt.show();
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer





def de_categorize_df(df):

    # variable categorical en numericas

    v = DictVectorizer()

    # varialbe no numericas de estudiante

    qualitative_features = ['reason', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

    X_qual = v.fit_transform(df[qualitative_features].to_dict('records'))

    # utilizar one hot encoder por evitar valores numerico que sean malinterpretados por el modelo

    enc_onehot = OneHotEncoder()

    # estrae las variables

    train_cat_data = enc_onehot.fit_transform(X_qual.toarray()[:, 0:29]).toarray()

    # cat_columns = por_students.select_dtypes(['category']).columns

    # por_students[cat_columns] = por_students[cat_columns].apply(lambda x: x.cat.codes)

    Y_1 = df.values[:, 30]

    Y_2 = df.values[:, 31]

    Y_Final = df.values[:, 32]

    # Assuming that in portugal 14 is the minimum to pass

    # reduce to approved not approved (binary class)

    Y_Final_reduced = map(lambda x: 0 if x < 6 else 0 if x < 9 else 1 if x < 15 else 1 if x < 19 else 1, Y_Final)

    # Split-out validation dataset

    return (train_cat_data, list(Y_Final_reduced))





def train_models(X, Y):

    # reduce to approved not approved (binary class)

    # parametros de validacion de los modelos

    validation_size = 0.20

    seed = 7

    scoring = 'accuracy'

    # generacion de training y testing sets (20% de test, 80% de train)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # modelos de clasificacion

    models = []

    models.append(('LR', LogisticRegression()))

    models.append(('LDA', LinearDiscriminantAnalysis()))

    models.append(('KNN', KNeighborsClassifier()))

    models.append(('CART', DecisionTreeClassifier()))

    models.append(('NB', GaussianNB()))

    models.append(('SVM', SVC()))

    # evaluacion de los modelos

    results = []

    names = []

    for name, model in models:

    # 10-fold cross validation

        kfold = model_selection.KFold(n_splits=10, random_state=seed)

        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        # print accuracy de los modelos

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

    # grafico de resumen de los modelos de clasificacion

    fig = plt.figure()

    fig.suptitle('Algorithm Comparison')

    ax = fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()





# cargando los datasets

mat_students = pd.read_csv('../input/student-mat.csv')

por_students = pd.read_csv('../input/student-por.csv')

#tranformar variable en numericas

X_mat, Y_GF_mat = de_categorize_df(mat_students)

X_por, Y_GF_por = de_categorize_df(por_students)

#entrenar y validar modelos

train_models(X_mat,Y_GF_mat)

train_models(X_por,Y_GF_por)