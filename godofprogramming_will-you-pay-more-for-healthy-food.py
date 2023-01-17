URL = r"../input/responses.csv"



# Import utilities

from pandas import read_table

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Import classifiers

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV
# For Plotting

# __________________________________________



def plot_avg_p_r_curves(precision, recall, average_precision):

    plt.figure()

    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,

             where='post')

    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,

                     color='b')



    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'

                .format(average_precision["micro"]))



def plot_per_class_p_r_curves(precision, recall, average_precision, classes):

    from itertools import cycle

    # setup plot details

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])



    plt.figure(figsize=(7, 8))

    lines = []

    labels = []

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)

    lines.append(l)

    labels.append('micro-average Precision-recall (area = {0:0.2f})'

                  ''.format(average_precision["micro"]))



    for i, color in zip(range(len(classes)), colors):

        l, = plt.plot(recall[i], precision[i], color=color, lw=2)

        lines.append(l)

        labels.append('Precision-recall for class {0} (area = {1:0.2f})'

                      ''.format(classes[i], average_precision[i]))



    fig = plt.gcf()

    fig.subplots_adjust(bottom=0.25)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('Extension of Precision-Recall curve to multi-class')

    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()



def pca_visualization(X_train, y, n_classes, start_label=0):

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2) #2-dimensional PCA

    transformed = pd.DataFrame(pca.fit_transform(X_train))



    colors = ['red','blue','lightgreen','brown','cyan']



    # Modify indices accordingly if classes don't start from 0

    for i in range(start_label, n_classes+start_label*1):

        plt.scatter(transformed[y==i][0], transformed[y==i][1], label='Class {}'.format(i), c=colors[i-start_label])

    #plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class 2', c='blue')

    #plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

    #plt.scatter(transformed[y==4][0], transformed[y==4][1], label='Class 4', c='brown')

    #plt.scatter(transformed[y==5][0], transformed[y==5][1], label='Class 5', c='cyan')



    plt.legend()

    plt.show()



# ________________________________________________

# For Preprocessing

# ________________________________________________



def transform_categorical_to_numerical(frame):

    '''

    Transforms categorical columns to numerical.

    '''

    cat_columns = frame.select_dtypes(['O']).columns

    frame[cat_columns] = frame[cat_columns].apply(lambda x: x.astype('category'))   # change type

    frame[cat_columns] = frame[cat_columns].apply(lambda x: x.cat.codes)    # change to numerical

    return frame



def get_features_and_labels(frame, lbl_col, classes=None, binarize=False):

    '''

    Transforms and scales the input data and returns numpy arrays for

    training and testing inputs and targets.

    'classes' required if binarize=True.

    '''

    frame = transform_categorical_to_numerical(frame)



    # Convert values to floats

    arr = np.array(frame, dtype=np.float)



    # Use the lbl_col as the target value

    X, y = np.delete(arr, lbl_col, 1), arr[:, [lbl_col]]



    # Transform labels to binomial dist. (PS: This is a bespoke step to say 'yes' or 1, if y score is >=3)

    #y = np.array([0 if x<3 else 1 for x in y])

    #y_ = []

    #for e in y:

    #    if e < 3:

    #        y_.append(1)

    #    elif e==3 or e != e:    # account for nan

    #        y_.append(2)

    #    else:

    #        y_.append(3)



    #y = np.array(y_)



    # Use 80% of the data for training; test against the rest

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



    # Replace missing values in labels with 0.0. Use

    # scikit-learn to calculate missing values (below)

    #frame[frame.isnull()] = 0.0



    # Impute missing values from the training data

    from sklearn.preprocessing import Imputer

    imputer = Imputer(strategy='mean')

    imputer.fit(X_train)

    X_train = imputer.transform(X_train)

    X_test = imputer.transform(X_test)

    imputer = Imputer(strategy='most_frequent')

    imputer.fit(y_train)

    y_train = imputer.transform(y_train)

    y_test = imputer.transform(y_test)

    

    # Binarize labels for multiclass PR support

    if binarize==True:

        from sklearn.preprocessing import label_binarize

        y_train = label_binarize(y_train, classes)

        y_test = label_binarize(y_test, classes)

    

    # Normalize the attribute values to mean=0 and variance=1

    #from sklearn.preprocessing import StandardScaler

    #scaler = StandardScaler()

    # To scale to a specified range, we can use MinMaxScaler

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    

    # Fit the scaler based on the training data, then apply the same

    # scaling to both training and test sets.

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)



    # Return the training and test sets

    return X_train, X_test, y_train, y_test



# =====================================================================
frame = read_table(

        URL,

        # Specify the file encoding

        # Latin-1 is common for data from US sources

        encoding='latin-1',

        #encoding='utf-8',  # UTF-8 is also common



        # Specify the separator in the data

        sep=',',            # comma separated values



        # Ignore spaces after the separator

        skipinitialspace=True,



        # Generate row labels from each row number

        index_col=None,



        # Generate column headers row from each column number

        header=0,          # use the first line as headers

    )
f = list(frame) # Extract headers

print(f)
D = np.array(frame)

for element in [e for e in D]:

    [print(i,x) for (i,x) in enumerate(element) if x != x]
# Transforms categorical columns to numerical.

cat_columns = frame.select_dtypes(['O']).columns

frame[cat_columns] = frame[cat_columns].apply(lambda x: x.astype('category'))   # change type

frame[cat_columns] = frame[cat_columns].apply(lambda x: x.cat.codes)    # change to numerical



# Convert values to floats

arr = np.array(frame, dtype=np.float)



lbl_col = 139  # Our label is 'Spending on healthy eating' (1-5)

print("Our label is: " + f[lbl_col])

classes = [1,2,3,4,5]  # For use with label binarizer



# Use the lbl_col as the target value

X, y = np.delete(arr, lbl_col, 1), arr[:, [lbl_col]]



# Use 80% of the data for training; test against the rest

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Replace missing values in labels with 0.0. Use

# scikit-learn to calculate missing values (below)

#frame[frame.isnull()] = 0.0



def impute_and_normalize(X_train, X_test, y_train, y_test, impute_y=True):

    # Impute missing values from the training data

    from sklearn.preprocessing import Imputer

    imputer = Imputer(strategy='mean')

    imputer.fit(X_train)

    X_train = imputer.transform(X_train)

    X_test = imputer.transform(X_test)

    

    if impute_y == True:

        imputer = Imputer(strategy='most_frequent')

        imputer.fit(y_train)

        y_train = imputer.transform(y_train)

        y_test = imputer.transform(y_test)



    # Perform data normalization

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))



    # Fit the scaler based on the training data, then apply the same

    # scaling to both training and test sets.

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = impute_and_normalize(X_train, X_test, y_train, y_test)
pca_visualization(X_train, y_train, 5, 1)
# Binarize labels for multiclass PR support

from sklearn.preprocessing import label_binarize

y_train = label_binarize(y_train, classes)

y_test = label_binarize(y_test, classes)
# Define an execution function with Precision-Recall helper

from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

def execute(clf, clf_name, X_train, X_test, y_train, y_test, skip_pr):

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = f1_score(y_test, pred, average='micro')

    acc = accuracy_score(y_test, pred)



    # For generating the P-R curve

    try:

        y_prob = clf.decision_function(X_test)

    except AttributeError:

        # Handle BernoilliNB

        y_prob = clf.predict_proba(X_test)



    # PS: OVR wrapper must be used for multiclass, with label binarizer.

    precision, recall, avg = get_per_class_pr_re_and_avg(y_test, y_prob) if skip_pr==False else ({},{},{})



    # Include the score in the title

    print('\n{} (F1 score={:.3f}, Accuracy={:.4f})'.format(clf_name, score, acc))

    

    if skip_pr==False:

        print('Average precision score, micro-averaged over all classsses: {0:0.2f}'.format(avg["micro"]))

        

    return precision, recall, avg



def get_per_class_pr_re_and_avg(Y_test, y_score):

    from sklearn.metrics import precision_recall_curve

    from sklearn.metrics import average_precision_score



    # For each class

    precision = dict()

    recall = dict()

    average_precision = dict()



    for i in range(Y_test.shape[1]):

        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])

        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])



    # A "micro-average": quantifying score on all classes jointly

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())

    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    return precision, recall, average_precision
def runClassifiers(X_train, X_test, y_train, y_test, skip_pr=False):

    '''

     Skip_pr when not using OVR wrapper with multiclass label binarizer.

     This is due to compatibility reasons with PR curve functions.

    '''

    # Evaluate multiple classifiers on the data

    print("Evaluating classifiers")

    classifiers = []

    classifiers.append([LinearSVC(C=1.0), "LinearSVC"])

    classifiers.append([MultinomialNB(), "MultiNB"])

    classifiers.append([KNeighborsClassifier(n_neighbors=10), "kNN"])

    classifiers.append([AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R'), "AdaBoost"])

    classifiers.append([RandomForestClassifier(n_estimators=100), "Random forest"])

    classifiers.append([SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo'), "Baseline ovo rbf SVM"])



    results = []



    for clf in classifiers:

        results.append(execute(OneVsRestClassifier(clf[0]), clf[1], X_train, X_test, y_train, y_test, skip_pr))

    return results



results = runClassifiers(X_train, X_test, y_train, y_test)
plot_per_class_p_r_curves(results[3][0], results[3][1], results[3][2], classes)
def get_k_best_features(X_train, X_test, y_train, top_k=20):

    # Select top 20 features that explain the classes

    from sklearn.feature_selection import SelectKBest, chi2



    ch2 = SelectKBest(chi2, k=top_k)

    X_train = ch2.fit_transform(X_train, y_train)

    X_test = ch2.transform(X_test)



    # Print selected feature names

    top_features = [f[i] for i in ch2.get_support(indices=True)]

    print("Top {} features = {}".format(top_k, top_features))

    return X_train, X_test



X_train, X_test = get_k_best_features(X_train, X_test, y_train)
results = runClassifiers(X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_test, y_train, y_test = impute_and_normalize(X_train, X_test, y_train, y_test)
results = runClassifiers(X_train, X_test, y_train, y_test, True)
X_train, X_test = get_k_best_features(X_train, X_test, y_train)

results = runClassifiers(X_train, X_test, y_train, y_test, True)
classifiers2 = []



param_grid = {'C': np.logspace(-2, 1, 10),

                  'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],     # Use for testing rbf, poly

                 'degree': [1,2,3] }

clf_rbf_ovo = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

clf_rbf_ovr = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovr'), param_grid, cv=5)

clf_linear_ovo = GridSearchCV(SVC(kernel='linear', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

clf_poly_ovo = GridSearchCV(SVC(kernel='poly', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

classifiers2.append([clf_rbf_ovo, "clf_rbf_ovo"])

classifiers2.append([clf_rbf_ovr, "clf_rbf_ovr"])

classifiers2.append([clf_linear_ovo, "clf_linear_ovo"])

classifiers2.append([clf_poly_ovo, "clf_poly_ovo"])



for clf in classifiers2:

    execute(clf[0], clf[1], X_train, X_test, y_train.ravel(), y_test.ravel(), True)

    print(clf[0].best_estimator_)

    print()
y_ = []

for e in y:

    if e < 3:

        y_.append(1)

    elif e==3 or e != e:    # account for nan, append neutral.

        y_.append(2)

    else:

        y_.append(3)



y_ = np.array(y_)



X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)

X_train, X_test, y_train, y_test = impute_and_normalize(X_train, X_test, y_train, y_test, impute_y=False)



X_train, X_test = get_k_best_features(X_train, X_test, y_train)

results = runClassifiers(X_train, X_test, y_train.ravel(), y_test.ravel(), True)
pca_visualization(X_train, y_train, 3, 1)
# Performing GridSearchCV on RBF SVM on 3 classes

param_grid = {'C': np.logspace(-2, 1, 10),

                  'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],     # Use for testing rbf, poly

                 'degree': [1,2,3] }

clf_rbf_ovo = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

execute(clf_rbf_ovo, "RBF SVM OVO", X_train, X_test, y_train.ravel(), y_test.ravel(), True)

print(clf_rbf_ovo.best_estimator_)
y_ = np.array([0 if x<=3 else 1 for x in y])



X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)

X_train, X_test, y_train, y_test = impute_and_normalize(X_train, X_test, y_train, y_test, impute_y=False)



X_train, X_test = get_k_best_features(X_train, X_test, y_train)

results = runClassifiers(X_train, X_test, y_train.ravel(), y_test.ravel(), True)
pca_visualization(X_train, y_train, 2)
# Performing GridSearchCV on RBF SVM on 2 classes

param_grid = {'C': np.logspace(-2, 1, 10),

                  'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],     # Use for testing rbf, poly

                 'degree': [1,2,3] }

clf_rbf_ovo = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

execute(clf_rbf_ovo, "RBF SVM OVO", X_train, X_test, y_train.ravel(), y_test.ravel(), True)

print(clf_rbf_ovo.best_estimator_)



clf_linear_ovo = GridSearchCV(SVC(kernel='linear', class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)

execute(clf_linear_ovo, "Linear SVM OVO", X_train, X_test, y_train.ravel(), y_test.ravel(), True)

print(clf_linear_ovo.best_estimator_)
def execute_regression(clf, clf_name, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = r2_score(y_test, pred)

    acc = mean_squared_error(y_test, pred)



    # Include the score in the title

    print('\n{} (R2 score={:.3f}, MSE={:.4f})'.format(clf_name, score, acc))

    

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_squared_error



regressors = []

regressors.append([LinearRegression(), "Linear Regression"])

regressors.append([SVR(), "SVR"])



for clf in regressors:

    execute_regression(clf[0], clf[1], X_train, X_test, y_train.ravel(), y_test.ravel())