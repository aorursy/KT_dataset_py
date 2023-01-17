import seaborn as sns

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np
data = pd.read_csv("../input/train.csv")
def plot_distrib(data, category=None) :

    

    #Size the figure that will hold all the subplots

    plt.figure(figsize=(len(data.columns)*2, len(data.columns)*2))

    i = 1

    

    #For all the columns, make a subplot

    for col in data.columns :

        plt.subplot(len(data.columns)//3 + 1, 3, i)

        

        #If this is a numerical type : 

        if data[col].dtypes != object :

            

            #Plot the values

            sns.distplot(data[col], kde=False)

            

            if category != None :

                sns.distplot(data[data[category] == 0][col], kde=False, label="Perished")

            

            #Add information on the most popular and the most successfull parameters

            plt.title("{}".format(col))

            plt.legend()

            

        #If this is a categorical type :

        else :

            

            #Plot the values and cleaned values, in descending order of occurence

            maxi = data[col].value_counts().sort_values(ascending=False).index

            sns.countplot(data[col], order=maxi)

            

            if category != None :

                sns.countplot(data[data[category] == 0][col], order=maxi, label="Perished", saturation=.3)

            

            #Add information on the most popular and the most successfull parameters

            plt.title("{}".format(col))

            plt.legend()

        

        #Rotate the labels on x-axis

        plt.xticks(rotation=90)

        i += 1

    

    #Espace each subplot to avoid overlap

    plt.subplots_adjust(hspace=0.7)
def filler(data) :

    #For each column, if the class is object, fill with "unknown"

    for col in data.columns :

        if data[col].dtypes == object :

            data[col] = data[col].fillna("Unknown")

        

        #If the class is numerical, fill with random values distributed around mean value

        else :

            col_avg = data[col].mean()

            col_std = data[col].std()

            col_null_count = data[col].isnull().sum()

            if col_null_count != 0:

                col_null_random_list = np.random.randint(low=col_avg - col_std, high=col_avg + col_std, size=col_null_count)

                data[col][np.isnan(data[col])] = col_null_random_list

                data[col] = data[col].astype(int)
def plot_cross_num(data) :

    

    #Search for the numerical types

    col_num = []

    for col in data.columns :

        if data[col].dtypes != object :

            col_num.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(len(data.columns)*2, len(data.columns)*2))

    i = 1

    

    #For each column

    for col in col_num :

        col_num.remove(col)

        for col2 in col_num :

        

            #Plot the values

            plt.subplot(len(data.columns)//3 +1, 3, i)

            sns.lmplot(x=col, y=col2, data=data, fit_reg=False, hue="Survived", palette="Set1")            

            sns.kdeplot(data[col], data[col2], n_levels=20)



            #Add information

            plt.title("{} co-plotted with {}".format(col, col2))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

    

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)
def plot_cross_cat(data) :

    

    #Search for the categorical types

    col_cat = []

    for col in data.columns :

        if data[col].dtypes == object :

            col_cat.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(len(data.columns)*2, len(data.columns)*2))

    i = 1

    

    #For each column

    for col in col_cat :

        col_cat.remove(col)

        for col2 in col_cat :

        

            #Plot the values

            plt.subplot(len(data.columns)//3 + 1, 3, i)

            table_count = pd.pivot_table(data,values=['Fare'],index=[col],columns=[col2],aggfunc='count',margins=False)

            sns.heatmap(table_count['Fare'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)



            #Add information

            plt.title("{} co-plotted with {}".format(col, col2))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

        

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)
def plot_cross_cross(data) :

    

    #Search for the types

    col_cat = []

    col_num = []

    for col in data.columns :

        if data[col].dtypes == object :

            col_cat.append(col)

        else :

            col_num.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(len(data.columns)*2, len(data.columns)*2))

    i = 1

    j = len(col_cat)

    k = len(col_num)

    #For each column

    for cat in col_cat :

        for num in col_num :

        

            #Plot the values

            plt.subplot(j,k,i)

            

            sns.violinplot(x=cat , y=num, data=data, inner=None) 

            #sns.swarmplot(x=cat, y=num, data=data, alpha=0.7) 



            #Add information

            plt.title("{} co-plotted with {}".format(cat, num))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

        

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)
data.head()
data.info()
data.describe()
filler(data)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['IsAlone'] = data['IsAlone'].astype(object)



data = data.drop(['SibSp', 'Parch', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['IsAlone'] = data['IsAlone'].astype(object)

data['Survived'] = data['Survived'].astype(object)

data['Pclass'] = data['Pclass'].astype(object)
plot_distrib(data, "Survived")

plt.show()
plot_cross_cat(data)

plt.show()
plot_cross_cross(data)

plt.show()
from sklearn import svm

from sklearn import neighbors

from sklearn import ensemble



from sklearn import cross_validation

from sklearn import grid_search

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import itertools
def cross_validate_with_scaling(design_matrix, labels, classifier, cv_folds):

    pred = np.zeros(labels.shape) # vector of 0 in which to store the predictions

    for tr, te in cv_folds:

        

        # Restrict data to train/test folds

        Xtr = design_matrix[tr, :]

        ytr = labels[tr]

        Xte = design_matrix[te, :]



        # Scale data

        scaler = preprocessing.StandardScaler() # create scaler

        Xtr = scaler.fit_transform(Xtr) # fit the scaler to the training data and transform training data

        Xte = scaler.transform(Xte) # transform test data



        # Fit classifier

        classifier.fit(Xtr, ytr)



        # Predict probabilities (of belonging to +1 class) on test data

        pred[te] = classifier.predict(Xte) # two-dimensional array

        # Identify the index, in yte_pred, of the positive class (y=1)

        # using classifier.classes_

        # index_of_class_1 = np.nonzero(classifier.classes_ == 1)[0][0] 

              

    return pred
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

def categorize(data_train) :

    for col in data_train :

        #If this is a category, it is transformed in a number

        if data_train[col].dtypes == object :

            data_train[col] = le.fit_transform(data_train[col].values)

            data_train[col] = data_train[col].astype(float)

        #If there is too many (>10) possible value for a category, groups them

        elif len(data_train[col].value_counts()) > 10 :

            data_train[col] = pd.cut(data_train[col], 9)

            data_train[col] = le.fit_transform(data_train[col].values)

        data_train[col] = data_train[col].astype(float)

    return data_train.values
def evaluation(y, y_predc, target_names) :

    print(classification_report(y, y_predc, target_names=target_names))



    #Recall : [1,1]/[1,1]+[1,0]  ou  tp/tp+fp

    print("Recall : ", metrics.recall_score(y, y_predc))



    #Precision : [1,1]/[1,1]+[0,1]  ou  ou  tp/tp+fn

    print("Accuracy : ", metrics.precision_score(y, y_predc))



    #ROC Curve : plot le taux de vrai positif sur faux positif. Uniquement pour les binaires

    fpr, tpr, thresholds = metrics.roc_curve(y, y_predc)

    print("Roc score : ", metrics.roc_auc_score(y, y_predc)) #Aire sous la courbe

    

    auc = metrics.auc(fpr, tpr)



    plt.plot(fpr, tpr, '-', color='orange', label='AUC = %0.3f' % auc)

    plt.plot(fpr, fpr, '--')



    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.title('ROC curve', fontsize=16)

    plt.legend(loc="lower right")

    plt.show()

    

    cnf = confusion_matrix(y, y_predc)



    plot_confusion_matrix(cnf, classes=target_names, normalize=True, title='Normalized confusion matrix')

    plt.show()
def importance(data_train, clf) :

    features_list = data_train.columns.values

    feature_importance = clf.feature_importances_

    sorted_idx = np.argsort(feature_importance)

 

    plt.figure(figsize=(5,7))

    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

    plt.xlabel('Importance')

    plt.title('Feature importances')

    plt.draw()

    plt.show()
#Transform dataset

data = data.sample(frac=1)

y = data["Survived"].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y)



#Transform the data so that it doesn't obviously have the response in it, and transform label in numbers

data_train = data.copy().drop(["Survived"], axis=1)



data_values = categorize(data_train)



folds = cross_validation.StratifiedKFold(y, 10, shuffle=True)

data_train.head()

target_names = ('Perished', 'Survived')
clf = svm.SVC()



param = {'C': [.01, .1, 1, 5, 10, 30, 50], 'kernel': ['rbf', 'sigmoid'], 'gamma': [1/1000, 1/891, 1/500, 1/200, 1/100, 1/50, 1/10, 1/2]}

grid = grid_search.GridSearchCV(clf, param)



y_predc = cross_validate_with_scaling(data_values, y, grid, folds)

print(grid.best_params_)
evaluation(y, y_predc, target_names)
clf = neighbors.KNeighborsClassifier()



param = {'n_neighbors': list(range(2,10)), 'weights': ['uniform', 'distance']}

grid = grid_search.GridSearchCV(clf, param)



y_predc = cross_validate_with_scaling(data_values, y, grid, folds)

print(grid.best_params_)
evaluation(y, y_predc, target_names)
clf = ensemble.RandomForestClassifier(n_estimators=70)



clf.fit(data_values, y)



param = { 

         'max_features': ['auto', 'log2', None], 

         'class_weight': ['balanced', {0: 1/2, 1: 1/2}], 

         'min_samples_leaf': list(range(1,3)), 

         'min_samples_split': list(range(2,3)),

         

        }

grid = grid_search.GridSearchCV(clf, param)



y_predc = cross_validate_with_scaling(data_values, y, grid, folds)

print(grid.best_params_)
evaluation(y, y_predc, target_names)
importance(data_train, clf)
clf = ensemble.ExtraTreesClassifier()



clf.fit(data_values, y)



param = {'n_estimators': list(range(5,50))}

grid = grid_search.GridSearchCV(clf, param)



y_predc = cross_validate_with_scaling(data_values, y, grid, folds)

print(grid.best_params_)
evaluation(y, y_predc, target_names)
importance(data_train, clf)