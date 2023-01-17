###################################
import warnings
warnings.filterwarnings('ignore')

# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as learn
from sklearn import preprocessing
from sklearn import metrics
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os

print(os.listdir("../input"))
def preProcessing():
    # the training data...
    train_orig = pd.read_csv('../input/train.csv')
    # the test data...
    test_orig = pd.read_csv('../input/test.csv')

    train2 = train_orig.copy()
    
    # here, we do some mapping to make the features more readable...
    train2['Survived_Label'] = train2['Survived'].map({0:'No',1:'Yes'})
    train2['Sex'] = train2['Sex'].map({'male':0,'female':1})
    train2['Embarked'] = train2['Embarked'].map({'C':'Cherbourg',
                                                 'Q':'Queenstown',
                                                 'S':'Southhampton'})
    train2['Embarked_Number'] = train2['Embarked'].map({'Cherbourg':1,'Queenstown':2,'Southhampton':3})
    train2 = train2.drop(['Survived', 'Cabin', 'Ticket', 'PassengerId'], axis=1);
    
    # fill in missing age data with the median... 
    train2['Age']=train2['Age'].fillna(train2['Age'].median())
    # fill in missing embarked number with the most common (Southhampton)...
    train2['Embarked_Number']=train2['Embarked_Number'].fillna(3)
    return train2
train2 = preProcessing()
train2.describe()
train2.head()
# now, let's make a "count plot."
# this will count the different types of data you give it.
# in this case, we will just give it the "Species" column of our data set.

sns.countplot(train2['Survived_Label'], palette = 'muted');
####################################
# violin plot function
@interactive # this line declares the following function as an "interactive" object
def interactiveViolinPlot(feature=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'], 
                          target = ['Survived_Label','Embarked'],
                          inner_style = ['quartiles','box','stick'],
                          colors=['muted','bright','colorblind']):
    sns.set_style("whitegrid") # this just gives us gridlines
    sns.violinplot(x=target, # this is the "category" we want to make plots for
                   y=feature, # this is the data we want to plot for each category
                   data=train2, # and this is the pandas dataframe we are using
                   inner=inner_style,
                   palette=colors) # the color scheme is given by whatever selection the user gives
    plt.show()
# let's take a look at some of the data we have...
interactiveViolinPlot
#####################################
# scatter plot to better understand individual features
@interactive
def interactiveScatterPlot(feature1=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                           feature2=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                           target = ['Survived_Label','Embarked'],
                           colors=['muted','bright','colorblind']):
    sns.lmplot(x=feature1, 
               y=feature2, 
               data=train2, 
               fit_reg=False, 
               hue=target,
               palette=colors,
               scatter_kws={'alpha':0.4})
    plt.show();
interactiveScatterPlot
######################################
# select multiple example
@interactive
def interactivePairPlot(features=widgets.SelectMultiple(description="features",
                                                        options=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'], # all the options
                                                        value=('Age','Fare',)), # the initial options selected by the widget
                        target = ['Survived_Label','Embarked'], # this is just a selection as in the previous examples...
                        colors = ['muted','bright','colorblind']): # ... and this too
    try:
        f = []
        for i in features:
            f.append(i)
        included = [target]+f
        sns.pairplot(train2[included], 
                     hue = target, # the "hue" colors the data by the tag you give (in this case, 'Survived_Label' or 'Embarked')
                     palette = colors, 
                     diag_kind='kde');
        plt.show()
    except TypeError:
        pass
interactivePairPlot
#############################################
# basic feature engineering
@interactive
def engineerFeatures(operation=['multiply','average'],
                     feature1=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                     feature2=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                     target = ['Survived_Label','Embarked'],
                     show = ['pairplot','PCA'],
                     colors = ['muted','bright','colorblind']):
    
    if feature1 != feature2:
        training_features = ['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number']
        scale = preprocessing.StandardScaler()
        train_scaled=scale.fit(train2[training_features]) # scaling the features to make sure they are equally weighted
        train_engr = train_scaled.transform(train2[training_features])
        train_engr = pd.DataFrame(data=train_engr, columns = training_features)
        
        # here, specify the details for each operation you provide in the "operation" list...
        if operation == 'multiply':
            train_engr['engineered_feature'] = train_engr.apply(lambda row: row[feature1]*row[feature2],axis=1)
        elif operation == 'average':
            train_engr['engineered_feature'] = train_engr.apply(lambda row: (row[feature1]+row[feature2])/2.,axis=1)
        
        # and visualize...
        included = [target]+['engineered_feature']+[feature1,feature2]
        for i in included:
            if i not in training_features:
                if i != 'engineered_feature':
                    train_engr[i] = train2[i]
        if show == 'pairplot':
            sns.pairplot(train_engr[included], 
                         hue = target, # the "hue" colors the data by the tag you give
                         palette = colors, 
                         diag_kind='kde',
                         plot_kws={'alpha': 0.4});
        elif show == 'PCA':
            train_temp = train2.copy()
            for c in train_temp.columns:
                if c not in training_features:
                    train_temp = train_temp.drop([c],axis=1)
            train_temp['engineered_feature'] = train_engr['engineered_feature']
            train_scaled=scale.fit(train_temp)
            train_temp = train_scaled.transform(train_temp)
            train_temp = pd.DataFrame(data=train_temp)
            pca = decomposition.PCA()
            pca_fit = pca.fit(train_temp)
            # print(pca_fit.explained_variance_ratio_)
            sns.barplot(x=np.arange(len(pca_fit.explained_variance_ratio_)),
                        y=pca_fit.explained_variance_ratio_,
                        palette='muted');
            plt.show()
    else:
        pass
engineerFeatures
##########################################
# confusion matrix code from github
'''
Note: the source for this function code may be found here:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
    else:
        cm = cm
#         print('Confusion matrix, without normalization')

#     print(cm)

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
    return

#########################################
## decision tree
@interactive
def getDecisionTreeModel(features = widgets.SelectMultiple(description='features',
                                                           options=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                                                           value=('Age',)),
                         target = ['Survived_Label']):
    
    try:
        training_features = []
        if 'value' not in dir(features):
            for i in features:
                training_features.append(i)
        else:
            training_features = list(features)
        print(training_features, target)
        print(train2['Embarked'].value_counts())
        kf = KFold(4, shuffle = True) # we will split the data 4 times, and test to see how well the classification performs each time.
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1)

        for k, (training_data, testing_data) in enumerate(kf.split(train2[training_features], train2[target])):

            # train the model using the training data for this "fold"...
            clf.fit(train2.iloc[training_data][training_features], # the feature data
                    train2.iloc[training_data][target]) # the target data

            # get the prediction...
            predicted = clf.predict(train2.iloc[testing_data][training_features])
            # and get the actual values
            actual = train2.iloc[testing_data][target]
            # get the confusion matrix
            cm = confusion_matrix(actual, predicted)
            plt.figure()
            print('trial',k)
            print(metrics.classification_report(actual, predicted))
            print()
            plot_confusion_matrix(cm, classes=train2[target].unique())
    except TypeError:
        pass


#####################################
# random forest
# trees = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100, 150, 200, 250, 300, 500],
# tree_depth = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50, 60, 70, 80]

@interactive
def getDecisionForestModel(features = widgets.SelectMultiple(description='features',
                                                           options=['Age', 'Fare', 'Sex', 'SibSp', 'Parch','Pclass','Embarked_Number'],
                                                           value=('Age',)),
                           target = ['Survived_Label'],
                           trees = widgets.IntSlider(value=1,min=1,max=500,step=2,description='Trees'), # a slider for integer inputs
                           tree_depth = widgets.IntSlider(value=1,min=1,max=80,step=2,description='Tree Depth')):
    
    try:
        training_features = []
        if 'value' not in dir(features):
            for i in features:
                training_features.append(i)
        else:
            training_features = list(features)
        print(training_features)
        print(train2['Embarked'].value_counts())
        kf = KFold(4, shuffle = True) # we will split the data 4 times, and test to see how well the classification performs each time.
        rfc = RandomForestClassifier(max_depth=tree_depth, n_estimators=trees,
                                     min_impurity_decrease = 0.0)

        for k, (training_data, testing_data) in enumerate(kf.split(train2[training_features], train2[target])):

            # train the model using the training data for this "fold"...
            rfc.fit(train2.iloc[training_data][training_features], # the feature data
                    train2.iloc[training_data][target]) # the target data

            # get the prediction...
            predicted = rfc.predict(train2.iloc[testing_data][training_features])
            # and get the actual values
            actual = train2.iloc[testing_data][target]
            # get the confusion matrix
            cm = confusion_matrix(actual, predicted)
            plt.figure()
            print('trial',k)
            print(metrics.classification_report(actual, predicted))
            print()
            plot_confusion_matrix(cm, classes=train2[target].unique())
    except TypeError:
        pass
getDecisionTreeModel
getDecisionForestModel
