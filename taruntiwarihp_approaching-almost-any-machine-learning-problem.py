# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import some usefull libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# scikit-learn to get data and perform t-SNE

from sklearn import datasets

from sklearn import manifold



%matplotlib inline

#Graphics in retina format are more sharp and Legible

%config inlinebackend.figure_format = 'retina'
data = datasets.fetch_openml(

    'mnist_784', 

    version=1,

    return_X_y=True

)



pixel_values, targets = data

targets = targets.astype(int)
pixel_values.shape
targets.shape
single_image = pixel_values[1, :].reshape(28, 28)



plt.imshow(single_image, cmap='gray')
single_image = pixel_values[2, :].reshape(28, 28)



plt.imshow(single_image, cmap='gray')
single_image = pixel_values[7, :].reshape(28, 28)



plt.imshow(single_image, cmap='gray')
tsne = manifold.TSNE(n_components=2, random_state=42)



transformed_data = tsne.fit_transform(pixel_values[:3000, :])
tsne_df = pd.DataFrame(

    np.column_stack((transformed_data, targets[:3000])),

    columns=["X","Y","Targets"]

)



tsne_df.loc[:, "Targets"] = tsne_df.Targets.astype(int)
tsne_df.head(10)
grid = sns.FacetGrid(tsne_df, hue="Targets", size=8)



grid.map(plt.scatter,"X","Y").add_legend()
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/taruntiwarihp/dataSets/master/winequality-red.csv")
df.head(10) # This dataset looks something like this: 
df.quality.unique()
# a mapping dictionary that maps the  quality values from 0 to 5.

quality_mapping = {

    3: 0,

    4: 1,

    5: 2,

    6: 3,

    7: 4,

    8: 5

}



# you can use the map function of pandas with any dictionary to covert the values in a given column to values in the dictionary

df.loc[:,"quality"] = df.quality.map(quality_mapping)
df.shape
# use sample with frac=1 to shuffle the dataframe 

# we reset the indices since they change after 

# shuffling the dataframe

df = df.sample(frac=1).reset_index(drop=True)



# top 1000 rows are selected for training 

df_train = df.head(1000)



# botton 599 values are selected for testing/validation

df_test = df.tail(599)
# import from scikit-learn

from sklearn import tree

from sklearn import metrics



# initialize decision tree classifier class with a max_depth of 3

clf = tree.DecisionTreeClassifier(max_depth=3)



# choose the columns you want to train on 

# these are the features for model

cols = ['fixed acidity',

        'volatile acidity',

        'citric acid',

        'residual sugar',

        'chlorides',

        'free sulfur dioxide',

        'total sulfur dioxide',

        'density',

        'pH',

        'sulphates',

        'alcohol'

]



# train the mode on the provided features and mapped quality from before

clf.fit(df_train[cols], df_train.quality)
# generate prediction on the training set 

train_predictions =  clf.predict(df_train[cols])



# generate prediction on the test set

test_predictions = clf.predict(df_test[cols])



# calculate the accuracy of prediction on training dataset

train_accuracy = metrics.accuracy_score(

    df_train.quality, train_predictions

)



# calculate the accuracy of prediction on test dataset

test_accuracy = metrics.accuracy_score(

    df_test.quality, test_predictions

)
train_accuracy
test_accuracy
# import  scikit-learn tree and metric

from sklearn import tree

from sklearn import metrics



# import matplotlib and seaborn for plotting

import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns



# this is our global size of label text on the plots

matplotlib.rc('xtick',labelsize=20)

matplotlib.rc('ytick',labelsize=20)



# This line ensures that the plot is displayed inside the notebook

%matplotlib inline



# initalize lists to store accuracies for training and test data

# we start with 50% accuracy

train_accuracies = [0.5]

test_accuracies = [0.5]
# iterate over a few depth values

for depth in range(1, 25):

    # init the model

    clf = tree.DecisionTreeClassifier(max_depth=depth)

    

    # fit the model on given feature

    clf.fit(df_train[cols],df_train.quality)

    

    # create training and test predictions

    train_predictions =  clf.predict(df_train[cols])

    test_predictions = clf.predict(df_test[cols])



    # calculate training and test accuracies

    train_accuracy = metrics.accuracy_score(

        df_train.quality, train_predictions

    )



    test_accuracy = metrics.accuracy_score(

        df_test.quality, test_predictions

    )

    

    # append accuracies

    train_accuracies.append(train_accuracy)

    test_accuracies.append(test_accuracy)
# create two plots using matplotlib and seaborn 

plt.figure(figsize=(10,5))

sns.set_style("whitegrid")

plt.plot(train_accuracies,label="Train Accuracy")

plt.plot(test_accuracies,label="Test Accuracy")

plt.legend(loc="upper left", prop={'size':15})

plt.xticks(range(0, 26, 5))

plt.xlabel("max_depth",size=20)

plt.ylabel("accuracy",size=20)

plt.show()

# import pandas and model_selection module of scikit-learn

import pandas as pd

from sklearn import model_selection



if __name__ == "__main__":

    # Training data is in a csv file called train.csv

    df = pd.read_csv("../input/novartis-data/Train.csv")

    

    # we create a new  column called kfold and fill it with -1

    df["kfold"] = -1

    

    # the next step is to randomize the rows of the data

    df = df.sample(frac=1).reset_index(drop=True)

    

    # initiate the kfold class from model_selection module

    kf = model_selection.KFold(n_splits=5)

    

    # fill the new kfold column

    for fold, (trn_, val_) in enumerate (kf.split(X=df)):

        df.loc[val_, 'kfold'] = fold

        

    # save the new csv with kflod column

    df.to_csv("train_folds.csv", index=False)
df['MULTIPLE_OFFENSE'].value_counts()
# import pandas and model_selection module of scikit-learn

import pandas as pd

from sklearn import model_selection



if __name__ == "__main__":

    # Training data is in a csv file called train.csv

    df = pd.read_csv("../input/novartis-data/Train.csv")

    

    # we create a new column called kfold and fill it with -1

    df["kfold"] = -1

    

    # the next step is to randomize the rows of the data

    df = df.sample(frac=1).reset_index(drop=True)

    

    # fetch target

    y = df.MULTIPLE_OFFENSE.values

    

    # initiate the kfold class from model_selection module

    kf = model_selection.StratifiedKFold(n_splits=5)

    

    # fill the newkfold column

    for f, (t_, v_) in enumerate(kf.split(X=df,y=y)):

        df.loc[v_, 'kfold'] = f

        

    # save the new csv with kfold column

    df.to_csv("train_sffolds.csv", index=False)
kf = pd.read_csv("train_folds.csv")
sf = pd.read_csv("train_sffolds.csv")
pd.crosstab(kf['MULTIPLE_OFFENSE'], kf['kfold'])
pd.crosstab(sf['MULTIPLE_OFFENSE'], sf['kfold'])
# Stratified k-fold for regression

import pandas as pd

import numpy as np



from sklearn import datasets

from sklearn import model_selection



def create_folds(data):

    # we create a new column called kfold and fill it with -1

    data['kfold'] = -1

    

    # the next step is to randomize the rows of the data

    data = data.sample(frac=1).reset_index(drop=True)

    

    # calculate the number of bins by Sturge's rule 

    # I take the floor of the value, you can also just round it

    num_bins = int(np.floor(1 + np.log2(len(data))))

    

    # bin targets

    data.loc[:, "bins"] = pd.cut(

        data["target"], bins=num_bins, labels=False

    )

    

    # initiate the kfold class from model_selection module

    kf = model_selection.StratifiedKFold(n_splits=5)

    

    # fill the new kfold column note that, instead of targets, we use bins!

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):

        data.loc[v_, 'kfold'] = f

    

    # drop the bins column

    data = data.drop("bins",axis=1)

    # return dataframe with folds

    

    return data



if __name__ == "__main__":

    # we create a sample dataset with 15000 samples and 100 features and 1 target

    X, y = datasets.make_regression(

        n_samples=15000, n_features=100, n_targets=1

    )

    

    # create a dataframe out of our numpy array

    df = pd.DataFrame(

        X,

        columns=[f"f_{i}" for i in range(X.shape[1])]

    )

    df.loc[:,"target"] = y

    

    # create folds

    df = create_folds(df)
df.head()
# Accuracy 

def accuracy(y_true, y_pred):

    """

    Function to calculate accuracy

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: accuracy score

    """

    # initialize a simple counter for correct prediction

    correct_counter = 0

    # loop over all elements of y_true and y_pred "together"

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:

            # if prediction is equal to truth, increase the counter

            correct_counter += 1

    

    # return accuracy

    # which is correct predictions over the number of samples

    return correct_counter / len(y_true)
# We can also caculate accuracy using scikit-learn and accuracy function

from sklearn import metrics

l1 = [0,1,1,1,0,0,0,1]

l2 = [0,1,0,1,0,1,0,0]

print(metrics.accuracy_score(l1,l2))

print(accuracy(l1,l2))
def true_positive(y_true, y_pred):

    """

    Function to calculate True Positives

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: number of true positives

    """

    # initialize

    tp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 1:

            tp += 1

    return tp



def true_negative(y_true, y_pred):

    """

    Function to calculate True Negative

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: number of true negatives

    """

    # initialize

    tn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 0:

            tn +=1

    return tn



def false_positive(y_true, y_pred):

    """

    Function to calculate False Positive

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: number of False positives

    """

    # initialize

    fp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 1:

            fp += 1

    return fp



def false_negative(y_true, y_pred):

    """

    Function to calculate False Negative

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: number of False negatives

    """

    # initialize

    fn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:

            fn += 1

    return fn
print(true_positive(l1,l2))

print(false_positive(l1,l2))

print(false_negative(l1,l2))

print(true_negative(l1,l2))
def accuracy_v2(y_true, y_pred):

    """

    Function to calculate accuracy using tp/tn/fp/fn

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: accuracy score

    """

    tp = true_positive(y_true, y_pred)

    fp = false_positive(y_true, y_pred)

    fn = false_negative(y_true, y_pred)

    tn = true_negative(y_true,y_pred)

    accuracy_score = (tp + tn) / (tp + tn + fp + fn)

    return accuracy_score
print(accuracy(l1,l2))

print(accuracy_v2(l1,l2))

print(metrics.accuracy_score(l1,l2))
def precision(y_true, y_pred):

    """

    Function to calculate precision

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: precision score

    """

    tp = true_positive(y_true, y_pred)

    fp = false_positive(y_true, y_pred)

    precision = tp/ (tp + fp)

    return precision
precision(l1,l2)
def recall(y_true, y_pred):

    """

    Function to calculate recall

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: recall score

    """

    tp = true_positive(y_true, y_pred)

    fn = false_negative(y_true, y_pred)

    recall  = tp / (tp + fn)

    return recall
recall(l1,l2)
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 

          1, 0, 0, 0, 0, 0, 0, 0, 1, 0]



y_pred = [0.02638412, 0.11114267, 0.31620708,

         0.0490937, 0.0191491, 0.17554844, 

         0.15952202, 0.03819563, 0.11639273, 

         0.079377, 0.08584789, 0.39095342, 

         0.27259048, 0.03447096, 0.04644807, 

         0.03543574, 0.18521942, 0.05934905, 

         0.61977213, 0.33056815]
precisions = []

recalls = []

# how we assumed these thresholds is a long story

thresholds = [0.0490937, 0.05934905, 0.079377, 

              0.08584789, 0.11114267, 0.11639273, 

              0.15952202, 0.17554844, 0.18521942, 

              0.27259048, 0.31620708, 0.33056815,

              0.39095342, 0.61977213]



# for every threshold, calculate predictions in library and append calculated precisions and recalls to their respective lists

for i in thresholds:

    temp_prediction = [1 if x >= i else 0 for x in y_pred]

    p = precision(y_true, temp_prediction)

    r = recall(y_true, temp_prediction)

    precisions.append(p)

    recalls.append(r)
precisions
recalls
plt.figure(figsize=(7, 7))

plt.plot(recalls, precisions)

plt.xlabel("Recall", fontsize = 15)

plt.ylabel("Precision", fontsize=15)
def f1(y_true, y_pred):

    """

    Function to calculate f1 score

    :param y_true: list of true values 

    :param y_pred: list of predicted values

    :return: f1 score

    """

    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    

    score = 2 * p * r / (p + r)

    

    return score
f1(l1,l2)
from sklearn import metrics



metrics.f1_score(l1, l2)
# TPR or True Positive Rate



def tpr (y_true, y_pred):

    """

    Function to calculate tpr

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: tpr/recall

    """

    return recall(y_true, y_pred)
# FPR or False Positive Rate



def fpr (y_true, y_pred):

    """

    Function to calculate fpr

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: fpr

    """

    fp = false_positive(y_true, y_pred)

    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp)
# empty lists to store tpr

# and fpr values

tpr_list = []

fpr_list = []



# actual targets

y_true = [0, 0, 0, 0, 1, 0, 1,

         0, 0, 1, 0, 1, 0, 0, 1]



# predicted probabilities of sample being 1

y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.5, 

         0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 

         0.85, 0.15, 0.99]



# handmade thresholds

thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,

              0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]



# loop over all thresholds

for thresh in thresholds:

    # calculate predictions for a given threshold

    temp_pred = [1 if x >= thresh else 0 for x in y_pred]

    # calculate tpr

    temp_tpr = tpr(y_true, temp_pred)

    # calculate fpr

    temp_fpr = fpr(y_true, temp_pred)

    #append tpr and fpr to lists

    tpr_list.append(temp_tpr)

    fpr_list.append(temp_fpr)
pd.DataFrame(data={'Thresholds':thresholds, 'TPR':tpr_list, 'FPR':fpr_list}) 
plt.figure(figsize=(7, 7))

plt.fill_between(fpr_list, tpr_list, alpha=0.4)

plt.xlim(0, 1.0)

plt.ylim(0, 1.0)

plt.xlabel('FPR', fontsize=15)

plt.ylabel('TPR',fontsize=15)

plt.show()
y_true = [0, 0, 0, 0, 1, 0, 1, 

         0, 0, 1, 0, 1, 0, 0, 1]

y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,

         0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 

         0.85, 0.15, 0.99]

metrics.roc_auc_score(y_true, y_pred)
# empty lists to store ture positive and false positive values

tp_list = []

fp_list = []



# actual targets

y_true = [0, 0, 0, 0, 1, 0, 1, 

         0, 0, 1, 0, 1, 0, 0, 1]



# predicted probabilities of a sample being 1

y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,

         0.9, 0.5, 0.3, 0.66, 0.3, 0.2,

          0.85, 0.15, 0.99]



# some handmade thresholds 

thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 

             0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]



# loop over all thresholds

for thresh in thresholds:

    # calculate predictions for a given threshold

    temp_pred = [1 if x >= thresh else 0 for x in y_pred]

    # calculate tp

    temp_tp = true_positive(y_true, temp_pred)

    # calculate fp

    temp_fp = false_positive(y_true, temp_pred)

    tp_list.append(float(temp_tp))

    fp_list.append(float(temp_fp))
pd.DataFrame(data={'Thresholds':thresholds, 'TP':tp_list, 'FP':fp_list}) 
def log_loss(y_true, y_proba):

    """

    Function to calculate log loss

    :param y_true: list of true values

    :param y_proba: list of probabilities for 1

    :return: overall log loss

    """

    # define an epsilon value 

    # this can also be an input

    # this value is used to clip probabilities 

    epsilon = 1e-15

    # initialize empty list to store 

    # individual losses

    loss=[]

    # loop over all true and predicted probability values

    for yt, yp in zip(y_true, y_proba):

        # adjust probability

        # 0 gets converted to 1e-15

        # 1 gets converted to 1-1e-15

        # Why? Think about it!

        yp = np.clip(yp, epsilon, 1 - epsilon)

        # calculate loss for one sample 

        temp_loss = - 1.0 * (

            yt * np.log(yp)

            + (1 - yt) * np.log(1 - yp)

        )

        # add to loss list

        loss.append(temp_loss)

    # return mean loss over all samples

    return np.mean(loss)
y_true = [0, 0, 0, 0, 1, 0, 1, 

         0, 0, 1, 0, 1, 0, 0, 1]



y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,

          0.9, 0.5, 0.3, 0.66, 0.3, 0.2,

          0.85, 0.15, 0.99]



log_loss(y_true, y_proba)
from sklearn import metrics



metrics.log_loss(y_true, y_proba)
def macro_precision(y_true, y_pred):

    """

    Function to calculate macro averaged precision

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: macro precision score

    """

    # find the number of classes by taking lenght of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # initialize precision to 0 

    precision = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate true positive for current class

        tp = true_positive(temp_true, temp_pred)

        

        # calculate false positive for current class

        fp = false_positive(temp_true, temp_pred)

        

        # calculate precision for current class

        temp_precision= tp / (tp + fp)

        

        # keep adding precision for all classes

        precision += temp_precision

        

    # calculate and retrun average precision over all classes

    precision /= num_classes

    return precision
def micro_precision(y_true, y_pred):

    """

    Function to calculate micro averaged precision

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: micro precision score

    """

    # find the number of classes by taking length of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # initialize tp and fp to 0

    tp = 0

    fp = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate true positive for current class and update overall tp

        tp += true_positive(temp_true, temp_pred)

        

        # calculate false positive for current class and update overall fp

        fp += false_positive(temp_true, temp_pred)

        

    # calculate precision for current class

    precision= tp / (tp + fp)

    return precision
from collections import Counter



def weighted_precision(y_true, y_pred):

    """

    Function to calculate weighted averaged precision

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: weighted precision score

    """

    # find the number of classes by taking lenght of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # create class:sample count dictionary

    # it looks something like this:

    # {0: 20, 1: 15, 2: 21}

    class_counts = Counter(y_true)

    

    # initialize precision to 0 

    precision = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate tp and fp for class

        tp = true_positive(temp_true, temp_pred)

        fp = false_positive(temp_true, temp_pred)

        

        # calculate precision for current class

        temp_precision= tp / (tp + fp)

        

        # multiply precision with count of samples in class

        weighted_precision = class_counts[class_] * temp_precision

        

        # add to overall precision

        precision += weighted_precision

        

    # calculate overall precision by dividing by total number of samples

    overall_precision = precision / len(y_true)

    

    return overall_precision
from sklearn import metrics



y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]



y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]



print(macro_precision(y_true, y_pred))



print(metrics.precision_score(y_true, y_pred, average="macro"))



print(micro_precision(y_true, y_pred))



print(metrics.precision_score(y_true, y_pred, average="micro"))



print(weighted_precision(y_true, y_pred))



print(metrics.precision_score(y_true, y_pred, average="weighted"))
def macro_recall(y_true, y_pred):

    """

    Function to calculate macro averaged recall

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: macro recall score

    """

    # find the number of classes by taking lenght of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # initialize recall to 0 

    recall = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate true positive for current class

        tp = true_positive(temp_true, temp_pred)

        

        # calculate false negative for current class

        fn = false_negative(temp_true, temp_pred)

        

        # calculate recall for current class

        temp_recall= tp / (tp + fn)

        

        # keep adding recall for all classes

        recall += temp_recall

        

    # calculate and retrun average recall over all classes

    recall /= num_classes

    return recall
def micro_recall(y_true, y_pred):

    """

    Function to calculate micro averaged recall

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: micro recall score

    """

    # find the number of classes by taking length of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # initialize tp and fn to 0

    tp = 0

    fn = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate true positive for current class and update overall tp

        tp += true_positive(temp_true, temp_pred)

        

        # calculate false negative for current class and update overall fp

        fn += false_negative(temp_true, temp_pred)

        

    # calculate recall for current class

    recall = tp / (tp + fn)

    return recall
from collections import Counter



def weighted_recall(y_true, y_pred):

    """

    Function to calculate weighted averaged recall score

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: weighted recall score

    """

    # find the number of classes by taking lenght of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # create class:sample count dictionary

    # it looks something like this:

    # {0: 20, 1: 15, 2: 21}

    class_counts = Counter(y_true)

    

    # initialize recall to 0 

    recall = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate tp and fn for class

        tp = true_positive(temp_true, temp_pred)

        fn = false_negative(temp_true, temp_pred)

        

        # calculate recall for current class

        temp_recall = tp / (tp + fn)

        

        # multiply recall with count of samples in class

        weighted_recall = class_counts[class_] * temp_recall

        

        # add to overall recall

        recall += weighted_recall

        

    # calculate overall recall by dividing by total number of samples

    overall_recall = recall / len(y_true)

    

    return overall_recall
from sklearn import metrics



y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]



y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]



print(macro_recall(y_true, y_pred))



print(metrics.recall_score(y_true, y_pred, average="macro"))



print(micro_recall(y_true, y_pred))



print(metrics.recall_score(y_true, y_pred, average="micro"))



print(weighted_recall(y_true, y_pred))



print(metrics.recall_score(y_true, y_pred, average="weighted"))
def weighted_f1(y_true, y_pred):

    """

    Function to calculate weighted averaged f1 score

    :param y_true: list of true values

    :param y_pred: list of predicted values

    :return: weighted f1 score

    """

    # find the number of classes by taking lenght of unique values in true list

    num_classes = len(np.unique(y_true))

    

    # create class:sample count dictionary

    # it looks something like this:

    # {0: 20, 1: 15, 2: 21}

    class_counts = Counter(y_true)

    

    # initialize f1 to 0 

    f1 = 0

    

    # loop over all classes

    for class_ in range(num_classes):

        

        # all classes except current are considered negative

        temp_true = [1 if p == class_ else 0 for p in y_true]

        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        

        # calculate precision and recall for class

        p = precision(temp_true, temp_pred)

        r = recall(temp_true, temp_pred)

        

        # calculate f1 of class

        if p + r != 0:

            temp_f1 = 2 * p * r / (p + r)

        else:

            temp_f1 = 0

        

        # multiply f1 with count of samples in class

        weighted_f1 = class_counts[class_] * temp_f1

        

        # add to f1 precision

        f1 += weighted_f1

        

    # calculate overall f1 by dividing by total number of samples

    overall_f1 = f1 / len(y_true)

    

    return overall_f1
from sklearn import metrics



y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]



y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]



print(weighted_f1(y_true, y_pred))



print(metrics.f1_score(y_true, y_pred, average="weighted"))
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics



# some targets

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]



# some predictions

y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]



# get confusion matrix from sklearn

cm = metrics.confusion_matrix(y_true, y_pred)



# plot using matplotlib and seaborn

plt.figure(figsize=(10,10))

cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,

                            as_cmap=True)

sns.set(font_scale=2.5)

sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)

plt.ylabel("Actual Labels", fontsize=20)

plt.xlabel("Predicted Label", fontsize=20)
def pk(y_true, y_pred, k):

    """

    This function calculates precision at k

    for a single sample 

    :param y_true: list of values, actual classes.

    :param y_pred: list of values, predicted classes.

    :return: precision at a given value k

    """

    # if k is 0, return 0. we should never have this as k is always >= 1

    if k == 0:

        return 0

    # we are interested only in top-k predictions

    y_pred = y_pred[:k]

    # convert predictions to set

    pred_set = set(y_pred)

    # convert actual values to set

    true_set = set(y_true)

    # find common values

    common_values = pred_set.intersection(true_set)

    # return length of common values over k

    return len(common_values) / len(y_pred[:k])
def apk(y_true, y_pred, k):

    """

    This function calculates average precision at k for single sample

    :param y_true: list of values, actual classes

    :param y_pred: list of values, predicted classes

    :return: average precision at a given value k

    """

    # initialize p@k list of values

    pk_values = []

    # loop over all k. from 1 to k + 1

    for i in range(1, k + 1):

        # calculate p@i and append to list

        pk_values.append(pk(y_true, y_pred, i))

    

    # if we have no values in the list, return 0

    if len(pk_values) == 0:

        return 0

    # else, we return the sum of list over length of list

    return sum(pk_values) / len(pk_values)
y_true = [

    [1, 2, 3],

    [0, 2],

    [1],

    [2, 3],

    [1, 0],

    []

]

y_pred = [

    [0, 1, 2],

    [1],

    [0, 2, 3],

    [2, 3, 4, 0],

    [0, 1, 2],

    [0]

]

for i in range(len(y_true)):

    for j in range(1, 4):

        print(

            f"""

            y_true={y_true[i]},

            y_pred={y_pred[i]},

            AP@{j}={apk(y_true[i], y_pred[i], k=j)}

            """

            )
def mapk(y_true, y_pred, k):

    """This function calculates mean avg precision at k for a single sample

    :param y_true: list of values, actual classes

    :param y_pred: list of values, predicted classes

    :return: mean av precision at a given value k

    """

    # initialize empty list for apk values

    apk_values = []

    # loop over all samples

    for i in range(len(y_true)):

        # store apk values for every sample

        apk_values.append(

            apk(y_true[i], y_pred[i], k=k)

        )

    # return mean of apk values list

    return sum(apk_values) / len(apk_values)
# Now we can calculate MAP@k for k=1, 2, 3 and 4 for the same list of lists.



y_true = [

    [1, 2, 3],

    [0, 2],

    [1],

    [2, 3],

    [1, 0],

    []

]

y_pred = [

    [0, 1, 2],

    [1],

    [0, 2, 3],

    [2, 3, 4, 0],

    [0, 1, 2],

    [0]

]



print(mapk(y_true, y_pred, k=1))



print(mapk(y_true, y_pred, k=2))



print(mapk(y_true, y_pred, k=3))
# taken from:

# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.

    This function computes the AP at k between two lists of imtes.

    Paramerters

    ----------

    actual : list

             A list of elements that are to be predicted (order doesn't matter)

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i,p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)



    if not actual:

        return 0.0



    return score / min(len(actual), k)
def mean_absolute_error(y_true, y_pred):

    """

    This function calculates me

    :param y_true: list of real numbers, true values

    :param y_pred: list of real numbers, predicted values

    :return: mean absolute error

    """

    # initialize error at 0

    error = 0

    # loop over all samples in the true and predicted list

    for yt, yp in zip(y_true, y_pred):

        # calculate absolute error and add to error

        error += np.abs(yt - yp)

    #return mean error

    return error / len(y_true)
def mean_squared_error(y_true, y_pred):

    """

    This function calculates mse

    :param y_true: list of real numbers, true values

    :param y_pred: list of real number, predicted values

    :return: mean squared error

    """

    # initialize error at 0

    error = 0 

    # loop over all samples in the true and predicted list

    for yt, yp in zip(y_true, y_pred):

        # calculate squared error

        # and add to error

        error += (yt - yp) ** 2

    # return mean error

    return error / len(y_true)
def mean_squared_log_error (y_true, y_pred):

    """

    This function calculates msle 

    :param y_true: list of real numbers, true values

    :param y_pred: list of real numbers, predicted values

    :return: mean squared logarithmic error

    """

    # initialize error at 0

    error = 0

    # loop over all samples in true and predicted list

    for yt, yp in zip(y_true, y_pred):

        # calculate squared log error and add to error

        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2

    # return mean error 

    return error / len(y_true) 
def mean_percentage_error(y_true, y_pred):

    """

    This function calculates mpe

    :param y_true: list of real numbers, true values

    :param y_pred: list of real numbers, predicted values

    :return: mean percentage error

    """

    # initialize error at 0

    error = 0 

    

    # loop over all samples in true and predicted list

    for yt, yp in zip(y_true, y_pred):

        # calculate percentage error and add to error 

        error += (yt - yp) / yt

        

    # return mean percentage error

    return error / len(y_true)
def mean_abs_percentage_error(y_true, y_pred):

    """

    This function calculates MAPE

    :param y_true: list of real numbers, true values

    :param y_pred: list of real numbers, predicted values

    :return: mean absolute percentage error

    """

    # initialize error at 0

    error = 0

    # loop over all samples in true and predicted list 

    for yt, yp in zip (y_true, y_pred):

        # calculate percentage error and add to error

        error += np.abs(yt - yp) / yt

    # return mean percentage error

    return error / len(y_true)
def r2(y_true, y_pred):

    """

    This function calculates r-squared score

    :param y_true: list of real numbers, true values

    :param y_pred: list of real numbers, predicted values

    :return: r2 score

    """

    # calculate the mean value of true values

    mean_true_value = np.mea(y_true)

    

    # initialize numerator with 0

    numerator = 0

    

    # initialize denominator with 0

    denominator = 0

    

    # loop over all true and predicted values

    for yt, yp in zip(y_true, y_pred):

        # update numerator 

        numerator += (yt - yp) ** 2

        # update denominator 

        denominator += (yt - mean_true_value) ** 2

    # calculate the ratio

    ratio = numerator / denominator

    # return 1 - ratio

    retrun (1 - ratio)
def mae_np(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))