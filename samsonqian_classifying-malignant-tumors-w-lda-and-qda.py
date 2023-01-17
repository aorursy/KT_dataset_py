import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



from scipy.stats import multivariate_normal  # Estimating Gaussian distribution
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
fig = plt.gcf()

fig.set_size_inches(8, 5)



sns.scatterplot(x="radius_mean", y="texture_mean", hue="diagnosis", palette=["red", "blue"], data=train)

plt.title("Texture Mean vs. Radius Mean")
# Standardize each feature before analysis

def standardize(train, test):

    """

    Standardizes training and testing set data based on the means and standard

    deviations of the training set. x = (x-µ)/σ

    

    :param train: training set

    :param test: testing set

    :returns: standardized training and testing set

    """

    for col in train.columns:

        if col == "diagnosis":

            continue

        mean = np.mean(train[col])

        std = np.std(train[col])

        train[col] = (train[col]-mean)/std  # standardize training data

        test[col] = (test[col]-mean)/std  # standardize testing data

    return train, test



train, test = standardize(train, test)
# Split data into two classes

benign = train[train["diagnosis"] == "B"]

malignant = train[train["diagnosis"] == "M"]
# Calculate Covariance Matrix for each class

def covariance_matrix(data):

    """

    Calculates the covariance matrix of all features for the dataset.

    

    :param data: dataset for covariance matrix

    :return: covariance matrix of data

    """

    df = data.drop(["diagnosis"], axis=1)

    for col in df.columns:

        df[col] = df[col] - np.mean(df[col])

    X = df.to_numpy()

    n = len(data)

    return (1/n)*np.dot(X.T, X)



C1 = covariance_matrix(benign)  # benign covariance matrix

C2 = covariance_matrix(malignant)  # malignant covariance matrix
# Overall Covariance Matrix with weighted average

n1 = len(benign)

n2 = len(malignant)

C = (n1*C1+n2*C2)/(n1+n2)

C  # joined covariance matrix
def _LDA(features):

    """

    Performs LDA classification on one data point, given array of its features.

    

    :param features: array of feature data

    :return: label prediction

    """

    p_benign = len(benign)/len(train)  # probability of benign 

    p_malignant = len(malignant)/len(train)  # probability of malignant

    

    mean_benign = np.array(benign.mean(axis=0))  # mean of benign

    mean_malignant = np.array(malignant.mean(axis=0))  # mean of malignant

    

    pdf_benign = multivariate_normal.pdf(features, mean_benign, C)  # conditional probability given benign

    pdf_malignant = multivariate_normal.pdf(features, mean_malignant, C)  # conditional probability given malignant

    if pdf_benign * p_benign > pdf_malignant * p_malignant:

        return "B"

    else:

        return "M"
X_train = train.drop(["diagnosis"], axis=1).to_numpy()

y_train = train["diagnosis"]

X_test = test.drop(["diagnosis"], axis=1).to_numpy()

y_test = test["diagnosis"]



def LDA(data):

    """

    Performs LDA classification on an array of data.

    

    :param data: array of data to make predictions on

    :return: array of predictions on data

    """

    preds = np.array([])

    for i in data:

        preds = np.append(preds, _LDA(i))

    return preds



LDA_preds_train = LDA(X_train)  # LDA predictions for training set

print("LDA Training Accuracy: ", np.sum(LDA_preds_train==y_train)/len(train))

print("LDA Training Error: ", np.sum(LDA_preds_train!=y_train)/len(train))



LDA_preds_test = LDA(X_test)   # LDA predictions for testing set

print("LDA Testing Accuracy: ", np.sum(LDA_preds_test==y_test)/len(test))

print("LDA Testing Error: ", np.sum(LDA_preds_test!=y_test)/len(test))
def _QDA(features):

    """

    Performs QDA classification on one data point, given array of its features.

    

    :param features: array of feature data

    :return: label prediction

    """

    p_benign = len(benign)/len(train)  # probability of benign 

    p_malignant = len(malignant)/len(train)  # probability of malignant

    

    mean_benign = np.array(benign.mean(axis=0))  # mean of benign

    mean_malignant = np.array(malignant.mean(axis=0))  # mean of malignant

    

    pdf_benign = multivariate_normal.pdf(features, mean_benign, C1)  # conditional probability of benign

    pdf_malignant = multivariate_normal.pdf(features, mean_malignant, C2)  # conditional probability of malignant

    if pdf_benign * p_benign > pdf_malignant * p_malignant:

        return "B"

    else:

        return "M"
def QDA(data):

    """

    Performs QDA classification on an array of data.

    

    :param data: array of data to make predictions on

    :return: array of predictions on data

    """

    preds = np.array([])

    for i in data:

        preds = np.append(preds, _QDA(i))

    return preds



QDA_preds_train = QDA(X_train)  # LDA predictions for training set

print("QDA Training Accuracy: ", np.sum(QDA_preds_train==y_train)/len(train))

print("QDA Training Error: ", np.sum(QDA_preds_train!=y_train)/len(train))



QDA_preds_test = QDA(X_test)   # LDA predictions for testing set

print("QDA Testing Accuracy: ", np.sum(QDA_preds_test==y_test)/len(test))

print("QDA Testing Error: ", np.sum(QDA_preds_test!=y_test)/len(test))