# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression # Logistics Regression Model.

import matplotlib.pyplot as matty #For plotting.

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def readTrainingDataset():

    

    # Reading training dataset into the software using pandas.

    data = pd.read_csv('../input/train.csv');

    

    print(data.shape);

    column_list = data.columns;

    

    # Check mean, max, min, count in the given dataset.

    print(data.describe());

    

    # find features which have a missing value. 

    MISSING_FEATURE_LIST = list(data.columns[data.isnull().any()]);

    

    # showing missing feature list.     

    print("Is their any data missing from any rows ?");

    print("Yes" if len(MISSING_FEATURE_LIST) else "No");

    print(MISSING_FEATURE_LIST);

    

    # Total rows in the dataset.    

    TOTAL_VALUES = data.shape[0];

    print(TOTAL_VALUES);

    

    # displaying % of values unique in each column with respect to total values.   

    print("Column Name  Total Unique Values %percentage");

    for column in column_list:

        total_unique_values = len(data[column].value_counts());

        print(column, total_unique_values, total_unique_values/float(TOTAL_VALUES));

        

    return data;



data = readTrainingDataset();
# Plot Age Feature for Further Division

def plotAge(data):

    data['Age'].hist(cumulative=True, normed=1);

    matty.title("Age Histogram");

    matty.xlabel("Age");

    matty.ylabel("Frequencies");

    matty.show();



plotAge(data);
# Replace missing value in Feature Age with mean of Age.

def replaceNanWithAgeMean(data):

    AGE_MEAN = 29.699118;

    data.loc[data['Age'].isnull(),'Age'] = AGE_MEAN;
replaceNanWithAgeMean(data);
def createAgeCategoryFeature(data):

    # bins in which we have to divide out dataset.

    bins = [0,13,19,50,100];

    # 0 represents Child.

    # 1 represents Teenage.

    # 2 represents Adult.

    # 3 represents Old.    

    category_name = [0, 1, 2, 3];

    data['AgeCategory'] = pd.cut(data['Age'], bins, labels=category_name);
createAgeCategoryFeature(data);
def dropFeature(data, DROP_COLUMN_LIST):

    for column in DROP_COLUMN_LIST:

        data.drop([column], axis = 1, inplace = True);
EXTRA_FEATURES = ['Fare', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Embarked'];

dropFeature(data, EXTRA_FEATURES);
print(data.head());
# Use binary values to represent gender in Feature sex.

# Use 1 to represent Male and 0 to represent Female.

data['Sex'] = data['Sex'].apply(lambda value: int(value == 'male'));
print(data.head());
# Generating training and cross_validation dataset using pandas dataframe.

def getTrainAndCvDataset(data):

    # Shuffling dataset for randomnly selecting training and cv dataset.     

    data.sample(frac=1).reset_index(drop=True)

    TRAINING_DATA_SIZE = int(0.70*len(data));

    train_data = data[:TRAINING_DATA_SIZE];

    cv_data = data[TRAINING_DATA_SIZE:];

    return train_data, cv_data;
training_data, cross_validation_data = getTrainAndCvDataset(data);
print(len(data));

print(len(training_data));

print(len(cross_validation_data));
# pandas dataframe to numpy matrix for further computation.

train_data = training_data.as_matrix();

cv_data = cross_validation_data.as_matrix();
# return model trained over given dataset.

# dataset is a n*m matrix in which columns 1 contains class variable and 

# remaining columns contains features variable.

def getModelLogisticsRegression(dataset, regularization_parameter = 1):

    inverse = 1/float(regularization_parameter);

    model = LogisticRegression(C=inverse);

    model.fit(dataset[:,1:], list(dataset[:,0]));

    return model;
# train_data and cv_data are n*m matrix.

def trainModelLogisticsRegression(train_data, cv_data, regularization_parameter = 1):

    model = getModelLogisticsRegression(train_data, regularization_parameter);

    prediction = list(model.predict(cv_data[:,1:]));

    actual = list(cv_data[:,0]);

    return prediction, actual;
# prediction contain list of values generated from our trained model.

# actual is a list of actual_values which are helpful in finding the strength of our model.

def computeStrengthOfModel(prediction, actual):

    confusion_matrix = np.array(0).repeat(4).reshape(2,2);

    for index, predicted_value in enumerate(prediction):

        confusion_matrix[int(actual[index])][int(predicted_value)] = confusion_matrix[int(actual[index])][int(predicted_value)] + 1;

    

    total_travellers = len(prediction);

    recall = 0.0;

    precision = 0.0;

    accuracy = 0.0;

    

    for index, row in enumerate(confusion_matrix):

        accuracy = accuracy + confusion_matrix[index][index];

    accuracy = accuracy/float(total_travellers);

    

    precision = confusion_matrix[1][1]/float(sum(confusion_matrix[:,1]));

    recall = confusion_matrix[1][1]/float(sum(confusion_matrix[1]));

    

    return [precision, recall, accuracy];
# Used for plotting line plot.

def plotData(x_axis_dataset, y_axis_dataset, x_axis_label, y_axis_label, plot_title, plot_style="darkgrid"):

    sns.set_style(plot_style);

    matty.plot(x_axis_dataset, y_axis_dataset);

    matty.xlabel(x_axis_label);

    matty.ylabel(y_axis_label);

    matty.title(plot_title);

    matty.show();
# Use to generate Regularization v/s Accuracy plot.

# This is useful for determining best alpha value for a particular model. 

def accuracyVsRegularizationPlot(train_data, cv_data, UPPER_LIMIT = 50):

    accuracy_list = [];

    regularization_values = range(1,UPPER_LIMIT);

    for regularization in regularization_values:

        prediction, actual = trainModelLogisticsRegression(train_data, cv_data, regularization);

        accuracy_list.append(computeStrengthOfModel(prediction, actual)[2]);

    plotData(regularization_values, accuracy_list, "Regularization", "Accuracy", "Regularization v/s Accuracy");
accuracyVsRegularizationPlot(train_data, cv_data);
accuracyVsRegularizationPlot(train_data, cv_data, UPPER_LIMIT=15);
# Use to generate preprocessed test dataset.

def getPreprocessedTestDataset():

    data = pd.read_csv("../input/test.csv");

    replaceNanWithAgeMean(data);

    createAgeCategoryFeature(data);

    EXTRA_FEATURES = ['Fare', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Embarked'];

    dropFeature(data, EXTRA_FEATURES);

    data['Sex'] = data['Sex'].apply(lambda value: int(value == 'male'));

    return data;



test_data = getPreprocessedTestDataset();
model = getModelLogisticsRegression(train_data, regularization_parameter = 10);

predicted_test_values = list(model.predict(test_data));
survival_test_data = pd.read_csv('../input/genderclassmodel.csv');

actual_test_values = list(survival_test_data['Survived']);
precision, recall, accuracy = computeStrengthOfModel(predicted_test_values, actual_test_values);

print("Precision :-",precision);

print("Recall :-",recall)

print("Accuracy :-",accuracy)