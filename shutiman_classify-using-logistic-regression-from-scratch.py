def train_with_file(data_file,iters):

    import pandas as pd

    import numpy as np

    """Trains a logisitc regression classifier.

    Args:

    data_file: a path to a csv file containing training data, without headers.

    iters: the number of iterations to use when training the classifier



    Returns:

    weights: a column vector (1d numpy array) containing the weights learned in your classifier.

    normalization_params: a dict mapping column names to (min, max) values from the training set



    """



    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",

               "marital_status", "occupation", "relationship", "race", "gender",

               "capital_gain", "capital_loss", "hours_per_week", "native_country",

               "income_bracket"]



    LABEL_COLUMN = "Label"



    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",

                           "relationship", "race", "gender","native_country"]



    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",

                          "hours_per_week"]

    

    features = pd.read_csv(data_file,  names = COLUMNS, skipinitialspace = True)

    features = features.dropna(how="any",axis = 0)

    

    #one hot + categorical data

    

    features[LABEL_COLUMN] = (features["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    

    Xtrain = features.drop(['income_bracket','Label'], axis=1)

    ytrain = features['Label']

    

    #Categorical tratament 

    

    for col in CATEGORICAL_COLUMNS:

        Xtrain = pd.concat([Xtrain, pd.get_dummies(Xtrain[col], prefix=col, prefix_sep=':')], axis=1)

        Xtrain.drop(col, axis=1, inplace=True)

    

    # Normalization

    

    fmean = np.mean(Xtrain)

    frange = np.amax(Xtrain) - np.amin(Xtrain)



    #Vector Subtraction

    Xtrain -= fmean



    #Vector Division

    Xtrain /= frange



    normalization_params = [fmean,frange]

    

    # train :3

    

    N = len(features)

    weights = [0] * Xtrain.shape[1]

    lr = 0.01





    for i in range(iters):

    #1 - Get Predictions

        labels = 1 / (1 + np.exp(-np.dot(Xtrain, weights) ))



        gradient = np.dot(Xtrain.T,  labels - ytrain)



    # Take the average cost derivative for each feature

        gradient = (gradient/N)*lr



    # - Subtract from our weights to minimize cost

        weights -= gradient

    

    return weights,normalization_params







def classify(data_file, weights, normalization_params):

    import pandas as pd

    import numpy as np

 

    """

Classifies data based on the supplied logistic regression weights.



  Args:

    data_file: a path to a csv file containing test data, without headers.

    weights: a column vectors containing the weights learned during training.

    normalization_params: a dict mapping column names to (min, max) values from the training set



  Returns:

    a column vector containing either a 1 or a 0 for each row in data_file

"""    

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",

               "marital_status", "occupation", "relationship", "race", "gender",

               "capital_gain", "capital_loss", "hours_per_week", "native_country",

               "income_bracket"]



    LABEL_COLUMN = "Label"



    CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",

                           "relationship", "race", "gender","native_country"]



    CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",

                          "hours_per_week"]



    features = pd.read_csv(data_file,  names = COLUMNS, skipinitialspace = True, skiprows=1)

    features = features.dropna(how="any",axis = 0)

    

    #one hot + categorical data

    

    features[LABEL_COLUMN] = (features["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    

    Xtrain = features.drop(['income_bracket','Label'], axis=1)

    ytrain = features['Label']

    

    for col in CATEGORICAL_COLUMNS:

        Xtrain = pd.concat([Xtrain, pd.get_dummies(Xtrain[col], prefix=col, prefix_sep=':')], axis=1)

        Xtrain.drop(col, axis=1, inplace=True)

    

    # Normalization

    

    fmean,frange = normalization_params 



    #Vector Subtraction

    Xtrain -= fmean



    #Vector Division

    Xtrain /= frange

    

    labels = 1 / (1 + np.exp(-np.dot(Xtrain, weights[:Xtrain.shape[1]]) ))

    labels = np.round(labels,0)

    

    

    return labels,ytrain
def accuracy(predicted_labels, actual_labels):

    import pandas as pd

    import numpy as np

    diff = predicted_labels - actual_labels

    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))



weights, normalization_params = train_with_file("../input/adult-training.csv",1000)



labels,ytrain = classify("../input/adult-test.csv", weights, normalization_params)



print('The accuracy to a test dataset is:' , accuracy(labels,ytrain))


