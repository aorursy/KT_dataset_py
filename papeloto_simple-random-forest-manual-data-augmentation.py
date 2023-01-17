import numpy as np

import pandas as pd



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
def get_data():

    """ Returns the data splitted into the images and their labels. """

    train_dataset = pd.read_csv('../input/train.csv')

    

    X = train_dataset.iloc[:, 1:].values / 255

    y = train_dataset.iloc[:, 0].values

    return X, y





def compute_score(X, y, model):

    """ Computes the cross val score of the model. """

    scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)

    print('Accuracy of', type(model).__name__)

    print('Mean:', np.mean(scores))

    print('STD:', np.std(scores))





def shift_images(digits, dx, dy):

    """ Shift the input images by dx and dy pixels. """

    shifted_digits = np.empty(digits.shape)

    for i in range(len(digits)):

        shifted_digits[i] =  np.roll(digits[i], (dx, dy))

        

    return shifted_digits



def augment_data(X, y):

    """ Augments the data by creating shifted copies one pixel to all directions. """

    X = np.vstack([X, shift_images(X, 1, 0), shift_images(X, -1, 0), shift_images(X, 0, 1), shift_images(X, 0, -1)])

    y = np.tile(y, 5)

    

    shuffle_idx = np.random.permutation(len(X))

    X = X[shuffle_idx]

    y = y[shuffle_idx]

    

    return X, y



def sumbit_predictions(model):

    """ Outputs the predictions for the competition. """

    test_set = pd.read_csv('../input/test.csv').values / 255

    predictions = model.predict(test_set)

    submission = pd.DataFrame({'ImageID': range(1, len(predictions) + 1), 'Label': predictions})

    submission.to_csv('submission.csv', index=False)
digits, labels = get_data()



forest = RandomForestClassifier()

compute_score(digits, labels, forest)
aug_digits, aug_labels = augment_data(digits, labels)

del digits, labels



compute_score(aug_digits, aug_labels, forest)
forest.fit(aug_digits, aug_labels)

sumbit_predictions(forest)