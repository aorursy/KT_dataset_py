from string import punctuation, digits
import numpy as np
import random

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.
    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    return max(0, 1 - label * (np.sum(feature_vector * theta) + theta_0))

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    return np.maximum(0, 1 - labels*(np.sum(feature_matrix * theta, axis = 1) + theta_0)).mean()

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    new_theta, new_theta_0 = current_theta.copy(), current_theta_0
    
    if label * (np.sum(feature_vector * current_theta) + current_theta_0) <= 0 :
        new_theta += label * feature_vector
        new_theta_0 += label
        
    return (new_theta, new_theta_0)

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta, theta_0 = np.zeros((feature_matrix.shape[1],)), 0
    
    for _ in range(T):
        
        for i in get_order(feature_matrix.shape[0]):
            
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_0)
            
    return (theta, theta_0)

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta, theta_0 = np.zeros((feature_matrix.shape[1],)), 0
    c_theta, c_theta_0 = np.zeros((feature_matrix.shape[1],)), 0
    
    for i in range(T):
        
        for i in get_order(feature_matrix.shape[0]):
            
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_0)
            c_theta, c_theta_0 = c_theta + theta, c_theta_0 + theta_0
            
    n_samples = T * feature_matrix.shape[0]
    
    return c_theta / n_samples, c_theta_0 / n_samples
def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.
    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """ 
    if label * (np.sum(feature_vector * current_theta) + current_theta_0) <= 1 :
        
        new_theta = (1 - eta*L)*current_theta  +  eta*label*feature_vector
        new_theta_0 = current_theta_0 + eta*label 
        
    else:
        
        new_theta = (1 - eta*L)*current_theta
        new_theta_0 = current_theta_0
        
    return (new_theta, new_theta_0)
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.
    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta, theta_0 = np.zeros((feature_matrix.shape[1],)), 0
    t = 0
    
    for _ in range(T):
        
        for i in range(feature_matrix.shape[0]):
            
            t = t + 1
            eta = 1/np.sqrt(t)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i,:], labels[i], L, eta, theta, theta_0)
            
    return (theta, theta_0)
import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. 
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)


#-------------------------------------------------------------------------------



toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

T = 10
L = 0.2

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)



def plot_toy_results(algo_name, thetas):
    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    print('theta_0 for', algo_name, 'is', str(thetas[1]))
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    return ((np.sum(feature_matrix*theta, axis=1) + theta_0) > 0)*2 - 1
def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.
    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)
    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predict_labels = classify(train_feature_matrix, theta, theta_0)
    val_predict_labels = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predict_labels, train_labels)
    val_accuracy = accuracy(val_predict_labels, val_labels)
    
    return (train_accuracy, val_accuracy)
T = 10
L = 0.01

pct_train_accuracy, pct_val_accuracy = p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))


avg_peg_train_accuracy, avg_peg_val_accuracy = p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = utils.tune_perceptron(Ts, *data)
pct_best_acc, pct_best_T = np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(pct_best_acc, pct_best_T))
pct_best_results = (pct_best_acc, (pct_best_T,))
print("Perceptron Best Result:", pct_best_results)

avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
avg_pct_best_acc, avg_pct_best_T = np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(avg_pct_best_acc, avg_pct_best_T))
avg_pct_best_results = (avg_pct_best_acc, (avg_pct_best_T,))
print("Average Perceptron Best Result:", avg_pct_best_results)

# fix values for L and T while tuning Pegasos T and L, respective

fix_L = 0.01

peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
peg_best_T_acc, peg_best_T  = np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]
print('best = {:.4f}, T={:.4f}'.format(peg_best_T_acc, peg_best_T))

peg_tune_results_L = utils.tune_pegasos_L(peg_best_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
peg_best_L, peg_best_acc = Ls[np.argmax(peg_tune_results_L[1])], np.max(peg_tune_results_L[1])
print('best = {:.4f}, L={:.4f}'.format(peg_best_acc, peg_best_L))

peg_best_results = (peg_best_acc, (peg_best_T, peg_best_L))
print("Pegasos Best Result:", peg_best_results)

# Consoloidate Results
methods = [ ("Perceptron", p1.perceptron), 
            ("Average Perceptron", p1.average_perceptron), 
            ("Pegasos", p1.pegasos) ]

training_results = [    pct_best_results,
                        avg_pct_best_results,
                        peg_best_results]

utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)
#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

best_result_i = max(range(len(methods)), key=lambda i:training_results[i])
best_method = methods[best_result_i]
best_training_result = training_results[best_result_i]

print("The Best Method Was: {}, Accuracy = {}, Parameters = {}".format(best_method[0], *best_training_result))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

best_theta, best_theta_0 = best_method[1](train_bow_features, train_labels, *best_training_result[1])
print("Best Theta: ", best_theta)
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Positive Word Features")
print(sorted_word_features[:10])
print("Less Explanatory Positive Word Features")
print(sorted_word_features[-10:])