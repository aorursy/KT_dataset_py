# porto_seguro_insur.py

#  Assumes python vers. 3.6

# __author__ = 'mizio'



import csv as csv

import numpy as np

import pandas as pd

import pylab as plt

from fancyimpute import MICE

import random

from sklearn.model_selection import cross_val_score

import datetime

import tensorflow as tf





class PortoSeguroInsur:

    def __init__(self):

        self.df = PortoSeguroInsur.df

        self.df_test = PortoSeguroInsur.df_test

        self.df_submission = PortoSeguroInsur.df_submission

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')





    # Load data into Pandas DataFrame

    # For .read_csv, always use header=0 when you know row 0 is the header row

    df = pd.read_csv('../input/train.csv', header=0)

    df_test = pd.read_csv('../input/test.csv', header=0)

    df_submission = pd.read_csv('../input/sample_submission.csv', header=0)



    @staticmethod

    def features_with_null_logical(df, axis=1):

        row_length = len(df._get_axis(0))

        # Axis to count non null values in. aggregate_axis=0 implies counting for every feature

        aggregate_axis = 1 - axis

        features_non_null_series = df.count(axis=aggregate_axis)

        # Whenever count() differs from row_length it implies a null value exists in feature column and a False in mask

        mask = row_length == features_non_null_series

        return mask



    def missing_values_in_dataframe(self, df):

        mask = self.features_with_null_logical(df)

        print(df[mask[mask == 0].index.values].isnull().sum())

        print('\n')



    @staticmethod

    def extract_numerical_features(df):

        df = df.copy()

        df = df.copy()

        non_numerical_feature_names = df.columns[np.where(PortoSeguroInsur.numerical_feature_logical_incl_hidden_num(

            df) == 0)]

        return non_numerical_feature_names



    @staticmethod

    def extract_non_numerical_features(df):

        df = df.copy()

        non_numerical_feature_names = df.columns[np.where(PortoSeguroInsur.numerical_feature_logical_incl_hidden_num(

            df))]

        return non_numerical_feature_names



    @staticmethod

    def numerical_feature_logical_incl_hidden_num(df):

        logical_of_non_numeric_features = np.zeros(df.columns.shape[0], dtype=int)

        for ite in np.arange(0, df.columns.shape[0]):

            try:

                str(df[df.columns[ite]][0]) + df[df.columns[ite]][0]

                logical_of_non_numeric_features[ite] = True

            except TypeError:

                print('Oops')

        return logical_of_non_numeric_features



    def clean_data(self, df, is_train_data=1):

        df = df.copy()

        if df.isnull().sum().sum() > 0:

            if is_train_data:

                df = df.dropna()

            else:

                df = df.dropna(1)

        return df



    def reformat_data(self, labels, num_labels):

        # Map labels/target value to one-hot-encoded frame. None is same as implying newaxis() just replicating array

        # if num_labels > 2:

        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float64)

        return labels



    def accuracy(self, predictions, labels):

        # Sum the number of cases where the predictions are correct and divide by the number of predictions

        number_of_correct_predictions = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

        return 100*number_of_correct_predictions/predictions.shape[0]



    def linear_model(self, input_vector, weight_matrix, bias_vector):

        # f(x) = Wx + b

        # W is the weight matrix with elements w_ij

        # x is the input vector

        # b is the bias vector

        # In the machine learning literature f(x) is called an activation

        return tf.matmul(input_vector, weight_matrix) + bias_vector



    def activation_out(self, logit):

        return self.activation(logit, switch_var=0)



    def activation_hidden(self, logit):

        return self.activation(logit, switch_var=0)



    def activation(self, logit, switch_var=0):

        # Also called the activation function

        if switch_var == 0:

            # Logistic sigmoid function.

            # sigma(a) = 1/(1+exp(-a))

            return tf.nn.sigmoid(logit)

        elif switch_var == 1:

            # Using Rectifier as activation function. Rectified linear unit (ReLU). Compared to sigmoid or other

            # activation functions it allows for faster and effective training of neural architectures.

            # f(x) = max(x,0)

            return tf.nn.relu(logit)

        else:

            # Softmax function.

            # S(y_i) = e^y_i/(Sum_j e^y_j)

            return tf.nn.softmax(logit)





def main():

    porto_seguro_insur = PortoSeguroInsur()

    df = porto_seguro_insur.df.copy()

    df_test = porto_seguro_insur.df_test.copy()

    df_submission = porto_seguro_insur.df_submission.copy()



    df = df.replace(-1, np.NaN)

    df_test = df_test.replace(-1, np.NaN)



    print(df.shape)

    print(df_test.shape)

    # Clean data for NaN

    df = porto_seguro_insur.clean_data(df)

    df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)

    print('df_test.shape: %s' % str(df_test.shape))  # (892816, 46)

    # df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)

    id_df_test = df_test['id']  # Submission column

    print("After dropping NaN")

    print(df.shape)

    print(df_test.shape)



    is_explore_data = 1

    if is_explore_data:

        # Overview of train data

        print('\n TRAINING DATA:----------------------------------------------- \n')

        print(df.head(3))

        print('\n')

        print(df.info())

        print('\n')

        print(df.describe())

        print('\n')

        print(df.dtypes)

        print(df.get_dtype_counts())



        # missing_values

        print('All df set missing values')

        porto_seguro_insur.missing_values_in_dataframe(df)



        print('Uniques')

        uniques_in_id = np.unique(df.id.values).shape[0]

        print(uniques_in_id)

        print('uniques_in_id == df.shape[0]')

        print(uniques_in_id == df.shape[0])



        # Overview of sample_submission format

        print('\n sample_submission \n')

        print(df_submission.head(3))

        print('\n')

        print(df_submission.info())

        print('\n')



    is_prepare_data = 1

    if is_prepare_data:

        df_test_num_features = porto_seguro_insur.extract_numerical_features(df_test)

        df_y = df.loc[:, ['target']]

        df = df.loc[:, df_test_num_features]



    is_prediction = 1

    if is_prediction:

        # Subset the data to make it run faster

        # (595212, 59)

        # (892816, 58)

        # After dropping NaN

        # (124931, 59)

        # (186567, 58)

        subset_size = 10000



        num_labels = np.unique(df_y.loc[:subset_size, 'target'].values).shape[0]

        num_columns = df[(df.columns[(df.columns != 'target') & (df.columns != 'id')])].shape[1]

        # Reformat datasets

        x_train = df.loc[:subset_size, (df.columns[(df.columns != 'target') & (df.columns != 'id')])].values

        y_train = df_y.loc[:subset_size, 'target'].values

        # We only need to one-hot-encode our labels since otherwise they will not match the dimension of the

        # logits in our later computation.

        y_train = porto_seguro_insur.reformat_data(y_train, num_labels=num_labels)



        # Todo: we need testdata with labels to benchmark test results.

        # Hack. Use subset of training data (not used in training model) as test data, since it has a label/target value

        # In case there are duplicates in training data it may imply that test results are too good, when using

        # a subset of training data for test.

        # Todo: do cross-validation instead of only one subset testing as below in x_val

        # Validation data is a subset of training data.

        x_val = df.loc[subset_size:2*subset_size, (df.columns[(df.columns != 'target') & (df.columns != 'id')])].values

        y_val = df_y.loc[subset_size:2*subset_size, 'target'].values

        y_val = porto_seguro_insur.reformat_data(y_val, num_labels=num_labels)

        # Test data.

        x_test = df_test.loc[:, (df_test.columns[(df_test.columns != 'id')])].values



        # Todo: we need validation data with labels to perform crossvalidation while training and get a better result.



        # Tensorflow uses a dataflow graph to represent your computations in terms of dependencies.

        graph = tf.Graph()

        with graph.as_default():

            # Load training and test data into constants that are attached to the graph

            tf_train = tf.constant(x_train)

            tf_train_labels = tf.constant(y_train)

            tf_val = tf.constant(x_val)

            tf_test = tf.constant(x_test)



            # As in a neural network the goal is to compute the cross-entropy D(S(w,x), L)

            # x, input training data

            # w_ij, are elements of the weight matrix

            # L, labels or target values of the training data (classification problem)

            # S(), is softmax function

            # Do the Multinomial Logistic Classification

            # step 1.

            # Compute y from the linear model y = WX + b, where b is the bias and W is randomly chosen on

            # Gaussian distribution and bias is set to zero. The result is called the logit.

            # step 2.

            # Compute the softmax function S(Y) which gives distribution

            # step 3.

            # Compute cross-entropy D(S, L) = - Sum_i L_i*log(S_i)

            # step 4.

            # Compute loss L = 1/N * D(S, L)

            # step 5.

            # Use gradient-descent to find minimum of loss wrt. w and b by minimizing L(w,b).

            # Update your weight and bias until minimum of loss function is reached

            # w_i -> w_i - alpha*delta_w L

            # b -> b - alpha*delta_b L

            # OBS. step 5 is faster optimized if you have transformed the data to have zero mean and equal variance

            # mu(x_i) = 0

            # sigma(x_i) = sigma(x_j)

            # This transformation makes it a well conditioned problem.



            # Make a 2-layer Neural network (count number of layers of adaptive weights) with num_columns nodes

            # in hidden layer.

            # Initialize weights on truncated normal distribution. Initialize biases to zero.

            # For every input vector corresponding to one sample we have D features s.t.

            # a_j = Sum_i^D w^(1)_ji x_i + w^(1)_j0 , where index j is the number of nodes in the first hidden layer

            # and it runs j=1,...,M

            # Vectorizing makes the notation more compact

            #     | --- x_1 --- |

            #     | --- x_2 --- |

            # X = | --- ..  --- |

            #     | --- x_N --- |

            # where each x is now a sample vector of dimension (1 x D) and where N is the number of samples.

            # Similarly, define a tiling of N weight matrices w,

            #     | --- w --- |

            #     | --- w --- |

            # W = | --- ..--- |

            #     | --- w --- |

            # where each w is now a matrix of dimension (M x D)

            # We now form the tensor product between W and X but we need to transpose X as x.T to get (M x D).(D x 1)

            # multiplication,

            #       |  w.(x_1.T) |

            #       |  w.(x_2.T) |

            # W.X = |  ..        |

            #       |  w.(x_N.T) |

            # with W.X having dimensions (M*N x 1).

            # Additionally, define a tiling of N bias vectors b that each are of dimension (M x 1),

            #     |  b  |

            #     |  b  |

            # B = |  .. |

            #     |  b  |

            # with B having dimensions (M*N x 1).

            # Finally, the activation is a (M*N x 1) vector given as A = W.X + B.

            # Next, this is passed to an activation function like a simoid and then inserted in second layer of the NN.

            # Let Z = sigmoid(A)

            # Let C be the activation of the second layer,

            # C = W^(2).Z + B^(2)

            # where W^(2) is the tiling N second layer weight matrices w^(2) each with dimension (K x M). K is the

            # number of outputs in the classification. The dimension of C is (K x N).

            # Lastly, apply the sigmoid function to get the predictions

            # P = sigmoid( C )

            # which has dimensions (K x N) and is as expected an output vector (K x 1) for every N samples in our

            # dataset. The output (K x 1)-vector is in a one-hot-encoded form.

			

			# Choose number of nodes > than number of features.

            M_nodes = x_train.shape[1]

            weights_1_layer = tf.Variable(tf.truncated_normal([num_columns, M_nodes], dtype=np.float64))

            biases_1_layer = tf.Variable(tf.zeros([M_nodes], dtype=np.float64))

            weights_2_layer = tf.Variable(tf.truncated_normal([M_nodes, num_labels], dtype=np.float64))

            biases_2_layer = tf.Variable(tf.zeros([num_labels], dtype=np.float64))



            # Logits and loss function.

            logits_hidden_1_layer = porto_seguro_insur.linear_model(tf_train, weights_1_layer, biases_1_layer)

            # Output unit activations of first layer

            a_1_layer = porto_seguro_insur.activation_hidden(logits_hidden_1_layer)

            logits_2_layer = porto_seguro_insur.linear_model(a_1_layer, weights_2_layer, biases_2_layer)

            switch_var = 0

            if switch_var == 1:

                loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,

                                                                                       logits=logits_2_layer))

            else:

                loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels,

                                                                                       logits=logits_2_layer))



            # Find minimum of loss function using gradient-descent.

            optimized_weights_and_bias = tf.train.GradientDescentOptimizer(0.5).minimize(loss=loss_function)



            # Accuracy variables using the initial values for weights and bias of our linear model.

            train_prediction = porto_seguro_insur.activation_out(logits_2_layer)

            # Applying optimized weights and bias to validation data

            logits_hidden_1_layer_val = porto_seguro_insur.linear_model(tf_val, weights_1_layer, biases_1_layer)

            a_1_layer_val = porto_seguro_insur.activation_hidden(logits_hidden_1_layer_val)

            logits_2_layer_val = porto_seguro_insur.linear_model(a_1_layer_val, weights_2_layer, biases_2_layer)

            val_prediction = porto_seguro_insur.activation_out(logits_2_layer_val)



            # Applying optimized weights and bias to test data

            logits_hidden_1_layer_test = porto_seguro_insur.linear_model(tf_test, weights_1_layer, biases_1_layer)

            a_1_layer_test = porto_seguro_insur.activation_hidden(logits_hidden_1_layer_test)

            logits_2_layer_test = porto_seguro_insur.linear_model(a_1_layer_test, weights_2_layer, biases_2_layer)

            test_prediction = porto_seguro_insur.activation_out(logits_2_layer_test)



        number_of_iterations = 900

        # Creating a tensorflow session to effeciently run same computation multiple times using definitions in defined

        # dataflow graph.

        with tf.Session(graph=graph) as session:

            # Ensure that variables are initialized as done in our graph defining the dataflow.

            tf.global_variables_initializer().run()

            for ite in range(number_of_iterations):

                # Compute loss and predictions

                loss, predictions = session.run([optimized_weights_and_bias, loss_function, train_prediction])[1:3]

                if ite % 100 == 0:

                    print('Loss at iteration %d: %f' % (ite, loss))

                    print('Training accuracy: %.1f%%' % porto_seguro_insur.accuracy(predictions, y_train))

            print('Test accuracy: %.1f%%' % porto_seguro_insur.accuracy(val_prediction.eval(), y_val))

            output = test_prediction.eval()



    is_make_prediction = 1

    if is_make_prediction:

        ''' Submission '''

        # Submission requires a csv file with id and target columns.

        submission = pd.DataFrame({'id': id_df_test, 'target': np.argmax(output, 1)})

        submission.to_csv(''.join(['submission_porto_seguro_insur_', porto_seguro_insur.timestamp, '.csv']), index=False)

        print(porto_seguro_insur.timestamp)



if __name__ == '__main__':

    main()