# Initial solution for the Store Sale Forecasting challange: 
# Approach:
#    A simple 4-layered feed-forward neural network that predicts the sales of a given item in a store at a given date.
#    Use this notebook just for practice because this solution is not suitable for forecasting problems
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time


import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# First let us load the datasets into different Dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_submission_df = pd.read_csv('../input/sample_submission.csv')

# Dimensions
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
print('Sample submission shape:', sample_submission_df.shape)
# Set of features we have are: date, store, and item
display(train_df.sample(10))
display(test_df.sample(10))
display(sample_submission_df.sample(10))
def process_date(d):
    """
    Process the given date string and return a vector of various features.
    The vector contains [year,month,day,weekday] fields
    """
    dp = datetime.strptime(d, "%Y-%m-%d")
    date_fields = []
    date_fields.append(dp.year)
    date_fields.append(dp.month)
    date_fields.append(dp.day)
    date_fields.append(dp.weekday())
    return date_fields

# Sample date parsing
df = process_date('2018-09-8')
print(df)
# Process the training data
vec_train_df = train_df.copy(deep=True)

# Vectorize the date field
vec_train_df['year'],vec_train_df['month'],vec_train_df['day'],vec_train_df['weekday'] = zip(*train_df.date.apply(lambda d: process_date(d)).values)
vec_train_df = vec_train_df.drop(columns='date')
display(vec_train_df.sample(10))
# Process the test data
vec_test_df = test_df.copy(deep=True)

# Vectorize the date field
vec_test_df['year'],vec_test_df['month'],vec_test_df['day'],vec_test_df['weekday'] = zip(*test_df.date.apply(lambda d: process_date(d)).values)
vec_test_df = vec_test_df.drop(columns='date')
display(vec_test_df.sample(10))
def vectorized_train_and_test():
    """
        Returns the final train and test dataframes, ready for training and predicting.
    """
    return vec_train_df,vec_test_df
### Below are related DNN
class FFNet:
    """
    A Feed-forward network with 4 hidden layers
    """
    def __init__(self, n, num_classes, hidden_neurons=[1024,512,128,64], learning_rate=0.01, 
                 l2_lambda = 0.01, use_regularization=False,
                 data_type = tf.float32):
        # Hidden units
        (n1, n2, n3, n4) = hidden_neurons
        self.n_neurons_1 = n1
        self.n_neurons_2 = n2
        self.n_neurons_3 = n3
        self.n_neurons_4 = n4
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, n], name="input_x")
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name="input_y")

        print('Hidden neurons:', str(hidden_neurons))
        print('Number of hidden layers:', str(len(hidden_neurons)), ' + 1')
        print('Learning rate:', self.learning_rate)
        print('Use regularization:', use_regularization)
        print('l2_lambda:', l2_lambda)
        # Initializers
        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        # Weights and biases initialization
        self.weights = {
            'w1': tf.Variable(weight_initializer([n, self.n_neurons_1])),
            'w2': tf.Variable(weight_initializer([self.n_neurons_1, self.n_neurons_2])),
            'w3': tf.Variable(weight_initializer([self.n_neurons_2, self.n_neurons_3])),
            'w4': tf.Variable(weight_initializer([self.n_neurons_3, self.n_neurons_4])),
            'w_out': tf.Variable(weight_initializer([self.n_neurons_4, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(bias_initializer([self.n_neurons_1])),
            'b2': tf.Variable(bias_initializer([self.n_neurons_2])),
            'b3': tf.Variable(bias_initializer([self.n_neurons_3])),
            'b4': tf.Variable(bias_initializer([self.n_neurons_4])),
            'b_out': tf.Variable(bias_initializer([self.num_classes]))
        }
        
        # Hidden layers
        with tf.name_scope("hidden_layers"):
            # Then fully connected layers
            a1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1']), name="hidden_1")
            a2 = tf.nn.relu(tf.add(tf.matmul(a1, self.weights['w2']), self.biases['b2']), name="hidden_2")
            a3 = tf.nn.relu(tf.add(tf.matmul(a2, self.weights['w3']), self.biases['b3']), name="hidden_3")
            a4 = tf.nn.relu(tf.add(tf.matmul(a3, self.weights['w4']), self.biases['b4']), name="hidden_4")
            
        # Output layer with softmax
        with tf.name_scope("output_layer"):
            self.y_out = tf.add(tf.matmul(a4, self.weights['w_out']), self.biases['b_out'], name="predictions")
            self.a_out = tf.nn.softmax(self.y_out, name="output_activations")
            print('Output layer shape', self.y_out.shape)
        
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            all_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_out, labels=self.Y)
            self.loss = tf.reduce_mean(all_losses)
            
            # Regularization, if required
            if use_regularization:
                print('USING L2 Regularization')
                l2_regularizers = tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.weights['w2']) + tf.nn.l2_loss(self.weights['w3']) + tf.nn.l2_loss(self.weights['w4']) + tf.nn.l2_loss(self.weights['w_out'])
                self.loss = tf.reduce_mean(self.loss + l2_lambda * l2_regularizers)
            
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.a_out, 1) , tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        # Save the model
        self.model_saver = tf.train.Saver()      
#-----------Training and saving the model-----------------
def train(ff_net, data_x, data_y, num_epoches=100, batch_size=1000, epoch_display_step=10, model_path = None):
    """
        Train and save the model at the given model_path. Only the model with best score will be saved.
    """
    # Number of features
    n = data_x.shape[1]
    # Number of classes
    num_classes = ff_net.num_classes
    
    # Prepare the train, test and train batches
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.10, random_state=26) #24
    print('After data split:')
    print('Train X shape', X_train.shape)
    print('Train Y shape', Y_train.shape)
    print('Test X shape', X_test.shape)
    print('Test Y shape', Y_test.shape)
    
    # Process the test data
    X_test, Y_test = get_test_data(X_test, Y_test, num_classes)
    
    # Training data size after the split
    m = X_train.shape[0]
    num_batches = int(m / batch_size)
    print('Num of batches per epoch', num_batches)
        
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        start_time = time.time()
        if model_path:
            # Summary writer
            summary_file_writer = tf.summary.FileWriter(model_path + '/summary', session.graph)
            # Summary merger tensor
            merged_summary = tf.summary.merge_all()
        
        # Model with best accuracy found so far
        best_validation_accuracy = 0.0
        
        # One complete cycle of training data
        for epoch in range(num_epoches):
            avg_cost = 0.
            avg_training_accuracy = 0.
            
            # Run all batches
            for i_batch in range(num_batches):
                batch_X, batch_Y = get_batch(X_train, Y_train, i_batch, batch_size, num_classes)
                
                # Run the optimizer
                train_dict = {ff_net.X:batch_X, ff_net.Y:batch_Y}
                session.run([ff_net.loss, ff_net.optimizer], feed_dict=train_dict)
                
                # Get the current accuracy
                cst, acc = session.run([ff_net.loss, ff_net.accuracy], feed_dict=train_dict)
                avg_cost += cst
                avg_training_accuracy += acc
                            
            # Display test accuracy occasionally
            if epoch % epoch_display_step == 0:
                # Test the model with test data
                test_accuracy = ff_net.accuracy.eval({ff_net.X:X_test, ff_net.Y:Y_test})
                print('Epoch:', '%04d' % epoch, 'loss=', "{:.9f}".format(avg_cost / num_batches),
                     'Training accuracy=', (avg_training_accuracy / num_batches),
                     'Test/validation accuracy=', test_accuracy)
                # Save the model as MAX-model with highest validation accuracy found so far
                if model_path:
                    if test_accuracy > best_validation_accuracy:
                        best_validation_accuracy = test_accuracy
                        max_model_path = model_path + '_MAX'
                        # Save the model
                        ff_net.model_saver.save(sess=session, save_path=max_model_path)
                        print('Best validation accuracy found so far:', best_validation_accuracy)
                        print("Model saved in file: %s" % max_model_path)
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Save all variables of the TensorFlow graph to file.
        if model_path:
            # Save the model
            ff_net.model_saver.save(sess=session, save_path=model_path)
            print("Model saved in file: %s" % model_path)
        else:
            print('No model_path specified! Not saving the model')

        # Print the time-usage.
        print("Time taken: " + str(timedelta(seconds=int(round(time_taken)))))
        print("Training Finished!")
        
#---------Restore the model and run predictions---------------
def predict(ff_net, model_path, scalar_dump_dir, predict_x):
    """
        Run the prediction on batch of documents in predict_x using the 
        saved model at model_path and the scalar dump at scalar_dump_dir.
    """
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # Restore the model
        ff_net.model_saver.restore(sess=session, save_path=model_path)
        
        # Stop if atleast one NaN value is found [No need of data_y here, so passing dummy array]
        if contains_NaNs(predict_x, np.zeros((1,1))):
            print('Invalid data found in the test data!!')
            return

        # Scale input data
        predict_x = scale_data_x(predict_x, scalar_dump_dir=scalar_dump_dir, use_saved_scalar=True)
        print('Scaled the predict x using saved scalar')
        
        # Convert one-hot vector to actual class indeces
        preds = ff_net.y_out.eval({ff_net.X:predict_x})
        preds = np.argmax(preds, axis=1)
        return preds
#---------Helper methods used for training and testnig-----------
def get_batch(x_train, y_train, batch_num, batch_size, num_classes):
    """
        Pre-process the training batch data.
        Splits the training data into small batches of size batch_size. Also, converts the 
        class into a one-hot vector in the batch. Returns the x and y for a batch, separately.
    """
    batch_start = batch_num * batch_size
    
    # Prepare one-hot-vector of y_train
    y_one_hot = []
    for l in y_train[batch_start: batch_start + batch_size]:
        ohv = np.zeros(num_classes, dtype=float)
        ohv[l] = 1
        y_one_hot.append(ohv)
    # Conver to np array
    y_one_hot = np.array(y_one_hot)
    
    #print('Batch x and y shapes ', x_train[batch_start: batch_start + batch_size].shape, y_one_hot.shape)
    return x_train[batch_start: batch_start + batch_size], y_one_hot

def get_test_data(x_test, y_test, num_classes):
    """
        Pre-process the test data. In case of test data, 
        only y needs to be converted into one-hot vector.
    """
    y_one_hot = []
    for l in y_test:
        ohv = np.zeros(num_classes, dtype=float)
        ohv[l] = 1
        y_one_hot.append(ohv)
    
    # Conver to np array
    y_one_hot = np.array(y_one_hot)
    
    return x_test, y_one_hot

#----------Data Scaling----------------
from sklearn import preprocessing
import _pickle as cPickle
def scale_data_x(data_x, scalar_dump_dir, use_saved_scalar=False):
    """ 
    Scale the feature data.
    If use_saved_scalar is True, then the given data_x will be scaled using the scalar 
    saved at scalar_dump_dir. Otherwise, given data_x will be used to fit the scalar and the scalar will be saved at scalar_dump_dir.
    Finally, scaled data_x will be returned.
    """
    if not use_saved_scalar:
        scaler = preprocessing.StandardScaler().fit(data_x)
        print('Created a new scalar')
    else:
        scaler = cPickle.load(open(scalar_dump_dir + 'scalar.pickle', "rb"))
        print('Using the existing scalar at', scalar_dump_dir)
    
    scaled_data_x = scaler.transform(data_x)
    if scalar_dump_dir:
        cPickle.dump(scaler, open(scalar_dump_dir + 'scalar.pickle', "wb"))
    print('Scaled data x to shape', scaled_data_x.shape)
    return scaled_data_x

#--------Check for NaNs----------
def contains_NaNs(X_data, Y_data):
    """
        Checks if the given X_data and Y_data has any NaNs. 
        Returns True, if there is atleast one NaN in the data.
    """
    if np.isfinite(X_data).any() and np.isfinite(Y_data).any():
        return False
    # Then there is some invalid data
    return True

a = np.empty((2,2))
b = np.empty((2,2))
a[:] = np.nan
b[:] = 0
contains_NaNs(a, b)
#-----Create the required directories------
tmp_scalar_dump_dir = 'tmp/scale/'
tmp_model_p = 'tmp/model'
sales_scalar_dump_dir = 'sales/scale/'
sales_model_p = 'sales/model'

def create_dir():
    dir_lst = [tmp_scalar_dump_dir, tmp_model_p, sales_scalar_dump_dir, sales_model_p]
    for d in dir_lst:
        if not os.path.exists(d):
            os.makedirs(d)
            print('Created', d)
            
def delete_dir():
    import shutil
    dir_lst = [tmp_scalar_dump_dir, tmp_model_p, sales_scalar_dump_dir, sales_model_p]
    for d in dir_lst:
        if os.path.exists(d):
            shutil.rmtree(d)
            print('Deleted', d)
            
# Delete existing and create empty directories
#ONE_TIME delete_dir()
create_dir()
#-------Test the FFNet-------
def try_net():
    """
        Sample training and prediction flow with random data
    """
    import random
    x = np.random.rand(1000,10)
    scale_data_x(x, tmp_scalar_dump_dir, use_saved_scalar=False)
    y = np.array([random.randint(0, 4) for i in range(1000)])
    
    # Network
    n = x.shape[1]
    ff_net = FFNet(n, num_classes = 5, hidden_neurons=[1024, 512, 128, 64], learning_rate=0.01, l2_lambda = 0.01, use_regularization=True)
    train(ff_net, x, y, num_epoches=5, batch_size=200, epoch_display_step=2, model_path=tmp_model_p)
    
    print('Predictions:')
    pred_x = np.random.rand(1,10)
    preds = predict(ff_net, tmp_model_p, tmp_scalar_dump_dir, pred_x)
    print('x', pred_x)
    print('y', preds)

try_net()
#---Training and Submission-----
y_column = 'sales'
MAX_SALE_VALUE = 500
# Number of features
n = 6
# Create the network
ff_net = FFNet(n, num_classes = MAX_SALE_VALUE, 
                   hidden_neurons=[1024, 1024, 768, 512], learning_rate=0.0015, 
                   l2_lambda = 0.005, use_regularization=False)
def train_sales_data():
    """
        Train the model on the actual sales train data and save the model.
    """
    train_data_df, _ = vectorized_train_and_test()
    #Separate the label y from the training data
    df_y_data = train_data_df[y_column]
    # Drop the y_column from the training data
    df_x_data = train_data_df.drop(y_column, axis=1)
    
    # Convert into numpy matrix
    X_data = df_x_data.values
    Y_data = df_y_data.values
    
    # Stop if atleast one NaN value is found
    if contains_NaNs(X_data, Y_data):
        print('Invalid data found in the training data!!')
        return
    
    # Scale input data x
    X_data = scale_data_x(X_data, scalar_dump_dir=sales_scalar_dump_dir, use_saved_scalar=False)
    #print('>>>>>[TODO] Training on small data!!')
    #X_data = X_data[:5000, :]
    #Y_data = Y_data[:5000]
    
    print('Training data X shape', X_data.shape)
    print('Training data Y shape', Y_data.shape)
    #np.set_printoptions(suppress=True)
    #print(X_data[0])    
    
    print('Starting the training process...')
    # Network
    n = X_data.shape[1]
    print('#>>> Value of n to re-create FFNet during prediction is:', n)
    train(ff_net, X_data, Y_data, num_epoches=120, batch_size=10000, epoch_display_step=1, model_path=sales_model_p)
    return ff_net
    
#------Tests----------    
def predict_sales_data(ff_net):
    """
    Generates the sample predictions
    """
    train_data_df, _ = vectorized_train_and_test()
    train_data_df = train_data_df.sample(10)
    
    # Drop the y_column from the training data
    df_x_data = train_data_df.drop(y_column, axis=1)
    
    # Convert into numpy matrix
    predict_x = df_x_data.values
    predictions = predict(model_path=sales_model_p, ff_net=ff_net, scalar_dump_dir=sales_scalar_dump_dir, predict_x=predict_x)
    
    # Add the predictions to final submission df
    df_x_data['predictions'] = predictions
    return df_x_data

# Train and predict
train_sales_data()
df_predictions = predict_sales_data(ff_net)
df_predictions['sales'] = train_df.iloc[df_predictions.index.values, :].sales.values
display(df_predictions)
def submission_data(ff_net):
    """
    Generates the final predictions and creates the submission file
    """
    _, test_data_df = vectorized_train_and_test()
    
    # Drop the y_column from the training data
    df_x_data = test_data_df.drop('id', axis=1)
    
    # Convert into numpy matrix
    predict_x = df_x_data.values
    
    predictions = predict(model_path=sales_model_p, ff_net=ff_net, scalar_dump_dir=sales_scalar_dump_dir, predict_x=predict_x)
    
    # Add the predictions to final submission df
    df_x_data['predictions'] = predictions
    return df_x_data

submission_df = submission_data(ff_net)
# Rename the columns and sort by id
submission_df['sales'] = submission_df.loc[:,['predictions']]
final_submission_df = submission_df.loc[:,['sales']]
display(final_submission_df)
print('Saved the sumbision results')
final_submission_df.to_csv('final_submission_result.csv', index_label='id')
