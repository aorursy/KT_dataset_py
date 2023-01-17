import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from IPython import display

from caffe2.proto import caffe2_pb2

from caffe2.python import cnn, core, utils, workspace, net_drawer
train_csv = pd.read_csv('/MNIST/data/train.csv')

test_csv = pd.read_csv('/MNIST/data/test.csv')
# Fetch data from CSV files

X_train = train_csv.ix[:,1:].values.astype('float32')

y_train = train_csv.ix[:,0].values.astype('int32')

X_test = test_csv.values.astype('float32')



# Reshape all images (1x784 -> 28x28)

X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_test = X_test.reshape(X_test.shape[0], 28, 28)



# Add feature layer

X_train = np.expand_dims(X_train, axis=1)

X_test = np.expand_dims(X_test, axis=1)
# Feature standardization

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)

standardize = lambda image: (image - mean_px) / std_px



# Preprocessing

X_train = np.array([standardize(image) for image in X_train])

X_test = np.array([standardize(image) for image in X_test])



# Split all dataset for training and validation set

X_validation = np.array(X_train[:1000])

y_validation = np.array(y_train[:1000])

X_train = np.array(X_train[1000:])

y_train = np.array(y_train[1000:])
# Prepare Caffe2 database for out MNIST data

def create_database(db_name, images, labels=None):

    # Create empty leveldb database

    db = core.C.create_db('leveldb', db_name, core.C.Mode.write)

    transaction = db.new_transaction()

    

    # Move all data to the database

    for i in range(images.shape[0]):

        tensor_protos = caffe2_pb2.TensorProtos()

        

        # Copy image with MNIST number

        img_tensor = tensor_protos.protos.add()

        img_tensor.dims.extend(images[i].shape)

        img_tensor.data_type = 1

        flatten_img = images[i].reshape(np.prod(images[i].shape))

        img_tensor.float_data.extend(flatten_img)



        # Copy label for each number

        label_tensor = tensor_protos.protos.add()

        label_tensor.data_type = 2

        if labels is not None:

            label_tensor.int32_data.append(labels[i])

        else:

            label_tensor.int32_data.append(-1)



        # Add data in transaction

        transaction.put('%0.6d' % i, tensor_protos.SerializeToString())



    # Close the transaction and close the database

    del transaction

    del db
# Create all databases

create_database('db_train', X_train, y_train)

create_database('db_validation', X_validation, y_validation)

create_database('db_test', X_test)
def create_model(model_name, db_name, batch_size=100, train=True, accuracy=True):

    # Create empty model with CCN model helper (and initialize if needed for training)

    if train:

        model = cnn.CNNModelHelper(order="NCHW", name=model_name)

    else:

        model = cnn.CNNModelHelper(order="NCHW", name=model_name, init_params=False)



    # Prepare data input operator that will fetch data from DB

    data, label = model.TensorProtosDBInput([], ['data', 'label'], batch_size=batch_size, db=db_name, db_type='leveldb')

    data = model.StopGradient(data, data)

    

    # First convolution: 28 x 28 -> 24 x 24

    conv1 = model.Conv(data, 'conv1', dim_in=1, dim_out=20, kernel=5)

    

    # First pooling: 24 x 24 -> 12 x 12

    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)

    

    # Second convolution: 12 x 12 -> 8 x 8

    conv2 = model.Conv(pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)

    

    # Second pooling: 8 x 8 -> 4 x 4

    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)

    

    # Fully connected layers at the end

    fc3 = model.FC(pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500) # 50 * 4 * 4 = dim_out from previous layer * image size

    fc3 = model.Relu(fc3, fc3)

    pred = model.FC(fc3, 'pred', 500, 10)

    softmax = model.Softmax(pred, 'softmax')

    

    # Check if we need to add training operators

    if train:

        # Prepare Cross Entropy operators with loss

        xent = model.LabelCrossEntropy([softmax, label], 'xent')

        loss = model.AveragedLoss(xent, "loss")



        # Add all gradient operators that will be needed to calculate our loss and train our model

        model.AddGradientOperators([loss])

        

        # Prepare variables for SGD

        ITER = model.Iter("iter")

        LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999)

        ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

        

        # Update all gradients for each params

        for param in model.params:

            # Note how we get the gradient of each parameter - CNNModelHelper keeps

            # track of that

            param_grad = model.param_to_grad[param]

            

            # The update is a simple weighted sum: param = param + param_grad * LR

            model.WeightedSum([param, ONE, param_grad, LR], param)

    

    # Add accuracy metrics if needed

    if accuracy:

        model.Accuracy([softmax, label], "accuracy")

    

    return model
# Create all needed models

training_model = create_model('mnist_train', 'db_train')

validation_model = create_model('mnist_validation', 'db_validation', train=False)

test_model = create_model('mnist_test', 'db_test', train=False, accuracy=False)
def calculate_validation_accuracy():

    # Initialize our model

    workspace.RunNetOnce(validation_model.param_init_net)

    workspace.CreateNet(validation_model.net)

    

    # Iterate over all validation dataset

    all_accuracy = []

    for i in range(X_validation.shape[0]/100):

        workspace.RunNet(validation_model.net.Proto().name)

        all_accuracy.append(workspace.FetchBlob('accuracy'))

    

    # Return mean accuracy for validation dataset

    return np.array(all_accuracy).mean()
# Initialize out training model

workspace.RunNetOnce(training_model.param_init_net)

workspace.CreateNet(training_model.net)



# Iterate over all epochs

NUMBER_OF_EPOCHS = 1000

for i in range(NUMBER_OF_EPOCHS):

    # Train our model

    start_time = time.time()

    workspace.RunNet(training_model.net.Proto().name)

    

    # Once per 20 epochs let's run validation and print results

    if (i+1) % 20 == 0:

        train_loss = workspace.FetchBlob('loss')

        train_accuracy = workspace.FetchBlob('accuracy')

        val_accuracy = calculate_validation_accuracy()

        epoch_time = time.time()-start_time

        print(('Epoch #%d/%d TIME_per_epoch: %.3fs '+

               'TRAIN_Loss: %.4f TRAIN_Acc: %.4f '+

               'VAL_Acc: %.4f') % (i+1, NUMBER_OF_EPOCHS, epoch_time, train_loss, train_accuracy, val_accuracy))
# Initialize out prediction model

workspace.RunNetOnce(test_model.param_init_net)

workspace.CreateNet(test_model.net)



# Iterate over all test dataset

predicted_labels = []

for i in range(X_test.shape[0]/100):

    # Run our model for predicting labels

    workspace.RunNet(test_model.net.Proto().name)

    batch_prediction = workspace.FetchBlob('softmax')

    if (i+1) % 20 == 0:

        print('Predicting #{}/{}...'.format(i+1, X_test.shape[0]/100))

    

    # Retrieve labels

    for prediction in batch_prediction:

        predicted_labels.append(np.argmax(prediction))  # Label = index of max argument
# Save all predicted labels into CSV file

submission = pd.DataFrame({

    "ImageId": list(range(1, len(predicted_labels)+1)),

    "Label": predicted_labels

})

submission.to_csv('/MNIST/data/output.csv', index=False, header=True)

print('Saved on disk!')