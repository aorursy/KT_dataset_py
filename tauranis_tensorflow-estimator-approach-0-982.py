from __future__ import division

import tensorflow as tf
import numpy as np
import json
import pandas as pd
train_raw = json.load(open('../input/train.json','r'))
test_raw = json.load(open('../input/test.json','r'))
train_raw[0].keys()
X_train_np = np.array([ np.mean(sample['audio_embedding'],axis=0) for sample in train_raw]).astype(np.float32)
X_train_id = [sample['vid_id'] for sample in train_raw]

Y_train_np = np.array([ sample['is_turkey'] for sample in train_raw]).astype(np.float32)

X_test_np = np.array([ np.mean(sample['audio_embedding'],axis=0) for sample in test_raw]).astype(np.float32)
X_test_id = [sample['vid_id'] for sample in test_raw]
def znorm_fn(input_data):
    mean = np.mean(input_data,axis=0)
    std = np.std(input_data,axis=0)
    
    def _znorm_fn(col):        
        return (col - mean)/std
    
    return _znorm_fn
norm_fn = znorm_fn(X_train_np)
feature_columns = [
    tf.feature_column.numeric_column('average_embedding',shape=[128],normalizer_fn=norm_fn)]
batch_size = 256
epochs = 100
def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({'average_embedding':X_train_np,'vid_id':X_train_id},Y_train_np))    
    dataset = dataset.shuffle(100*batch_size).repeat(epochs).batch(batch_size)
    return dataset
def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({'average_embedding':X_test_np,'vid_id':X_test_id}))    
    dataset = dataset.batch(batch_size)
    return dataset
lr = tf.estimator.LinearClassifier(feature_columns=feature_columns,model_dir='./logistic_regression_trained_models')
lr = tf.contrib.estimator.forward_features(lr,'vid_id')
lr.train(input_fn=input_fn)
with open('submission_logistic_regression.csv','w') as f:
    f.write('vid_id,is_turkey\n')
    for prediction in lr.predict(input_fn=predict_input_fn):    
        f.write("{},{}\n".format(prediction['vid_id'],prediction['class_ids'][0]))
dnn = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                 model_dir='./dnn_trained_models',
                                 n_classes=2,
                                 dropout=0.2,                                
                                 hidden_units = [128,32,8,2])
dnn = tf.contrib.estimator.forward_features(dnn,'vid_id')
dnn.train(input_fn=input_fn)
with open('submission_dnn.csv','w') as f:
    f.write('vid_id,is_turkey\n')
    for prediction in lr.predict(input_fn=predict_input_fn):    
        f.write("{},{}\n".format(prediction['vid_id'],prediction['class_ids'][0]))
train_raw = json.load(open('../input/train.json','r'))
test_raw = json.load(open('../input/test.json','r'))
batch_size = 256
epochs = 20
lstm_X_train = []
lstm_X_train_id = []
lstm_Y_train = []

lstm_X_test = []
lstm_X_test_id = []

for train_sample in train_raw:        
    # Perform padding
    if len(train_sample['audio_embedding']) < 10:
        while len(train_sample['audio_embedding']) <10:
            train_sample['audio_embedding'].append(np.mean(train_sample['audio_embedding'],axis=0))
    
    
    lstm_X_train.append(train_sample['audio_embedding'])
    lstm_Y_train.append(train_sample['is_turkey'])
    lstm_X_train_id.append(train_sample['vid_id'])    

for test_sample in test_raw:       
        
    # Perform padding
    if len(test_sample['audio_embedding']) < 10:
        while len(test_sample['audio_embedding']) <10:
            test_sample['audio_embedding'].append(np.mean(test_sample['audio_embedding'],axis=0))
                
    lstm_X_test.append(test_sample['audio_embedding'])    
    lstm_X_test_id.append(test_sample['vid_id'])
lstm_X_train_np = np.array(lstm_X_train).astype(np.float32)
lstm_Y_train_np = np.array(lstm_Y_train).astype(np.float32)

lstm_X_test_np = np.array(lstm_X_test).astype(np.float32)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(1,train_size=0.9)
lstm_train_index, lstm_eval_index = next(sss.split(lstm_X_train_np, lstm_Y_train_np))

lstm_X_train_train_np = lstm_X_train_np[lstm_train_index]
lstm_X_train_eval_np = lstm_X_train_np[lstm_eval_index]

lstm_Y_train_train_np = lstm_Y_train_np[lstm_train_index]
lstm_Y_train_eval_np = lstm_Y_train_np[lstm_eval_index]

lstm_X_train_train_id = list(np.array(lstm_X_train_id)[lstm_train_index])
lstm_X_train_eval_id = list(np.array(lstm_X_train_id)[lstm_eval_index])
print(lstm_X_train_train_np.shape)
print(lstm_X_train_eval_np.shape)

print(lstm_Y_train_train_np.shape)
print(lstm_Y_train_eval_np.shape)

print(lstm_X_test_np.shape)

print(len(lstm_X_train_train_id))
print(len(lstm_X_train_eval_id))
def znorm_fn(input_data):
    mean = np.mean(input_data,axis=0)
    std = np.std(input_data,axis=0)
    
    def _znorm_fn(mini_batch):        
        norm_mini_batch = tf.map_fn(lambda c: (c-mean)/std,mini_batch)                
        return norm_mini_batch
    
    return _znorm_fn
norm_fn = znorm_fn(lstm_X_train_train_np)
feature_columns = [
    tf.feature_column.numeric_column('audio_embedding',shape=[128],normalizer_fn=norm_fn)]
def lstm_train_eval_input_fn(X,Y,vid_id,batch_size):
    
    def raw_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({'audio_embedding':X,'vid_id':vid_id},Y))    
        dataset = dataset.batch(batch_size)
        return dataset
    
    return raw_input_fn

lstm_train_input_fn = lstm_train_eval_input_fn(lstm_X_train_train_np,
                                               lstm_Y_train_train_np,
                                               lstm_X_train_train_id,
                                              batch_size)

lstm_eval_input_fn = lstm_train_eval_input_fn(lstm_X_train_eval_np,
                                              lstm_Y_train_eval_np,
                                              lstm_X_train_eval_id,
                                             batch_size)
def lstm_predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({'audio_embedding':lstm_X_test_np,'vid_id':lstm_X_test_id}))    
    dataset = dataset.batch(batch_size)
    return dataset
def metric_ops(target,predictions):    
    return {
        'Accuracy': tf.metrics.accuracy(
        labels=target,
        predictions=predictions,
        name='accuracy')
    }
params = tf.contrib.training.HParams(
    learning_rate = 5e-3,
    dropout=0.3,
    reg_val=1e-3,
    feature_columns = feature_columns # here you pass the feature columns to the model_fn (see below)
)

run_config = tf.estimator.RunConfig(
    model_dir='./lstm_model',
    save_summary_steps=100,    
    save_checkpoints_steps=100,
    keep_checkpoint_max=3,    
)

def model_fn(features, labels, mode, params):
            
    is_training = mode == tf.estimator.ModeKeys.TRAIN               

    # Apply transformations on input the features given the feature_columns
    input_features = tf.feature_column.input_layer(features, params.feature_columns)
    
    # Unfortunately, one needs to reshape the input tensor
    # because the transformation above cancels the second dimension (window size)
    input_features = tf.reshape(input_features,[-1,10,128])

    # Unstack the second dimension (window size) to return a list of tensors
    feature_sequence = tf.unstack(input_features,axis=1)    
    
    num_units = [128,64,8]
    
    # Create Bidirectional LSTM network
    lstm_forward_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
    lstm_backward_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
    
    stacked_lstm_forward_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_forward_cells)
    stacked_lstm_backward_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_backward_cells)
    
    lstm_output, _,_ = tf.nn.static_bidirectional_rnn(stacked_lstm_forward_cell,
                                                   stacked_lstm_backward_cell,
                                                   feature_sequence,
                                                   dtype=tf.float32)
    last_lstm_output = lstm_output[-1]
    

    net = tf.layers.dropout(last_lstm_output,params.dropout,training=is_training)
    
    net = tf.layers.dense(inputs=net,units=8,activation=tf.nn.relu,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_val),
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                          bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                          bias_regularizer=tf.contrib.layers.l2_regularizer(params.reg_val))    
        
    net = tf.layers.dense(inputs=net,units=2,activation=tf.nn.relu,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_val),
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                          bias_regularizer=tf.contrib.layers.l2_regularizer(params.reg_val))
    net = tf.nn.softmax(net)

    predictions = {
        'class_id':tf.argmax(net, axis=1, name="prediction"),
        'class_proba': net}    


    total_loss = None
    loss = None
    train_op = None
    eval_metric_ops = {} 
        
    if mode != tf.estimator.ModeKeys.PREDICT:
        
        # As target is just 0 or 1, it is necessary to transform to one-hot encoding
        target = tf.one_hot(tf.cast(labels,dtype=tf.uint8),depth=2)        
        
         # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES BY HAND
        reg_loss = tf.losses.get_regularization_loss()        
        loss = tf.losses.softmax_cross_entropy(target, net)                    
        total_loss = loss + reg_loss

        learning_rate = tf.constant(params.learning_rate, name='fixed_learning_rate')            
        optimizer = tf.train.AdamOptimizer(learning_rate)
#         optimizer = tf.train.RMSPropOptimizer(learning_rate)

        if is_training:
            # If you plan to use batch_norm layers, you DO must get this collection in order to perform updates on batch_norm variables
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(                    
                    loss=total_loss, global_step=tf.train.get_global_step())

        eval_metric_ops = metric_ops(labels, predictions['class_id'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
lstm = tf.estimator.Estimator(
    model_fn=model_fn,
    params=params,
    config=run_config
)

lstm = tf.contrib.estimator.forward_features(lstm,'vid_id')
# If you want to train without any evaluation, uncomment the line below
# lstm.train(input_fn=lstm_train_input_fn)


lstm_train_spec = tf.estimator.TrainSpec(input_fn=lstm_train_input_fn,
                                        max_steps=int(epochs*(len(lstm_X_train_train_id)/batch_size)))
lstm_eval_spec = tf.estimator.EvalSpec(input_fn=lstm_eval_input_fn,
                                       start_delay_secs=30,
                                       throttle_secs=15,
                                      steps=None)

tf.estimator.train_and_evaluate(estimator=lstm,
                                train_spec=lstm_train_spec,
                                eval_spec=lstm_eval_spec)

raw_predictions = {'vid_id':[],'is_turkey':[]}

for prediction in lstm.predict(input_fn=lstm_predict_input_fn):    
    raw_predictions['vid_id'].append(prediction['vid_id'])
    raw_predictions['is_turkey'].append(prediction['class_proba'][1])    

raw_predictions_df = pd.DataFrame(raw_predictions)
raw_predictions_df.head(10)
raw_predictions_df.to_csv('./submission_lstm.csv',index=None,columns=['vid_id','is_turkey'])