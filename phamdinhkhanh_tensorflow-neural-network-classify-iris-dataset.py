import os
import io
import urllib.request as request
import numpy as np
import pandas as pd
import tensorflow as tf

#Khai báo params
IRIS_TRAIN = 'iris_training.csv'
IRIS_TRAIN_PATH = 'https://raw.githubusercontent.com/phamdinhkhanh/Tensorflow/master/iris_training.csv'
IRIS_TEST = 'iris_testing.csv'
IRIS_TEST_PATH  = 'https://raw.githubusercontent.com/phamdinhkhanh/Tensorflow/master/iris_testing.csv'
COLUMNS = ['SepWid', 'SepLen', 'PenWid', 'PenLen', 'Species']
BATCH_SIZE = 100
N_STEPS = 1000
LEARNING_RATE = 0.2

def get_data(url, filename, is_get_train = True):
    if not os.path.exists(filename):
        raw = request.urlopen(url).read().decode('utf-8')
        with io.open(filename, 'w') as f:
            f.write(raw)
            
    data = pd.read_csv(filename, header = 0, names = COLUMNS, encoding = 'utf-8')
    features, labels = data, data.pop('Species')
    
    #Tạo Class Dataset trong tensorflow
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if is_get_train: 
        dataset = dataset.shuffle(1000).repeat().batch(batch_size = BATCH_SIZE)
        return dataset.make_one_shot_iterator().get_next()
    else:
        return dataset.batch(batch_size = BATCH_SIZE)

get_data(IRIS_TRAIN_PATH, IRIS_TRAIN, is_get_train = True)
get_data(IRIS_TEST_PATH, IRIS_TEST,  is_get_train = False)
with tf.Session() as sess:
    test = get_data(IRIS_TEST_PATH, IRIS_TEST, is_get_train = False)
    print(sess.run(test.make_one_shot_iterator().get_next()))
def my_model(features, labels, mode, params):
   
    # 0. Xây dựng mạng nơ ron
    # Khởi tạo input_layer. Hàm input_layer sẽ map dữ liệu đầu vào là features với các estimators thông qua params['features_columns']
    nn = tf.feature_column.input_layer(features, params['feature_columns'])
    # Xây dựng các hidden layer tiếp theo
    for n_units in params['hidden_units']:
        nn = tf.layers.dense(nn, n_units, activation = tf.nn.relu)
    # Hàm logits dự báo xác xuất các classes (chính là output layer)
    logits = tf.layers.dense(nn, params['n_classes'], activation = tf.nn.softmax)
    if labels is not None:
        # Hàm loss function
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels = labels, 
            logits = logits)
    
    #---------------------------------------------------------------------------------
    # 1. Huấn luyện mô hình
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Hàm tối ưu hóa kiểm soát thuật toán gradient descent:
        optimizer = tf.train.AdagradOptimizer(learning_rate = LEARNING_RATE)
        # Hàm kích hoạt quá trình training mô hình:
        train_op = optimizer.minimize(
                   loss, 
                   global_step = tf.train.get_global_step())
        # Estimator trả về 
        return tf.estimator.EstimatorSpec(
            mode = mode, 
            loss = loss, 
            train_op = train_op)
    
    #---------------------------------------------------------------------------------
    # 2. Đánh giá mô hình
    elif mode == tf.estimator.ModeKeys.EVAL:
        # Lớp được dự báo 
        prediction_classes = tf.argmax(logits, 1)
        # Mức độ chính xác 
        accuracy = tf.metrics.accuracy(
            labels = labels,
            predictions = prediction_classes
        )
        # Estimator trả về
        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {'accuracy': accuracy}
        )
    #----------------------------------------------------------------------------------
    # 3. Dự báo mô hình
    if mode == tf.estimator.ModeKeys.PREDICT:
        #Tính class dự báo.
        predicted_class = tf.argmax(logits, 1)
        # Estimator trả về
        return tf.estimator.EstimatorSpec(
            mode = mode, 
            predictions = {
                'class_id': predicted_class,
                'logits': logits,
                'probabilities': tf.nn.softmax(logits)
            })
my_features = []
for column in COLUMNS[:-1]:
    my_features.append(tf.feature_column.numeric_column(column))
my_features
classifier = tf.estimator.Estimator(
    model_fn = my_model,
    params = {
        'feature_columns': my_features, # List tên các feature sử dụng để map dữ liệu từ Dataset với các estimator
        'hidden_units':[10, 20, 10], # Số đơn vị mỗi layer
        'n_classes': 3 # Số lượng nhóm cần phân loại
    }
)
classifier.train(
    input_fn = lambda:get_data(IRIS_TRAIN_PATH, IRIS_TRAIN),
    steps = N_STEPS
)
classifier.evaluate(
    input_fn = lambda:get_data(IRIS_TEST_PATH, IRIS_TEST, is_get_train = False)
)
classifier.evaluate(
    input_fn = lambda:get_data(IRIS_TRAIN_PATH, IRIS_TRAIN, is_get_train = False)
)
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepLen': [3.1, 5.9, 6.9],
    'SepWid': [2.3, 3.0, 3.1],
    'PenLen': [1.7, 2.2, 5.4],
    'PenWid': [0.5, 1.5, 2.1],
}

def input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def pred(input_features):
    predictions = classifier.predict(
        input_fn=lambda:input_fn(input_features, labels = None, batch_size = BATCH_SIZE))
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    for pred in list(predictions):
        class_id = pred['class_id']
        exp = expected[class_id]
        prob = pred['probabilities'][class_id]*100
        print(template.format(class_id, prob, exp))

pred(predict_x)