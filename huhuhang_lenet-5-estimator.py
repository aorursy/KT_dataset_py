import numpy as np
import tensorflow as tf

# 从实验楼服务器下载 MNIST NumPy 数据
DATA_URL = 'http://labfile.oss.aliyuncs.com/courses/1211/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    # 将 28x28 图像 Padding 至 32x32
    x_train = np.pad(data['x_train'].reshape([-1, 28, 28, 1]),
                     ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    y_train = data['y_train']
    x_test = np.pad(data['x_test'].reshape([-1, 28, 28, 1]),
                    ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    y_test = data['y_test']

x_train.shape, y_train.shape, x_test.shape, y_test.shape
tf.__version__
model = tf.keras.models.Sequential([
    # 卷积层，6 个 5x5 卷积核，步长为 1，relu 激活，第一层需指定 input_shape
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1),
                           activation='relu', input_shape=(32, 32, 1)),
    # 平均池化，池化窗口默认为 2
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    # 卷积层，16 个 5x5 卷积核，步为 1，relu 激活
    tf.keras.layers.Conv2D(filters=16, kernel_size=(
        5, 5), strides=(1, 1), activation='relu'),
    # 平均池化，池化窗口默认为 2
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    # 需展平后才能与全连接层相连
    tf.keras.layers.Flatten(),
    # 全连接层，输出为 120，relu 激活
    tf.keras.layers.Dense(units=120, activation='relu'),
    # 全连接层，输出为 84，relu 激活
    tf.keras.layers.Dense(units=84, activation='relu'),
    # 全连接层，输出为 10，Softmax 激活
    tf.keras.layers.Dense(units=10, activation='softmax')
])
# 编译模型，Adam 优化器，多分类交叉熵损失函数，准确度评估
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 查看网络结构
model.summary()
def train_input_fn():
    # 训练数据输入函数
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.astype('float32'), y_train.astype('int64')))
    dataset = train_dataset.batch(32).repeat()
    return dataset


def test_input_fn():
    # 测试数据输入函数
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test.astype('float32'), y_test.astype('int64')))
    dataset = test_dataset.batch(32)
    return dataset
import tempfile

model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)
keras_estimator
keras_estimator.train(input_fn=train_input_fn, steps=3000)
eval_result = keras_estimator.evaluate(input_fn=test_input_fn, steps=500)
print('Eval result: {}'.format(eval_result))