import tensorflow
import pandas
import sklearn
import matplotlib.pyplot
path              = '/kaggle/input/numeric-iris-dataset/iris_numeric_dataset.csv'
dataframe         = pandas.read_csv(path)
dataframe         = sklearn.utils.shuffle(dataframe, random_state=42) #random state is a seed value
# dataframe.head()
train             = dataframe.sample(frac=0.8,random_state=42)
val               = dataframe.drop(train.index)
x_train, y_train  = train.iloc[:, :4],  train['variety']
x_val, y_val      = val.iloc[:, :4],    val['variety']
train_input_shape = x_train.iloc[1].shape
model = tensorflow.keras.models.Sequential([
  tensorflow.keras.layers.Dense( 10, input_shape=train_input_shape, activation='relu'),
  tensorflow.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001),
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))
matplotlib.pyplot.title('Learning Curves')
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Cross Entropy')
matplotlib.pyplot.plot(history.history['loss'], label='train')
matplotlib.pyplot.plot(history.history['val_loss'], label='val')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()