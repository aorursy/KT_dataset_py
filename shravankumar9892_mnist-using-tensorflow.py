!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Dense implements the operation: output = activation(dot(input, kernel) + bias) 
# where activation is the element-wise activation function passed as the activation argument, 
# kernel is a weights matrix created by the layer, 
# and bias is a bias vector created by the layer (only applicable if use_bias is True).

# Created the model object
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))  # Layer 1
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Layer 2

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
# Training Accuracy
model.evaluate(x_train, y_train)
# Confusion Matrix
with tf.Session() as sess:  
  sess.run(tf.confusion_matrix(labels = y_test,
                      predictions = y_hat))
# Creating Submission file

from google.colab import files
y_hat = model.predict_classes(x_test)

ImageId = np.array([x for x in range(1,28001)]).reshape(28000, 1)
data = {
    'ImageId': ImageId.flatten(),
    'Label': y_hat.flatten(),
}
predictions = pd.DataFrame(data)
predictions.to_csv(path_or_buf = 'predictions_deep.csv', index = False, sep = ',')
files.download('predictions_deep.csv')