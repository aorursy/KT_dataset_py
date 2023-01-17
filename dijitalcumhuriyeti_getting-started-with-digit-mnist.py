import keras

from keras.datasets import mnist



(train_images,train_labels),(test_images,test_labels) = mnist.load_data()





model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128,activation='relu'),

    keras.layers.Dense(10)

])

model.compile(optimizer='adam',

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

model.summary()
model.fit(train_images,train_labels,epochs=10)

predict = model.predict(test_images)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("\nTest accuracy:",test_acc)
