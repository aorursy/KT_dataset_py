from sklearn.datasets import load_digits



df = load_digits()

x = df.data

y = df.target



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)



print(x_train.shape)



from keras.models import Sequential

model = Sequential()



from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=64))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])



model.fit(x_train, y_train, epochs=1000, batch_size=32)

print(model.evaluate(x_test, y_test, batch_size=128))

y_pred = model.predict(x_test, batch_size=128)