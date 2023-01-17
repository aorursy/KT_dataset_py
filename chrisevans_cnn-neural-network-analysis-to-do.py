model = Sequential()

# Conv Layer

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),

                 activation='relu',

                 input_shape=input_shape))

# Pooling Layer

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Conv Layer

model.add(Conv2D(64, (5, 5), activation='relu'))

# Pooling Layer

model.add(MaxPooling2D(pool_size=(2, 2)))

# Remove geometric awareness / prep for ordinary Dense layers

model.add(Flatten())

# Dense layer

model.add(Dense(1000, activation='relu'))

# Softmax output  (array of num_classes values, whose values sum to 1 => can be thought of as probabilities)

model.add(Dense(num_classes, activation='softmax'))
