X_train = np.load('../input/train-test-split-of-open-images-dataset/X_train.npy')

y_train = np.load('../input/train-test-split-of-open-images-dataset/y_train.npy')

X_val = np.load('../input/train-test-split-of-open-images-dataset/X_val.npy')

y_val = np.load('../input/train-test-split-of-open-images-dataset/y_val.npy')
L2 = 0.001

dropout_rate = 0.5

learning_rate = 0.00001

optimizer = Adam(learning_rate)

hidden_dense = 2

hidden_dropout = 2
import wandb

from wandb.keras import WandbCallback

wandb.init(config={"L2": L2, "Learning Rate":learning_rate, "Optimizer":optimizer, "Hidden Dense-Layers":hidden_dense

                   , "Dropout Rate":dropout_rate, "Hidden Dropout Layers":hidden_dropout}, project='ResNet')
base_model = ResNet50(input_shape = (240,240,3), include_top = False)

base_model.trainable = True
def ResNet():

    inputs = Input(X_train.shape[1:])

    

    X = BatchNormalization(axis=3)(inputs, training=False)

    X = base_model(X)

    X = Flatten()(X)

    X = Dense(512, activation='relu', trainable=True, kernel_regularizer=regularizers.l2(L2))(X)

    X = Dropout(dropout_rate)(X)

    X = Dense(512, activation='relu', trainable=True, kernel_regularizer=regularizers.l2(L2))(X)

    X = Dropout(dropout_rate)(X)

    X = Dense(29, activation='sigmoid', trainable=True)(X)

    

    model = Model(inputs, X, name='ResNet')

    

    return model
ResNet = ResNet()

ResNet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

ResNet.summary()
ResNet.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[WandbCallback()])
X_test = np.load('../input/train-test-split-of-open-images-dataset/X_test.npy')

y_test = np.load('../input/train-test-split-of-open-images-dataset/y_test.npy')
y_pred = (ResNet.predict(X_test) > 0.5)

matrix = multilabel_confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

print(report)

ResNet.save('Final_Model')
#[1'Toilet',2'Swimming pool',3'Bed',4'Billiard table',5'Sink',6'Fountain',7'Oven',8'Ceiling fan',9'Television',

 #10'Microwave oven',11'Gas stove',12'Refrigerator',13'Washing machine',14'Bathtub',15'Stairs',16'Fireplace',17'Pillow',18'Mirror',

 #19'Shower',20'Couch',21'Countertop',22'Coffeemaker',23'Dishwasher',24'Sofa bed',25'Tree house',26'Towel',27'Porch',28'Wine rack',29'Jacuzzi']