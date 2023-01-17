import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import time

start_time = time.time()
df = pd.read_csv("../input/insurance/insurance.csv")

print(df.head(10))
print(df.describe())
sex_map = {'male':0, 'female':1}

smoker_map = {'no':0, 'yes':1}

region_map = {'southwest':0, 'northwest':1, 'northeast':2, 'southeast':3}

df.sex = df.sex.map(sex_map)

df.smoker = df.smoker.map(smoker_map)

df.region = df.region.map(region_map)

actual_charges = df.charges.values.tolist()

df.charges = df.charges.map(lambda x: np.log(x))

print(df.head(10))

print(df.describe())
random_seed = 17025

np.random.seed(random_seed)



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Average, Multiply, LeakyReLU

from tensorflow.keras.layers import Input, BatchNormalization, concatenate

from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import load_model



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from skimage.transform import resize



import matplotlib.pyplot as plt
def new_reg_model(in_shape):

    #Input layer

    INPUT = Input(in_shape)

    #Filter Layer

    k = 16

    f = []

    for i in range(k):

        f.append(Dense(256, activation='relu', kernel_initializer='normal')(INPUT))

    for i in range(k):

        f[i] = Dense(128, activation='relu')(f[i])

        f[i] = Dropout(0.25)(f[i])

    y = []

    for i in range(k-2):

        y.append(concatenate(f[i:i+2], axis=0))

    x = Average()(f)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1)(x)

    

    model = Model(inputs=INPUT, outputs=[x])

    

    optimizer = Adam(lr=0.01, decay=1e-5)

    

    #Compile model

    model.compile(optimizer,

                  loss='msle',#'mse',

                  metrics=['mae']

                 )

    return model



model = new_reg_model(df.shape[1:])
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
lrr = ReduceLROnPlateau(monitor = 'val_mae',

                         patience = 10,

                         verbose = 1,

                         factor = 0.5,

                         min_lr = 0.00001)



es = EarlyStopping(monitor='val_loss',

                   mode='min',

                   verbose=1,

                   patience=50,

                   restore_best_weights=True)



cols = df.columns[:-1]

#['age', 'sex', 'bmi', 'children', 'smoker']

print(cols)

x_train, x_val, y_train, y_val = train_test_split(df[cols], df.charges, test_size=0.25, shuffle=True, random_state=101)

model = new_reg_model(x_train.shape[1:])

history = model.fit(x_train, y_train,

                    epochs = 2000,

                    validation_data = (x_val, y_val),

                    verbose=1,

                    callbacks=[lrr, es]

                   )
print('Mean Absolute Error:')

plt.plot(history.history['mae'][10:])

plt.plot(history.history['val_mae'][10:])

plt.title('Model MAE')

plt.ylabel('MAE')

#plt.gca().set_ylim([0, 20000])

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.savefig('history_mae.png')

plt.show()



print('Loss:')

plt.plot(history.history['loss'][10:])

plt.plot(history.history['val_loss'][10:])

plt.title('Model Loss')

plt.ylabel('Loss')

#plt.gca().set_ylim([0, 5e7])

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.savefig('history_loss.png')

plt.show()
predictions = model.predict(df[cols])

print("Actual Value vs Predicted Value:\tDifference:")

diff = []

for a, b in zip(df.charges, predictions):

    a = np.exp(a)

    b = np.exp(b)

    diff.append(a-b[0])

    print("%.2f \t %.2f \t\t %.2f" %(a, b[0], a-b[0]))
print("Maximum Deviation: %.2f" %max(abs(x) for x in diff))

print("Minimum Deviation: %.2f" %min(abs(x) for x in diff))

print("Average Deviation: %.2f" %np.mean([abs(x) for x in diff]))
from eli5.sklearn import PermutationImportance

import sklearn, eli5

print(sklearn.metrics.SCORERS.keys())

perm = PermutationImportance(model, random_state=1, scoring='neg_mean_absolute_error').fit(df[cols], df.charges)

eli5.show_weights(perm, feature_names = cols.values.tolist())
cols = ['age', 'smoker', 'children', 'bmi']

print(cols)

x_train, x_val, y_train, y_val = train_test_split(df[cols], df.charges, test_size=0.25, shuffle=True, random_state=101)

model = new_reg_model(x_train.shape[1:])

history = model.fit(x_train, y_train,

                    epochs = 2000,

                    validation_data = (x_val, y_val),

                    verbose=1,

                    callbacks=[lrr, es]

                   )
print('Mean Absolute Error:')

plt.plot(history.history['mae'][10:])

plt.plot(history.history['val_mae'][10:])

plt.title('Model MAE Adjusted')

plt.ylabel('MAE')

#plt.gca().set_ylim([0, 20000])

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.savefig('history_mae_adjusted.png')

plt.show()



print('Loss:')

plt.plot(history.history['loss'][10:])

plt.plot(history.history['val_loss'][10:])

plt.title('Model Loss Adjusted')

plt.ylabel('Loss')

#plt.gca().set_ylim([0, 5e7])

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.savefig('history_loss_adjusted.png')

plt.show()
predictions = model.predict(df[cols])

print("Actual Value vs Predicted Value:\tDifference:")

diff = []

for a, b in zip(df.charges, predictions):

    a = np.exp(a)

    b = np.exp(b)

    diff.append(a-b[0])

    print("%.2f \t %.2f \t\t %.2f" %(a, b[0], a-b[0]))
print("Maximum Deviation: %.2f" %max(abs(x) for x in diff))

print("Minimum Deviation: %.2f" %min(abs(x) for x in diff))

print("Average Deviation: %.2f" %np.mean([abs(x) for x in diff]))
end_time = time.time()

total_time = end_time - start_time

h = total_time//3600

m = (total_time%3600)//60

s = total_time%60

print("Total Time: %i hours, %i minutes and %i seconds." %(h, m, s))