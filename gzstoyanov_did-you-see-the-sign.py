%matplotlib inline
import numpy as np

import matplotlib.pyplot as plt

from math import sqrt, ceil



import pickle



import tensorflow as tf



from tensorflow.keras.utils import to_categorical



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dense, Dropout, ReLU, Softmax



from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import SparseCategoricalAccuracy

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
# CONSTANTS

EPOCHS = 15

BATCH_SIZE = 32

STEPS_PER_EPOCH = 86989 / BATCH_SIZE

#MODEL_SELECTION_VALIDATION_STEPS = 86989 / BATCH_SIZE
with open("../input/traffic-signs-preprocessed/data2.pickle", "rb") as f:

    data = pickle.load(f, encoding="latin1")



# Preparing y_train and y_validation for using in Keras

data["y_train"] = to_categorical(data["y_train"], num_classes=43)

data["y_validation"] = to_categorical(data["y_validation"], num_classes=43)



# Making channels come at the end

data["x_train"] = data["x_train"].transpose(0, 2, 3, 1)

data["x_validation"] = data["x_validation"].transpose(0, 2, 3, 1)

data["x_test"] = data["x_test"].transpose(0, 2, 3, 1)



# Showing loaded data from file

for i, j in data.items():

    if i == "labels":

        print(F"{i}:", len(j))

    else: 

        print(F"{i}:", j.shape)
data["y_validation"][0]
data["y_validation"] = np.argmax(data["y_validation"], axis=1)

data["y_train"] = np.argmax(data["y_train"], axis=1)
data["y_train"].shape
data["y_train"][0]
data["y_validation"].shape
data["y_validation"][0]
# Preparing function for ploting set of examples

# As input it will take 4D tensor and convert it to the grid

# Values will be scaled to the range [0, 255]

def convert_to_grid(x_input):

    N, H, W, C = x_input.shape

    grid_size = int(ceil(sqrt(N)))

    grid_height = H * grid_size + 1 * (grid_size - 1)

    grid_width = W * grid_size + 1 * (grid_size - 1)

    grid = np.zeros((grid_height, grid_width, C)) + 255

    next_idx = 0

    y0, y1 = 0, H

    for y in range(grid_size):

        x0, x1 = 0, W

        for x in range(grid_size):

            if next_idx < N:

                img = x_input[next_idx]

                low, high = np.min(img), np.max(img)

                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)

                next_idx += 1

            x0 += W + 1

            x1 += W + 1

        y0 += H + 1

        y1 += H + 1



    return grid
# Visualizing some examples of training data

examples = data["x_train"][100:200, :, :, :]

print(examples.shape)  # (81, 32, 32, 3)



# Plotting some examples

fig = plt.figure()

grid = convert_to_grid(examples)

plt.imshow(grid.astype("uint8"), cmap="gray")

plt.axis("off")

plt.gcf().set_size_inches(15, 15)

plt.title("Some examples of training data", fontsize=18)



# Showing the plot

plt.show()



# Saving the plot

fig.savefig("training_examples.png")

plt.close()
models_architectures_results = list()

final_models = list()

all_models = list()
def plot_hist(hist,

              first_param,

              second_param,

              name_of_plot,

              first_legend_label,

              second_legend_label,

              x_label,

              y_label,

              save_plot_img = False):

    

    plt.rcParams["figure.figsize"] = (15.0, 5.0)

    plt.rcParams["image.interpolation"] = "nearest"

    plt.rcParams["font.family"] = "Consolas"

    

    fig = plt.figure()

    plt.plot(hist.history[first_param], "-o", linewidth=3.0)

    plt.plot(hist.history[second_param], "-o", linewidth=3.0)

    

    plt.plot(range(hist.params["epochs"]),

             hist.history[first_param],

             c = "g",

             label = first_legend_label)

    plt.plot(range(hist.params["epochs"]),

             hist.history[second_param],

             c = "r",

             label = second_legend_label)

    plt.legend(fontsize="xx-large")

    

    plt.xticks(list(range(0, hist.params["epochs"])), range(1, hist.params["epochs"] + 1))

    max_ylim = max(max(hist.history[first_param]), max(hist.history[second_param])) + 0.02

    min_ylim = 0

    if "accuracy" in first_param:

        min_ylim = min(min(hist.history[first_param]), min(hist.history[second_param])) - 0.02

    plt.ylim(min_ylim, max_ylim)

    

    plt.title(name_of_plot, fontsize=22)

    plt.xlabel(x_label, fontsize = 18)

    plt.ylabel(y_label, fontsize = 18)

    plt.tick_params(labelsize=18)

    

    plt.show()

    

    if save_plot_img:

        fig.savefig(F"{hist.model.name}_{name_of_plot}.png")

        plt.close()
def train_and_plot_results(model, callbacks = []):

    model_hist = model.fit(data["x_train"],

                           data["y_train"],

                           batch_size=BATCH_SIZE,

                           epochs = EPOCHS,

                           steps_per_epoch = STEPS_PER_EPOCH,

                           validation_data = (data["x_validation"], data["y_validation"]),

                           callbacks = [TensorBoard(log_dir = F"TensorBoardLogs/{model.name}/", profile_batch = 100000000), 

                                        ModelCheckpoint(filepath = F"ModelsCheckpoints/{model.name}/")] + callbacks,

                           verbose = 0)

    plot_hist(model_hist,

          "loss",

          "val_loss",

          "Loss plot",

          "train loss",

          "validation loss",

          "Epoch",

          "Loss",

          True)

    plot_hist(model_hist,

          "sparse_categorical_accuracy",

          "val_sparse_categorical_accuracy",

          "Accuracy plot",

          "train accuracy",

          "validation accuracy",

          "Epoch",

          "Accuracy",

          True)

    train_loss, train_accuracy = model.evaluate(data["x_train"],

                                               data["y_train"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

    val_loss, val_accuracy = model.evaluate(data["x_validation"],

                                               data["y_validation"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

    print(F"Loss - Train: {train_loss:.3f}, Validation: {val_loss:.3f}")

    print(F"Accuracy - Train: {train_accuracy:.3f}, Validation: {val_accuracy:.3f}")

    print(F"Variance %: {((train_accuracy - val_accuracy) * 100):.3f}")

    return (model.name, train_loss, val_loss, train_accuracy, val_accuracy)
def evaluate_architectures_models(models):

    for model in models:

        train_loss, train_accuracy = model.evaluate(data["x_train"],

                                               data["y_train"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

        val_loss, val_accuracy = model.evaluate(data["x_validation"],

                                               data["y_validation"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

        print(model.name)

        print(F"Train        Loss: {train_loss:.3f}, Accuracy: {train_accuracy:.3f}")

        print(F"Validation   Loss: {val_loss:.3f}, Accuracy: {val_accuracy:.3f}")

        print(F"Variance %         {((train_accuracy - val_accuracy) * 100):.3f}")

        print()
n1_model = Sequential(

    name = "n1_model",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 32, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    Dense(30, activation = ReLU()),

    Dense(43, activation = Softmax())

])
n1_model.summary()
n1_model.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
models_architectures_results.append(train_and_plot_results(n1_model))
all_models.append(n1_model)
n2_model = Sequential(

    name = "n2_model",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 32, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    Dense(100, activation = ReLU()),

    Dropout(0.1),

    Dense(50, activation = ReLU()),

    Dropout(0.1),

    Dense(43, activation = Softmax())

])
n2_model.summary()
n2_model.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
models_architectures_results.append(train_and_plot_results(n2_model))
all_models.append(n2_model)
n3_model = Sequential(

    name = "n3_model",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 32, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(30, activation = ReLU()),

    Dense(43, activation = Softmax())

])
n3_model.summary()
n3_model.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
models_architectures_results.append(train_and_plot_results(n3_model))
all_models.append(n3_model)
n4_model = Sequential(

    name = "n4_model",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 32, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(100, activation = ReLU()),

    Dropout(0.1),

    Dense(50, activation = ReLU()),

    Dropout(0.1),

    Dense(43, activation = Softmax())

])
n4_model.summary()
n4_model.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
models_architectures_results.append(train_and_plot_results(n4_model))
all_models.append(n4_model)
n5_model = Sequential(

    name = "n5_model",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(200, activation = ReLU()),

    Dropout(0.2),

    Dense(150, activation = ReLU()),

    BatchNormalization(),

    Dropout(0.1),

    Dense(100, activation = ReLU()),

    Dropout(0.05),  

    Dense(43, activation = Softmax())

])
n5_model.summary()
n5_model.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
models_architectures_results.append(train_and_plot_results(n5_model))
all_models.append(n5_model)
evaluate_architectures_models(all_models)
final_model_1 = Sequential(

    name = "final_model_1",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU()),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2())

])
final_model_1.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_1)
all_models.append(final_model_1)

final_models.append(final_model_1)
final_model_2 = Sequential(

    name = "final_model_2",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(200, activation = ReLU()),

    Dropout(0.2),

    Dense(150, activation = ReLU()),

    Dropout(0.1),

    Dense(100, activation = ReLU()),

    Dropout(0.05),  

    Dense(43, activation = Softmax())

])
final_model_2.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_2)
all_models.append(final_model_2)

final_models.append(final_model_2)
final_model_3 = Sequential(

    name = "final_model_3",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2()),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2())

])
final_model_3.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_3)
all_models.append(final_model_3)

final_models.append(final_model_3)
final_model_4 = Sequential(

    name = "final_model_4",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    MaxPool2D(pool_size = 2),

    Flatten(),

    BatchNormalization(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2())

])
final_model_4.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_4)
all_models.append(final_model_4)

final_models.append(final_model_4)
final_model_5 = Sequential(

    name = "final_model_5",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-3)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Flatten(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2()),

    BatchNormalization(),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2())

])
final_model_5.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_5)
all_models.append(final_model_5)

final_models.append(final_model_5)
final_model_6 = Sequential(

    name = "final_model_6",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 7e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Flatten(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2()),

    BatchNormalization(),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2()),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2())

])
final_model_6.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + EPOCHS))
train_and_plot_results(final_model_6, [annealer])
all_models.append(final_model_6)

final_models.append(final_model_6)
final_model_7 = Sequential(

    name = "final_model_7",

    layers = [

    Input(shape = (32, 32, 3)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = ReLU(), kernel_regularizer = l2(l = 1e-4)),

    BatchNormalization(),

    MaxPool2D(pool_size = 2),

    Flatten(),

    Dense(200, activation = ReLU(), kernel_regularizer = l2(0.5)),

    Dropout(0.2),

    Dense(150, activation = ReLU(), kernel_regularizer = l2(0.5)),

    BatchNormalization(),

    Dropout(0.1),

    Dense(100, activation = ReLU(), kernel_regularizer = l2(0.5)),

    Dropout(0.05),  

    Dense(43, activation = Softmax(), kernel_regularizer = l2(0.5))

])
final_model_7.compile(

    optimizer = Adam(learning_rate = 0.001),

    loss = SparseCategoricalCrossentropy(),

    metrics = [SparseCategoricalAccuracy()])
train_and_plot_results(final_model_7, [annealer])
all_models.append(final_model_7)

final_models.append(final_model_7)
evaluate_architectures_models(final_models)
test_loss, test_accuracy = final_model_1.evaluate(data["x_test"],

                                               data["y_test"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

print(F"Test - Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.3f}")
def evaluate_all_models(models):

    for model in models:

        train_loss, train_accuracy = model.evaluate(data["x_train"],

                                               data["y_train"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

        val_loss, val_accuracy = model.evaluate(data["x_validation"],

                                               data["y_validation"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

        test_loss, test_accuracy = model.evaluate(data["x_test"],

                                               data["y_test"],

                                               batch_size = BATCH_SIZE,

                                               verbose = 0)

        print(model.name)

        print(F"Train        Loss: {train_loss:.3f}, Accuracy: {train_accuracy:.3f}")

        print(F"Validation   Loss: {val_loss:.3f}, Accuracy: {val_accuracy:.3f}")

        print(F"Test         Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.3f}")

        print()
evaluate_all_models(all_models)