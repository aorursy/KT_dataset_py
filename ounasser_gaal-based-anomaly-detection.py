import tensorflow
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import keras
import math
import argparse
# Generator
def create_generator(latent_size):
    gen = Sequential()
    #build a model layer by layer, each layer has weights that correspond to the layer that follows it
    gen.add(
        Dense(latent_size,
              input_dim=latent_size,
              activation='relu',
              kernel_initializer=keras.initializers.Identity(gain=1.0)))
    #Dense: layer type, amm nodes from previous layer connect to the nodes of the current layer
    #latent_size: number of nodes in each input layer, ALSO: nombre des features
    #activation:relu: rectified linear activation
    gen.add(
        Dense(latent_size,
              activation='relu',
              kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size, ))
    fake_data = gen(latent)
    return Model(latent, fake_data)


# Discriminator
def create_discriminator(latent_size, data_size):
    dis = Sequential()
    dis.add(
        Dense(math.ceil(math.sqrt(data_size)),
              input_dim=latent_size,
              activation='relu',
              kernel_initializer=keras.initializers.VarianceScaling(
                  scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(
        Dense(1,
              activation='sigmoid',
              kernel_initializer=keras.initializers.VarianceScaling(
                  scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = Input(shape=(latent_size, ))
    fake = dis(data)
    return Model(data, fake)


# combine model
def create_combine_model(latent_size, optimizer_d, optimizer_g):
    # Create discriminator
    discriminator = create_discriminator()
    discriminator.compile(optimizer=optimizer_d, loss='binary_crossentropy')

    # Create combine model
    generator = create_generator(latent_size)
    latent = Input(shape=(latent_size, ))
    fake = generator(latent)
    discriminator.trainable = False
    fake = discriminator(fake)
    combine_model = Model(latent, fake)
    combine_model.compile(optimizer=optimizer_g, loss='binary_crossentropy')

    return combine_model
# Load data
def load_data(filename):

    # preprocess all databases other than creditcard
    if not filename.startswith('credit'):
        data = pd.read_table(
            'C:/Users/hp/Downloads/Data/' + filename,
            sep=',',
            header=None)
        data = data.sample(frac=1).reset_index(drop=True)
        id = data.pop(0)
        y = data.pop(1)
        data_x = data.as_matrix()
        data_id = id.values
        data_y = y.values
    else:
        # preprocess creditcard dataset
        data = pd.read_table(
            'C:/Users/hp/Downloads/Data/' + filename,
            sep=',')
        mask_nor = data['Class'] == 0
        mask_out = data['Class'] == 1
        column_name = 'Class'
        data.loc[mask_nor, column_name] = 'nor'
        data.loc[mask_out, column_name] = 'out'
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]
        data.insert(loc=0, column='indx', value=0)
        for index, row in data.iterrows():
            data.at[index, 'indx'] = index
        data = data.sample(frac=1).reset_index(drop=True)
        id = data.pop("indx")
        y = data.pop("Class")
        data_x = data.as_matrix()
        data_id = id.values
        data_y = y.values

    return data_x, data_y, data_id


# Plot loss history
def plot(train_history, name):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    aucy = train_history['auc']
    x = np.linspace(1, len(dy), len(dy))
    fig, ax = plt.subplots()
    ax.plot(x, dy, color='green', label='discriminator')
    ax.plot(x, gy, color='red', label='generator')
    ax.plot(x, aucy, color='yellow', linewidth='3', label='AUC')
    plt.show()
# tuning function
def tune(filename,
         k_ranges,
         lr_d_range,
         lr_g_range,
         optimizers_name,
         stop_epochs_range,
         plot=False):
    '''
    tunes the models hyperparams on the dataset specified in filename;
    Arguments:
        filename: dataset to tune on
        k_ranges: list of k range of values
        lr_d_range: list of lr_d range of values
        lr_g_range: list of lr_g range of values
        optimizers_name: list of strings specifying the optimizers to use: ('sgd', 'adam')
        stop_epochs_range: list of stop_epochs range of values
    Returns:
        final_results_df: pandas dataframe recap of the tuning process
    '''
    print('\tTUNING ON ---->', filename)

    results = {}
    final_results_df = pd.DataFrame()

    decay = 1e-6
    momentum = 0.9
    train = True
    data_x, data_y, data_id = load_data(filename)
    latent_size = data_x.shape[1]  #nombre des features
    data_size = data_x.shape[0]  #nombre d'enregistrements
    print("The dimension of the training data :{}*{}".format(
        data_x.shape[0], data_x.shape[1]))
    print()

    for k in k_ranges:
        results['k'] = k
        for stop_epochs in stop_epochs_range:
            results['stop_epochs'] = stop_epochs
            epochs = stop_epochs * 3
            for lr_d, lr_g in zip(lr_d_range, lr_g_range):
                results['lr_d'] = lr_d
                results['lr_g'] = lr_g
                for optimizer_name in optimizers_name:
                    results['optimizer'] = optimizer_name
                    if optimizer_name == 'sgd':
                        optimizer_d = SGD(lr_d, decay=decay, momentum=momentum)
                        optimizer_g = SGD(lr_g, decay=decay, momentum=momentum)
                        optimizer = SGD(lr_d, decay=decay, momentum=momentum)
                    else:
                        optimizer = Adam(lr_d, decay=decay)
                        optimizer_d = Adam(lr_d, decay=decay)
                        optimizer_g = Adam(lr_g, decay=decay)

                    if train:
                        train_history = defaultdict(list)
                        names = locals()
                        stop = 0

                        # Create discriminator
                        discriminator = create_discriminator(
                            latent_size, data_size)
                        discriminator.compile(optimizer=optimizer_d,
                                              loss='binary_crossentropy')

                        # Create k combine models
                        for i in range(k):
                            names['sub_generator' +
                                  str(i)] = create_generator(latent_size)
                            latent = Input(shape=(latent_size, ))
                            names['fake' + str(i)] = names['sub_generator' +
                                                           str(i)](latent)
                            discriminator.trainable = False
                            names['fake' + str(i)] = discriminator(
                                names['fake' + str(i)])
                            names['combine_model' + str(i)] = Model(
                                latent, names['fake' + str(i)])
                            names['combine_model' + str(i)].compile(
                                optimizer=optimizer_g,
                                loss='binary_crossentropy')

                        # Start iteration
                        for epoch in range(epochs):
                            print('Epoch {} of {}'.format(epoch + 1, epochs))
                            batch_size = min(500, data_size)
                            num_batches = int(data_size / batch_size)

                            for index in range(num_batches):
                                print(
                                    '\nTesting for epoch {} index {}:'.format(
                                        epoch + 1, index + 1))

                                # Generate noise
                                noise_size = batch_size
                                noise = np.random.uniform(
                                    0, 1, (int(noise_size), latent_size))

                                # Get training data
                                data_batch = data_x[index *
                                                    batch_size:(index + 1) *
                                                    batch_size]

                                # Generate potential outliers
                                block = ((1 + k) * k) // 2
                                for i in range(k):
                                    if i != (k - 1):
                                        noise_start = int(
                                            (((k + (k - i + 1)) * i) / 2) *
                                            (noise_size // block))
                                        noise_end = int(
                                            (((k + (k - i)) * (i + 1)) / 2) *
                                            (noise_size // block))
                                        names['noise' + str(
                                            i)] = noise[noise_start:noise_end]
                                        names['generated_data' +
                                              str(i)] = names[
                                                  'sub_generator' +
                                                  str(i)].predict(
                                                      names['noise' + str(i)],
                                                      verbose=0)
                                    else:
                                        noise_start = int(
                                            (((k + (k - i + 1)) * i) / 2) *
                                            (noise_size // block))
                                        names['noise' + str(
                                            i)] = noise[noise_start:noise_size]
                                        names['generated_data' +
                                              str(i)] = names[
                                                  'sub_generator' +
                                                  str(i)].predict(
                                                      names['noise' + str(i)],
                                                      verbose=0)

                                # Concatenate real data to generated data
                                for i in range(k):
                                    if i == 0:
                                        X = np.concatenate(
                                            (data_batch,
                                             names['generated_data' + str(i)]))
                                    else:
                                        X = np.concatenate(
                                            (X,
                                             names['generated_data' + str(i)]))
                                Y = np.array([1] * batch_size +
                                             [0] * int(noise_size))

                                # Train discriminator
                                discriminator_loss = discriminator.train_on_batch(
                                    X, Y)
                                train_history['discriminator_loss'].append(
                                    discriminator_loss)

                                # Get the target value of sub-generator
                                p_value = discriminator.predict(data_x)
                                p_value = pd.DataFrame(p_value)
                                for i in range(k):
                                    names['T' + str(i)] = p_value.quantile(i /
                                                                           k)
                                    names['trick' + str(i)] = np.array(
                                        [float(names['T' + str(i)])] *
                                        noise_size)

                                # Train generator
                                noise = np.random.uniform(
                                    0, 1, (int(noise_size), latent_size))
                                if stop == 0:
                                    for i in range(k):
                                        names['sub_generator' + str(i) +
                                              '_loss'] = names[
                                                  'combine_model' +
                                                  str(i)].train_on_batch(
                                                      noise,
                                                      names['trick' + str(i)])
                                        train_history['sub_generator{}_loss'.
                                                      format(i)].append(
                                                          names['sub_generator'
                                                                + str(i) +
                                                                '_loss'])
                                else:
                                    for i in range(k):
                                        names['sub_generator' + str(i) +
                                              '_loss'] = names[
                                                  'combine_model' +
                                                  str(i)].evaluate(
                                                      noise,
                                                      names['trick' + str(i)])
                                        train_history['sub_generator{}_loss'.
                                                      format(i)].append(
                                                          names['sub_generator'
                                                                + str(i) +
                                                                '_loss'])

                                generator_loss = 0
                                for i in range(k):
                                    generator_loss = generator_loss + names[
                                        'sub_generator' + str(i) + '_loss']
                                generator_loss = generator_loss / k
                                train_history['generator_loss'].append(
                                    generator_loss)

                                # Stop training generator
                                #if epoch +1  > args.stop_epochs:
                                if epoch + 1 > stop_epochs:
                                    stop = 1

                            # Detection result
                            data_y = pd.DataFrame(data_y)
                            result = np.concatenate((p_value, data_y), axis=1)
                            result = pd.DataFrame(result, columns=['p', 'y'])
                            result = result.sort_values('p', ascending=True)

                            # Calculate the AUC
                            inlier_parray = result.loc[lambda df: df.y ==
                                                       "nor", 'p'].values
                            outlier_parray = result.loc[lambda df: df.y ==
                                                        "out", 'p'].values
                            sum = 0.0
                            for o in outlier_parray:
                                for i in inlier_parray:
                                    if o < i:
                                        sum += 1.0
                                    elif o == i:
                                        sum += 0.5
                                    else:
                                        sum += 0
                            AUC = float('{:.4f}'.format(
                                sum /
                                (len(inlier_parray) * len(outlier_parray))))
                            print('AUC:{}'.format(AUC))
                            for i in range(num_batches):
                                train_history['auc'].append(AUC)

                    if plot:
                        plot(train_history, 'loss')
                    print('{} ---> AUC:{}'.format(filename, AUC))
                    results['AUC'] = AUC
                    print('results', results)

                    final_results_df = final_results_df.append(
                        pd.DataFrame(results.copy(),
                                     index=np.arange(len(results))))

    return final_results_df.drop_duplicates().reset_index().drop('index',
                                                                 axis=1)


def highlight_max(x):
    '''
    highlight the max AUC value in the final dataframe
    '''
    return [
        'background-color: lightblue'
        if x.name == 'AUC' and v == x.max() else '' for v in x
    ]
file_list = ['kddcup.data_10_percent.gz',
             'creditcard_copy_2_mod.csv',
             'Annthyroid.csv',
             'onecluster.csv',
             'SpamBase.csv',
             'WDBC.csv',
             'creditcard_1_1.csv']
np.random.seed(19960916)
filename = 'onecluster.txt'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df
np.random.seed(19960916)
filename = 'kddCup.txt'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df
np.random.seed(19960916)
filename = 'Annthyroid.txt'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df
np.random.seed(19960916)
filename = 'SpamBase.txt'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df
np.random.seed(19960916)
filename = 'WDBC.TXT'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df
np.random.seed(19960916)
filename = 'copy_2_mod.csv'
stop_epochs = [4, 6]
k_ranges = [3, 4]
lr_d_range = [0.01, 0.1]
lr_g_range = [0.0001, 0.001]
optimizer_names = ['sgd', 'adam']
results_df = tune(filename, k_ranges, lr_d_range, lr_g_range, optimizer_names, stop_epochs).style.apply(highlight_max)
print("Tuning results on", filename)
results_df