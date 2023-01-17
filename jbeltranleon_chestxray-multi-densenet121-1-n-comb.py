from time import time as time

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

!pip install keras==2.1.3

from keras.optimizers import Adam, Adagrad, Adadelta

#Aux Scripts

import load_data

import train_model

import show_graphics
# reading the data

data = load_data.load_data_from_pkl("../input/chest-xray-data-multi14/data-chest-x-ray-multilabel-14.plk")

train_df, valid_df, test_df = load_data.do_train_test_split(data)
train_gen, valid_gen = load_data.create_flow_of_images(train_df, valid_df)
valid_X, valid_Y = load_data.create_flow_of_images_validation(valid_df)

test_X, test_Y = load_data.create_flow_of_images_test(test_df)
t_x, t_y = next(train_gen)

type(t_x), type(t_y)
all_labels = load_data.get_all_labels(data)
#Choices

weigths_list = ['imagenet', None]

learnRate_list = [1e-2, 1e-4, 1e-6]

optimizer_list = [Adam, Adagrad, Adadelta]

lossFunc_list = ['kullback_leibler_divergence', 'binary_crossentropy', 'categorical_crossentropy']

#Combinations

combinations = []

for weigth in weigths_list:

    for learn in learnRate_list:

        for opt in optimizer_list:

            for loss in lossFunc_list:

                combinations.append({'weight':weigth, 'lr':learn, 'optimizer':opt, 'loss':loss})  
results = pd.DataFrame(columns=['Model','weight','learn','opt','loss','test_loss_val', 'test_acc_val', 'test_bin_acc_val', 'elap_time(seg)'])



start_comb, end_comb = 18, 27

print("Combinations from {} to {} of 54".format(start_comb+1, end_comb))

for combination in combinations[start_comb:end_comb]:

    loss = combination['loss']

    optimizer_with_lr = combination['optimizer'](lr=combination['lr'])

    weights = combination['weight']

    multi_disease_model = train_model.create_model_dense_121(weights, t_x, all_labels, loss, optimizer_with_lr)

    history, checkpoint, early, callbacks_list = train_model.early_stopping()

    epochs = 50

    start = time()

    train_model.train(multi_disease_model, epochs, train_gen, valid_X, valid_Y, callbacks_list)

    end = time()

    loss_value, acc_value, bin_acc_value = train_model.evaluate(multi_disease_model, test_X, test_Y)

    results = results.append(

        {

            'Model': 'dense121', 'weight': combination['weight'], 'learn': combination['lr'],

            'opt' : combination['optimizer'].__name__, 'loss': combination['loss'], 'test_loss_val': loss_value, 

            'test_acc_val': acc_value, 'test_bin_acc_val': bin_acc_value, 'elap_time(seg)': round(end - start)

        },

        ignore_index=True

    )

    show_graphics.show(

        multi_disease_model, test_X, test_Y, history, all_labels, combination['lr'],

        combination['optimizer'].__name__, combination['loss'], loss_value, acc_value, bin_acc_value

    )


results