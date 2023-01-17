import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras import Sequential

from keras.backend import clear_session

from keras.layers import Dense, Dropout, BatchNormalization, Input

from keras.optimizers import Adam



from tensorflow_addons.layers import WeightNormalization

from tensorflow.keras import callbacks



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, KFold

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier



from skopt import gp_minimize

from skopt.space import Real, Categorical, Integer

from skopt.utils import use_named_args
#load data

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



train_features = train_features.set_index('sig_id')

train_targets = train_targets.set_index('sig_id')



train_targets.head()
print('Features shape:', train_features.shape)

print('Scored targets shape:', train_targets.shape)
#features are in categories

g_features = [x for x in train_features.columns if x.startswith('g-')]

c_features = [x for x in train_features.columns if x.startswith('c-')]

other_features = [x for x in train_features.columns if x not in g_features+c_features]
#encode binary features

train_features['cp_type'] = train_features['cp_type'].map({

    'trt_cp' : 0,

    'ctl_vehicle' : 1})

train_features['cp_dose'] = train_features['cp_dose'].map({

    'D1' : 0,

    'D2' : 1})
#top features for modelling

top_features = [  1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,

        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,

        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,

        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,

        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,

        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,

        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,

       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,

       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,

       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,

       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,

       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,

       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,

       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,

       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,

       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,

       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,

       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,

       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,

       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,

       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,

       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,

       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,

       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,

       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,

       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,

       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,

       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,

       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,

       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,

       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,

       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,

       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,

       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,

       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,

       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,

       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,

       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,

       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,

       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,

       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,

       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,

       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,

       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,

       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,

       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,

       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,

       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,

       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,

       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,

       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,

       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,

       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,

       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,

       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,

       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,

       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,

       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,

       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,

       870, 871, 872, 873, 874]
#prepare datasets for testing

n_targets = train_targets.shape[1]



X = train_features.iloc[:, top_features]

y = train_targets.iloc[:,:n_targets]



X = pd.get_dummies(X, columns = ['cp_time'])



X.columns
scale_cols = list(set(X.columns).intersection(set(g_features + c_features)))

scaler = StandardScaler()

X[scale_cols] = scaler.fit_transform(X[scale_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
#baseline model has two layers of 1000 neurons

model = Sequential()



model.add(Input(X_train.shape[1]))



model.add(BatchNormalization())

model.add(WeightNormalization(Dense(1000, activation = 'selu', kernel_initializer = 'lecun_normal')))

model.add(Dropout(0.2))



model.add(BatchNormalization())

model.add(WeightNormalization(Dense(500, activation = 'selu', kernel_initializer = 'lecun_normal')))

model.add(Dropout(0.4))



model.add(BatchNormalization())

model.add(WeightNormalization(Dense(250, activation = 'selu', kernel_initializer = 'lecun_normal')))

model.add(Dropout(0.4))



model.add(BatchNormalization())

model.add(WeightNormalization(Dense(y_train.shape[1], activation = 'sigmoid')))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy')



model.summary()
early_stopping = callbacks.EarlyStopping(patience = 10, restore_best_weights = True)

reduce_lr = callbacks.ReduceLROnPlateau(patience = 3, mode = 'min', monitor = 'val_loss')
model.fit(X_train, y_train,

         batch_size = 64,

         epochs = 20,

         validation_data = (X_test, y_test),

         callbacks = [early_stopping, reduce_lr])
#metric for this comp is log loss

def score(y_test, y_pred):

    metric = []

    y_test, y_pred = np.array(y_test), np.array(y_pred)

    for col in range(y_test.shape[1]):

        metric.append(log_loss(y_test[:, col], y_pred[:, col].astype('float'), labels = [0, 1]))

    return np.mean(metric)
y_pred = model.predict(X_test)

test_loss = score(y_test, y_pred)



train_loss = score(y_train, model.predict(X_train))



print('Train loss: {}'.format(train_loss))

print('Test loss: {}'.format(test_loss))
#function to create the model given settings

def build_model(hidden_layers, neurons, dropout_rate):

    model = Sequential()



    model.add(Input(X_train.shape[1]))

    

    for i in range(hidden_layers):

        model.add(BatchNormalization())

        model.add(WeightNormalization(Dense(neurons // 2**i, activation = 'selu', kernel_initializer = 'lecun_normal')))

        model.add(Dropout(dropout_rate))



    model.add(BatchNormalization())

    model.add(Dense(y_train.shape[1], activation = 'sigmoid'))



    model.compile(optimizer = Adam(), loss = 'binary_crossentropy')

    

    return model
#dimensions for each variable to optimise

dim_hidden_layers = Integer(low = 2, high = 4, name = 'hidden_layers')

dim_neurons = Integer(low = 1000, high = 2000, name = 'neurons')

dim_dropout_rate = Real(low = 0.2, high = 0.5, name = 'dropout_rate')

#dim_learning_rate = Real(low = 1e-4, high = 1e-2, prior = 'log-uniform', name = 'learning_rate')

#dim_batch_size = Integer(low = 4, high = 16, name = 'batch_size')

#dim_epochs = Integer(low = 10, high = 20, name = 'epochs')



dimensions = [

    dim_hidden_layers,

    dim_neurons,

    dim_dropout_rate]



default = [3, 1024, 0.3]
#the objective function that skopt will optimise

@use_named_args(dimensions = dimensions)

def obj_fun(hidden_layers, neurons, dropout_rate):

    model = build_model(

        hidden_layers = hidden_layers,

        neurons = neurons,

        dropout_rate = dropout_rate)

    

    history = model.fit(X_train, y_train,

                        batch_size = 64,

                        epochs = 20,

                        validation_data = (X_test, y_test),

                        callbacks = [early_stopping, reduce_lr])

    

    logloss = score(y_test, model.predict(X_test))

    

    clear_session()

    

    return logloss
clear_session() #good to clear session each time
#perform the optimisation

opt_result = gp_minimize(func = obj_fun,

                         dimensions = dimensions,

                         n_calls = 50,

                         n_jobs = -1,

                         x0 = default)
#put the results of the optimisation into a dataframe

results = pd.DataFrame(opt_result.x_iters,

                      columns = [

                          'hidden_layers',

                          'neurons',

                          'dropout_rate'])

results['obj_fun'] = opt_result.func_vals
#top 10 results

results.sort_values('obj_fun', ascending = True).head(10)
kf = KFold(n_splits = 7, shuffle = True, random_state = 69)
#prepare the submission input data

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test = test.set_index('sig_id')



test['cp_type'] = test['cp_type'].map({

    'trt_cp' : 0,

    'ctl_vehicle' : 1})

test['cp_dose'] = test['cp_dose'].map({

    'D1' : 0,

    'D2' : 1})



X_test = test.iloc[:, top_features]

X_test = pd.get_dummies(X_test, columns = ['cp_time'])

X_test[scale_cols] = scaler.transform(X_test[scale_cols])
# fit the model to each fold and store val loss and test predictions

val_losses = []

y_preds = []

for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):

    #split data

    X_train, X_val = X.to_numpy()[train_idx], X.to_numpy()[val_idx]

    y_train, y_val = y.to_numpy()[train_idx], y.to_numpy()[val_idx]

    

    clear_session()

    #fit model with opt params

    model = build_model(hidden_layers = opt_result.x[0],

                        neurons = opt_result.x[1],

                        dropout_rate = opt_result.x[2])



    history = model.fit(X_train, y_train,

             batch_size = 64,

             epochs = 25,

             validation_data = (X_val, y_val),

             callbacks = [early_stopping, reduce_lr])

    

    val_losses.append(history.history['val_loss'][-1])

    

    #test predictions

    y_pred = model.predict(X_test)

    y_preds.append(y_pred)
print('Mean val loss: {}'.format(np.mean(val_losses)))
predictions = np.mean(np.array(y_preds), axis = 0) #average the predictions from each fold
submission = pd.DataFrame(predictions, columns = train_targets.columns)

submission['sig_id'] = test.index

submission = submission[['sig_id']+list(train_targets.columns)]
#set ctl vehicle predictions to 0

submission.loc[list(test.cp_type == 1), train_targets.columns] = 0
print(submission.shape)

submission.head()
submission.to_csv('submission.csv', index = False)