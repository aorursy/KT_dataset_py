import numpy as np

import pandas as pd

from os.path import join as opj

from tqdm.notebook import tqdm
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_context('notebook')
from sklearn.model_selection import ParameterGrid, KFold



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

tf.__version__
path = '/kaggle/input/trends-feature-exploration-engineering/datasets'
# Load target values and corresponding scaler

import joblib

scaler_targets = joblib.load(opj(path, 'targets_scaler.pkl'))



targets = pd.read_hdf(opj(path, 'targets.h5'))

targets.head()
def load_dataset_and_scale(dataset_id='merge', scale_values=[1, 1, 1, 1]):



    # Load dataset

    X_tr = pd.read_hdf(opj(path, '%s_train.h5' % dataset_id))

    X_te = pd.read_hdf(opj(path, '%s_test.h5' % dataset_id))



    # Specify target

    y = pd.read_hdf(opj(path, 'targets.h5'))

    y_tr = y.loc[X_tr.index, 'domain2_var1':]



    # Remove missing values

    missval = y_tr.isnull().sum(axis=1)!=0

    idx = ~missval

    X_tr = X_tr[idx]

    X_te = X_te

    y_tr = y_tr[idx]

    print('Removing %s missing values from target dataset.' % missval.sum())

    

    # Centralize corr_coef features

    median_offset = X_tr.iloc[:, X_tr.columns.str.contains('corr_coef')].median().mean()

    X_tr.iloc[:, X_tr.columns.str.contains('corr_coef')] -= median_offset

    X_te.iloc[:, X_te.columns.str.contains('corr_coef')] -= median_offset



    # Establish masks for different kinds of data

    mask_ids = []

    for m in ['IC_', '_vs_', 'corr_coef', '^c[0-9]+_c[0-9]+']:

        mask_ids.append(X_tr.columns.str.contains(m))

    mask_ids = np.array(mask_ids)



    # Data scaling

    for i, m in enumerate(mask_ids):



        if m.sum()==0:

            continue

        

        # Apply Scale

        scale_value = scale_values[i]

        unify_mask_scale = np.percentile(X_tr.iloc[:, m].abs(), 90)



        X_te.iloc[:, m] /= unify_mask_scale

        X_tr.iloc[:, m] /= unify_mask_scale

    

        X_te.iloc[:, m] *= scale_value

        X_tr.iloc[:, m] *= scale_value



    # Drop irrelevant measurements

    X_tr.dropna(axis=1, inplace=True)

    X_te.dropna(axis=1, inplace=True)

    

    # Drop duplicate rows

    X_tr = X_tr.T.drop_duplicates().T

    X_te = X_te.T.drop_duplicates().T

    

    print('Size of dataset (train/test): ', X_tr.shape, X_te.shape)

    

    X_tr = X_tr.values

    X_te = X_te.values

    y_tr = y_tr.values

    

    return X_tr, X_te, y_tr
# Load the data

X_tr, X_te, y_tr = load_dataset_and_scale(dataset_id='merge', scale_values=[1, 1, 1, 1])
# Revert target transformation to original

y_tr_orig = y_tr*scaler_targets.scale_[3:]+scaler_targets.mean_[3:]

y_tr_orig[:, :3] = np.power(y_tr_orig[:, :3], 1./1.5)
# Plot domain2 relationship with and without rotation

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))



ax[0].set_title('Domain2 relationship (orig)')

ax[0].scatter(y_tr_orig[:, 0], y_tr_orig[:, 1], s=1, alpha=0.5, c='b')

ax[0].axis('equal')



ax[1].set_title('Domain2 relationship (rotated)')

ax[1].scatter(y_tr_orig[:, 2], y_tr_orig[:, 3], s=1, alpha=0.5, c='r')

ax[1].axis('equal')



plt.tight_layout()

plt.show();
class CustomLoss_focus(keras.losses.Loss):

    def __init__(self, scale=None, mean=None, focus=0, name="loss"):

        super().__init__(name=name + '_%d' % (focus + 1))

        self.scale = np.array(scale, dtype='float32')

        self.mean = np.array(mean, dtype='float32')

        self.focus = np.array(focus, dtype='int32')



    @tf.function

    def call(self, y_true, y_pred):



        # Rescale values

        y_true = tf.math.add(tf.math.multiply(self.scale, y_true), self.mean)

        y_pred = tf.math.add(tf.math.multiply(self.scale, y_pred), self.mean)



        # Revert power transformation

        y_true = tf.transpose(tf.stack([

            tf.math.pow(y_true[:, 0], 1./1.5),

            tf.math.pow(y_true[:, 1], 1./1.5)]))

        y_pred = tf.transpose(tf.stack([

            tf.math.pow(y_pred[:, 0], 1./1.5),

            tf.math.pow(y_pred[:, 1], 1./1.5)]))



        # Appli competition loss function

        scores = tf.math.divide(tf.math.reduce_sum(tf.math.abs(y_true - y_pred), axis=0),

                                tf.math.reduce_sum(tf.math.abs(y_true), axis=0))

        

        # Focus on var1 or var2

        score = scores[self.focus]



        return score
def build_model(input_size=100,

                activation='relu',

                dropout=0.5,

                kernel_initializer='glorot_uniform',

                lr=1e-2,

                use_batch=True,

                use_dropout=True,

                hidden=(32, 16),

                hinge=0.5,

               ):



    # Specify layer structure

    inputs = keras.Input(shape=input_size)

    

    for i, h in enumerate(hidden):

        if i==0:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(inputs)

        else:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(x)

        if use_batch:

            x = layers.BatchNormalization()(x)

        

        x = layers.Activation(activation)(x)



        # Add Dropout layer if requested

        if use_dropout:

            x = layers.Dropout(dropout)(x)



    out = layers.Dense(2)(x)

    out1 = layers.Activation(None, name="loss1")(out)

    out2 = layers.Activation(None, name="loss2")(out)



    # Create Model

    model = keras.Model(inputs, [out1, out2])



    # Define loss function, optimizer and metrics to track during training

    optimizer = keras.optimizers.Adagrad(

            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(

                lr, decay_steps=10000, decay_rate=0.96, staircase=True))

    

    l1 = CustomLoss_focus(scale=scaler_targets.scale_[3:5],

                          mean=scaler_targets.mean_[3:5],

                          focus=0)

    l2 = CustomLoss_focus(scale=scaler_targets.scale_[3:5],

                          mean=scaler_targets.mean_[3:5],

                          focus=1)

    

    losses = {'loss1': l1, 'loss2': l2}

    lossWeights = {"loss1": hinge, "loss2": 1-hinge}



    model.compile(

        optimizer = optimizer,

        loss = losses,

        loss_weights = lossWeights

    )

    return model
def create_grid(input_size=0):

    

    # Define parameter grid

    param_grid = [

    { 

        'nn__input_size': [input_size],

        'nn__lr': [0.05],

        'nn__activation': ['sigmoid'],

        'nn__hidden': [(128, 32),],

        'nn__dropout': [0.25],

        'nn__kernel_initializer': ['glorot_uniform'], #['glorot_uniform', 'normal', 'uniform'],

        'nn__use_batch': [True],

        'nn__use_dropout': [True],

        'nn__hinge': [0., 0.5, 1.],

        'batch_size': [32],

        'epochs': [50],

    }]

    

    # Create Parameter Grid and return it

    grids = ParameterGrid(param_grid)

    return grids



len(create_grid())
def run_grid(grid, X_tr, X_te, y_tr, cv=5, n=2):

    

    x_train = X_tr.copy()

    x_test = X_te.copy()

    print('Parameters:', grid)

        

    # Create callback object

    callbacks = [keras.callbacks.EarlyStopping(

        monitor="loss", min_delta=1e-5, patience=5, verbose=1)]



    # Build NN model

    nn_grid ={}

    for k in grid.keys():

        if 'nn__' in k:

            nn_grid[k[4:]] = grid[k]



    # Run NN fit on multiple folds

    history_runs = {'loss': [], 'val_loss': []}

    for i in range(n):

        history_runs['loss%d' % (i+1)] = []

        history_runs['val_loss%d' % (i+1)] = []



    kfold = KFold(n_splits=cv, shuffle=True)

    

    preds_cv_tr = []

    preds_cv_te = []

    counter = 0

    for train_idx, val_idx in kfold.split(x_train):

        

        # Prepare data for split

        x_train_k = x_train[train_idx]

        x_val_k = x_train[val_idx]



        y_train_k = y_tr[train_idx]

        y_val_k = y_tr[val_idx]



        # Prepare model

        model = build_model(**nn_grid)



        # Fit model

        history = model.fit(

            x_train_k,

            y_train_k,

            validation_data=(x_val_k, y_val_k),

            epochs=grid['epochs'],

            batch_size=grid['batch_size'],

            #callbacks=callbacks,

            shuffle=True, verbose=0

        )

        history_runs['loss'].append(history.history['loss'])

        history_runs['val_loss'].append(history.history['val_loss'])

        

        for i in range(n):

            history_runs['loss%d' % (i+1)].append(history.history['loss%d_loss' % (i+1)])

            history_runs['val_loss%d' % (i+1)].append(history.history['val_loss%d_loss' % (i+1)])

        

        # Compute predictions

        preds_cv_tr.append(model.predict(x_train)[0])

        preds_cv_te.append(model.predict(x_test)[0])

        

        # Report counter

        print('CV: %02d/%02d' % (counter+1, cv))

        counter += 1



    # Compute mean and standard deviation

    for l in [''] + [str(i+1) for i in range(n)]:

        history_runs['loss'+l+'_mean'] = pd.DataFrame(history_runs['loss'+l+'']).T.mean(axis=1).values

        history_runs['val_loss'+l+'_mean'] = pd.DataFrame(history_runs['val_loss'+l+'']).T.mean(axis=1).values

        history_runs['loss'+l+'_std'] = pd.DataFrame(history_runs['loss'+l+'']).T.std(axis=1).values

        history_runs['val_loss'+l+'_std'] = pd.DataFrame(history_runs['val_loss'+l+'']).T.std(axis=1).values

    

    # Collect mean and median predictions

    df_pred_mean_tr = pd.DataFrame(np.mean([e for e in preds_cv_tr if not np.isnan(e[0, 0])], axis=0), columns=['domain21', 'domain22'])

    df_pred_mean_te = pd.DataFrame(np.mean([e for e in preds_cv_te if not np.isnan(e[0, 0])], axis=0), columns=['domain21', 'domain22'])



    df_pred_median_tr = pd.DataFrame(np.median([e for e in preds_cv_tr if not np.isnan(e[0, 0])], axis=0), columns=['domain21', 'domain22'])

    df_pred_median_te = pd.DataFrame(np.median([e for e in preds_cv_te if not np.isnan(e[0, 0])], axis=0), columns=['domain21', 'domain22'])



    # Assign closest value from training set

    for tid in range(2):

        unique_values = np.unique(y_tr[:, tid])

        for d in [df_pred_mean_tr, df_pred_mean_te, df_pred_median_tr, df_pred_median_te]:

            for i in range(len(d)):

                d.iloc[i, tid] = unique_values[np.argmin(np.abs(d.iloc[i, tid]-unique_values))]



    return x_train, x_test, history_runs, model, df_pred_mean_tr, df_pred_mean_te, df_pred_median_tr, df_pred_median_te
def get_scores(y_tr, pred_tr, scaler=None, title=''):



    scores = []

    preds = []

    trues = []

    for i, tidx in enumerate([3, 4]):



        # Invert scaler

        t_true = scaler.inverse_transform(np.transpose([y_tr[:, i]] * 7))[:, tidx]

        t_pred = scaler.inverse_transform(np.transpose([pred_tr.iloc[:, i]] * 7))[:, tidx]



        # Invert power transformation

        t_true = np.power(t_true, 1./1.5)

        t_pred = np.power(t_pred, 1./1.5)



        # Compute the score

        score = np.mean(np.sum(np.abs(t_true - t_pred), axis=0) / np.sum(t_true, axis=0))

        scores.append(score)

        preds.append(t_pred)

        trues.append(t_true)



    scores = np.array(scores)

    preds = np.array(preds)

    trues = np.array(trues)



    pred_score = list(np.round(scores, 5))

    plt.figure(figsize=(6, 6))

    plt.title('Score on whole train set (var1, var2): ' + title + ' - ' + str(pred_score))

    print('Score on whole train set (var1, var2) %s: ' % title, pred_score)

    plt.scatter(trues[0], trues[1], s=0.5, alpha=0.5);

    plt.scatter(preds[0], preds[1], s=0.5, alpha=0.5);

    plt.axis('equal')



    plt.show()
def plot_history_and_predictions(history, pred_tr, y_tr, cutoff=0, n=2):



    # Plot history results

    for l in [''] + [str(i+1) for i in range(n)]:

        fig = plt.figure(constrained_layout=True, figsize=(14, 3.5))

        plt.plot(history['loss'+l+'_mean'][cutoff:], label='train loss'+l)

        plt.plot(history['val_loss'+l+'_mean'][cutoff:], label='val loss'+l, linestyle='dotted')



        for m in ['loss'+l, 'val_loss'+l]:

            plt.fill_between(np.arange(len(history['%s_mean' % m][cutoff:])),

                             history['%s_mean' % m][cutoff:]-history['%s_std' % m][cutoff:],

                             history['%s_mean' % m][cutoff:]+history['%s_std' % m][cutoff:],

                             alpha=0.3)

        plt.title('Validation loss{}: {:.4f} (mean last 5)'.format(str(l),

            np.mean(history['val_loss'+l+'_mean'][-5:])

        ))

        plt.xlabel('epoch')

        plt.ylabel('loss'+l+' value')

        plt.legend()

        plt.show()



    # Categories

    categories = ['domain2_var1', 'domain2_var2']



    # Create subplot grid

    fig = plt.figure(constrained_layout=True, figsize=(9, 4))

    gs = fig.add_gridspec(1, 2)

    ax = [fig.add_subplot(gs[i]) for i in range(2)]



    # Plot discrepancy on training set

    for i, axt in enumerate(ax):

        sns.regplot(x=pred_tr[0][i, :], y=y_tr[:, i], marker='.', ax=axt,

                    scatter_kws={'alpha': 0.33, 's': 1})

        sns.regplot(x=pred_tr[1][i, :], y=y_tr[:, i], marker='.', ax=axt,

                    scatter_kws={'alpha': 0.33, 's': 1})

        axt.set_title('Prediction X_tr: %s' % categories[i])

        axt.set_xlim(-3, 3)

        axt.set_ylim(-3, 3)

        axt.legend(['Mean', 'Median'])

        axt.set_aspect('equal')



    plt.show()
def get_scores(y_tr, pred_tr, scaler=None):



    scores = []

    preds = []

    trues = []

    for j, tidx in enumerate([3, 4]):



        # Invert scaler

        t_true = scaler.inverse_transform(np.transpose([y_tr[:, j]] * 7))[:, tidx]

        t_pred = scaler.inverse_transform(np.transpose([pred_tr.iloc[:, j]] * 7))[:, tidx]



        # Invert power transformation

        t_true = np.power(t_true, 1./1.5)

        t_pred = np.power(t_pred, 1./1.5)



        # Compute the score

        score = np.mean(np.sum(np.abs(t_true - t_pred), axis=0) / np.sum(t_true, axis=0))

        scores.append(score)

        preds.append(t_pred)

        trues.append(t_true)



    return np.array(scores), np.array(preds), np.array(trues)
def plot_predictions_m(df_pred_mean_tr, df_pred_median_tr, y_tr, scaler=None):



    # Categories

    categories = ['mean', 'median']

    preds_m = [df_pred_mean_tr, df_pred_median_tr]



    # Create subplot grid

    fig = plt.figure(constrained_layout=True, figsize=(11, 5))

    gs = fig.add_gridspec(1, 2)

    ax = [fig.add_subplot(gs[i]) for i in range(2)]



    # Plot discrepancy on training set

    for i, axt in enumerate(ax):

        scores, preds, trues = get_scores(y_tr, preds_m[i], scaler=scaler)

        pred_score = list(np.round(scores, 5))

        axt.scatter(trues[0], trues[1], s=0.5, alpha=0.5);

        axt.scatter(preds[0], preds[1], s=0.5, alpha=0.5);

        axt.set_title('Score on whole train set (var1, var2):\n' + categories[i] + ' - ' + str(pred_score))

        axt.legend(['Target', 'Prediction'])

        axt.set_aspect('equal')



    plt.show()
def extract_score_info(history, n=2):

    tr_mean = np.round(np.mean(history['loss_mean'][-5:]), 6)

    te_mean = np.round(np.mean(history['val_loss_mean'][-5:]), 6)



    res = [tr_mean, te_mean]

    

    for i in range(n):

        res.append(np.round(np.mean(history['loss%d_mean' % (i+1)][-5:]), 6))

        res.append(np.round(np.mean(history['val_loss%d_mean' % (i+1)][-5:]), 6))



    return res
def run_prediction(dataset_id='merge', cv=5, n=2, scale_values=None):



    # Extract dataset

    X_tr, X_te, y_tr = load_dataset_and_scale(dataset_id, scale_values=scale_values)

    

    # Reduce target to number of outputs

    y_tr = y_tr[:, :2]



    # Create grid search object

    grids = create_grid(input_size=X_tr.shape[1])

    print('%0d grid points will be checked!' % len(grids))

    print('\n---Start_Grid_Exploration---\n')

 

    # Go through the grids

    preds_tr = []

    preds_te = []

    df_grids = []

    for gidx, grid in enumerate(tqdm(grids)):



        # Run grid

        print('\nGrid point: %04d/%04d' % (gidx + 1, len(grids)))

        x_train, x_test, history, model, df_pred_mean_tr, df_pred_mean_te, df_pred_median_tr, df_pred_median_te = run_grid(grid, X_tr, X_te, y_tr, cv=cv, n=n)



        # Compute predictions

        preds_tr.append([df_pred_mean_tr, df_pred_median_tr])

        preds_te.append([df_pred_mean_te, df_pred_median_te])



        # Plot history results and predictions

        plot_history_and_predictions(history, [df_pred_mean_tr.T.values,df_pred_median_tr.T.values], y_tr, cutoff=0, n=n)

        plot_predictions_m(df_pred_mean_tr, df_pred_median_tr, y_tr, scaler=scaler_targets)

        

        # Extract scores

        score_means = extract_score_info(history, n=n)



        # Store everything in a grid

        df_grid = pd.DataFrame(columns=grid.keys())

        for k in grid:

            df_grid.loc[0, k] = str(grid[k])

        df_grid.insert(0, 'tr_mean', score_means[0])

        df_grid.insert(1, 'te_mean', score_means[1])

        for i in range(n):

            df_grid.insert(2+(i*2), 'tr_mean%d' % (i+1), score_means[2+(i*2)])

            df_grid.insert(3+(i*2), 'te_mean%d' % (i+1), score_means[3+(i*2)])

        df_grids.append(df_grid)



    # Summarize fit in datatable

    df_fit = pd.concat(df_grids).reset_index().drop(columns='index')

    display(df_fit)

    

    return preds_tr, preds_te, df_fit.iloc[0], df_grid, y_tr, history
# Define feature set scales (~average from my second notebook)

scale_values = [0.25, 0.025, 0.04, 0.02]



# Run model fit

preds_tr, preds_te, top_grid, df_grid, y_tr, history = run_prediction(cv=8, n=2, scale_values=scale_values)



# Feedback of overall score

print('Best score at %s / %s' % (top_grid['te_mean'], top_grid['tr_mean']))
class CustomLoss_dist(keras.losses.Loss):

    def __init__(self, name="loss"):

        super().__init__(name=name)



    @tf.function

    def call(self, y_true, y_pred):



        # Compute distance to target

        score = tf.math.pow(tf.reduce_mean(tf.math.pow(tf.norm(tf.math.subtract(y_true, y_pred), axis=1), 3.)), 1./3.)

        

        return score
class CustomLoss_dist_rot(keras.losses.Loss):

    def __init__(self, scale=None, mean=None, name="loss_rot"):

        super().__init__(name=name)

        self.scale = np.array(scale, dtype='float32')

        self.mean = np.array(mean, dtype='float32')



    @tf.function

    def call(self, y_true, y_pred):



        # Rescale values

        y_true = tf.math.add(tf.math.multiply(self.scale[:2], y_true), self.mean[:2])

        y_pred = tf.math.add(tf.math.multiply(self.scale[:2], y_pred), self.mean[:2])



        # Revert power transformation

        y_true = tf.transpose(tf.stack([

            tf.math.pow(y_true[:, 0], 1./1.5),

            tf.math.pow(y_true[:, 1], 1./1.5)]))

        y_pred = tf.transpose(tf.stack([

            tf.math.pow(y_pred[:, 0], 1./1.5),

            tf.math.pow(y_pred[:, 1], 1./1.5)]))



        # Apply rotation for true and preds

        d2_rot = 0.90771256655

        radians = tf.constant(d2_rot, dtype='float32')

        yt_rot = tf.transpose(tf.stack([

            tf.math.add(tf.math.multiply(y_true[:, 0], [tf.cos(radians)]),

                        tf.math.multiply(y_true[:, 1], [tf.sin(radians)])),

            tf.math.add(-tf.math.multiply(y_true[:, 0], [tf.sin(radians)]),

                        tf.math.multiply(y_true[:, 1], [tf.cos(radians)]))

        ]))

        yp_rot = tf.transpose(tf.stack([

            tf.math.add(tf.math.multiply(y_pred[:, 0], [tf.cos(radians)]),

                        tf.math.multiply(y_pred[:, 1], [tf.sin(radians)])),

            tf.math.add(-tf.math.multiply(y_pred[:, 0], [tf.sin(radians)]),

                        tf.math.multiply(y_pred[:, 1], [tf.cos(radians)]))

        ]))        



        # Rescale values

        yt_rot = tf.math.divide(tf.math.subtract(yt_rot, self.mean[2:]), self.scale[2:])

        yp_rot = tf.math.divide(tf.math.subtract(yp_rot, self.mean[2:]), self.scale[2:])

                

        # Compute distance to target

        score_dist = tf.math.pow(tf.reduce_mean(tf.math.pow(tf.norm(tf.math.subtract(yt_rot, yp_rot), axis=1), 3.)), 1./3.)



        return score_dist
def build_model(input_size=100,

                activation='relu',

                dropout=0.5,

                kernel_initializer='glorot_uniform',

                lr=1e-2,

                use_batch=True,

                use_dropout=True,

                hidden=(32, 16),

                hinge_dist=0.5,

               ):



    # Specify layer structure

    inputs = keras.Input(shape=input_size)

    

    for i, h in enumerate(hidden):

        if i==0:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(inputs)

        else:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(x)

        if use_batch:

            x = layers.BatchNormalization()(x)

        

        x = layers.Activation(activation)(x)



        # Add Dropout layer if requested

        if use_dropout:

            x = layers.Dropout(dropout)(x)



    out = layers.Dense(2)(x)

    out1 = layers.Activation(None, name="loss1")(out)

    out2 = layers.Activation(None, name="loss2")(out)



    # Create Model

    model = keras.Model(inputs, [out1, out2])

    

    # Define loss function, optimizer and metrics to track during training

    optimizer = keras.optimizers.Adagrad(

            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(

                lr, decay_steps=10000, decay_rate=0.96, staircase=True))

    

    l1 = CustomLoss_dist()

    l2 = CustomLoss_dist_rot(scale=scaler_targets.scale_[3:],

                             mean=scaler_targets.mean_[3:])

    

    losses = {'loss1': l1, 'loss2': l2}

    lossWeights = {"loss1": hinge_dist, "loss2": 1-hinge_dist}



    model.compile(

        optimizer = optimizer,

        loss = losses,

        loss_weights = lossWeights

    )

    return model
def create_grid(input_size=0):

    

    # Define parameter grid

    param_grid = [

    { 

        'nn__input_size': [input_size],

        'nn__lr': [0.05],

        'nn__activation': ['sigmoid'],

        'nn__hidden': [(128, 32),],

        'nn__dropout': [0.25],

        'nn__kernel_initializer': ['glorot_uniform'], #['glorot_uniform', 'normal', 'uniform'],

        'nn__use_batch': [True],

        'nn__use_dropout': [True],

        'nn__hinge_dist': [0., 0.5, 1.],

        'batch_size': [32],

        'epochs': [50],

    }]

    

    # Create Parameter Grid and return it

    grids = ParameterGrid(param_grid)

    return grids



len(create_grid())
# Define feature set scales (~average from my second notebook)

scale_values = [0.25, 0.025, 0.04, 0.02]



# Run model fit

preds_tr, preds_te, top_grid, df_grid, y_tr, history = run_prediction(cv=8, n=2, scale_values=scale_values)



# Feedback of overall score

print('Best score at %s / %s' % (top_grid['te_mean'], top_grid['tr_mean']))
def build_model(input_size=100,

                activation='relu',

                dropout=0.5,

                kernel_initializer='glorot_uniform',

                lr=1e-2,

                use_batch=True,

                use_dropout=True,

                hidden=(32, 16),

                hinge=0.5,

                hinge_dist=0.5,

               ):



    # Specify layer structure

    inputs = keras.Input(shape=input_size)

    

    for i, h in enumerate(hidden):

        if i==0:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(inputs)

        else:

            x = layers.Dense(h, kernel_initializer=kernel_initializer)(x)

        if use_batch:

            x = layers.BatchNormalization()(x)

        

        x = layers.Activation(activation)(x)



        # Add Dropout layer if requested

        if use_dropout:

            x = layers.Dropout(dropout)(x)



    out = layers.Dense(2)(x)

    out1 = layers.Activation(None, name="loss1")(out)

    out2 = layers.Activation(None, name="loss2")(out)

    out3 = layers.Activation(None, name="loss3")(out)

    out4 = layers.Activation(None, name="loss4")(out)



    # Create Model

    model = keras.Model(inputs, [out1, out2, out3, out4])



    # Define loss function, optimizer and metrics to track during training

    optimizer = keras.optimizers.Adagrad(

            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(

                lr, decay_steps=10000, decay_rate=0.96, staircase=True))

    

    l1 = CustomLoss_focus(scale=scaler_targets.scale_[3:5],

                          mean=scaler_targets.mean_[3:5],

                          focus=0)

    l2 = CustomLoss_focus(scale=scaler_targets.scale_[3:5],

                          mean=scaler_targets.mean_[3:5],

                          focus=1)

    l3 = CustomLoss_dist()

    l4 = CustomLoss_dist_rot(scale=scaler_targets.scale_[3:],

                             mean=scaler_targets.mean_[3:])

    

    losses = {'loss1': l1, 'loss2': l2, 'loss3': l3, 'loss4': l4}

    adjustment_factor = 1./10.

    lossWeights = {"loss1": hinge,

                   "loss2": 1-hinge,

                   "loss3": (hinge_dist)*adjustment_factor,

                   "loss4": (1-hinge_dist)*adjustment_factor}



    model.compile(

        optimizer = optimizer,

        loss = losses,

        loss_weights = lossWeights

    )

    return model
def create_grid(input_size=0):

    

    # Define parameter grid

    param_grid = [

    { 

        'nn__input_size': [input_size],

        'nn__lr': [0.05],

        'nn__activation': ['sigmoid'],

        'nn__hidden': [(128, 32),],

        'nn__dropout': [0.25],

        'nn__kernel_initializer': ['glorot_uniform'], #['glorot_uniform', 'normal', 'uniform'],

        'nn__use_batch': [True],

        'nn__use_dropout': [True],

        'nn__hinge': [0., 0.5, 1.],

        'nn__hinge_dist': [0., 0.5, 1.],

        'batch_size': [32],

        'epochs': [50],

    }]

    

    # Create Parameter Grid and return it

    grids = ParameterGrid(param_grid)

    return grids



len(create_grid())
# Define feature set scales (~average from my second notebook)

scale_values = [0.25, 0.025, 0.04, 0.02]



# Run model fit

preds_tr, preds_te, top_grid, df_grid, y_tr, history = run_prediction(cv=8, n=4, scale_values=scale_values)



# Feedback of overall score

print('Best score at %s / %s' % (top_grid['te_mean'], top_grid['tr_mean']))