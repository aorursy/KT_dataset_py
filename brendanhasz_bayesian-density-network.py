# Install packages

!pip install -q --upgrade tensorflow==2.0.0

!pip install -q --upgrade tensorflow-probability==0.8.0

!pip install -q catboost

!pip install -q --pre vaex

# pip install -q google-cloud-bigquery #TODO: have to install if not using kaggle kernels
# Packages

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import vaex



from sklearn.dummy import DummyRegressor

from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.model_selection import cross_val_score, cross_val_predict



import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions

from tensorflow_probability.python.math import random_rademacher

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout

from tensorflow.keras.optimizers import Adam



from catboost import CatBoostRegressor



from google.cloud import bigquery



# Settings

sns.set()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

%config InlineBackend.figure_format = 'svg'

np.random.seed(12345)

tf.random.set_seed(12345)
%%time



# SQL Query to retreive the data

# There are 131M records in full dataset!

QUERY = """

    SELECT

        pickup_datetime,

        dropoff_datetime,

        pickup_longitude,

        pickup_latitude,

        dropoff_longitude,

        dropoff_latitude

    FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016`

    LIMIT 1000000;

"""



# Load a subset of the dataset

data = bigquery.Client().query(QUERY).to_dataframe()
# Drop rows with empty values

data.dropna(inplace=True)



# Compute trip duration in seconds

data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.seconds



# Extract useful time information

data['min_of_day'] = (60*data['pickup_datetime'].dt.hour + 

                       data['pickup_datetime'].dt.minute)

data['day_of_week'] = data['pickup_datetime'].dt.dayofweek

data['day_of_year'] = data['pickup_datetime'].dt.dayofyear



# Remove datetime columns

data.drop('pickup_datetime', axis=1, inplace=True)

data.drop('dropoff_datetime', axis=1, inplace=True)



# Function to remove rows outside range

def clip(df, a, b, col):

    for c in col:

        df = df[(df[c]>a) & (df[c]<b)]

    return df



# Remove outliers

data = clip(data, 1, 4*3600, ['trip_duration'])

data = clip(data,  -75, -72.5,

             ['pickup_longitude', 'dropoff_longitude'])

data = clip(data, 40, 41.5,

             ['pickup_latitude', 'dropoff_latitude'])



# Transform target column

data['trip_duration'] = np.log(data['trip_duration'])



# Normalize data

data = (data - data.mean()) / data.std()



# Cast down to float32

data = data.astype('float32')



# Shuffle

data = data.sample(frac=1)



# Separate in- from dependent variables

x_taxi = data.drop('trip_duration', axis=1)

y_taxi = data['trip_duration']
x_taxi.head()
# Plot feature distributions

plt.figure(figsize=(6.4, 8))

for i in range(7):

    plt.subplot(4, 2, i+1)

    plt.hist(x_taxi.iloc[:, i], 13)

    plt.title(x_taxi.columns[i])

plt.tight_layout()

plt.show()
# Plot trip duration distribution

plt.hist(y_taxi, bins=np.linspace(-6, 6, 21))

plt.xlabel('Normalized Trip Duration')
# Make Mean Absolute Error scorer

mae_scorer = make_scorer(mean_absolute_error)



# Function to print cross-validated mean abs deviation

def cv_mae(regressor, x, y, cv=3, scorer=mae_scorer):

    scores = cross_val_score(regressor, 

                             x, y, cv=cv,

                             scoring=scorer)

    print('MAE:', scores.mean())
# MAE from predicting just the mean

cv_mae(DummyRegressor(), x_taxi, y_taxi)
%%time



# Distance between pickup and dropoff locations

dist = np.sqrt(

    np.power(x_taxi['pickup_longitude'] -

             x_taxi['dropoff_longitude'], 2) + 

    np.power(x_taxi['pickup_latitude'] - 

             x_taxi['dropoff_latitude'], 2))



# MAE from using just distance as predictor

cv_mae(IsotonicRegression(out_of_bounds='clip'), 

       dist, y_taxi)
%%time



# MAE using CatBoost

cv_mae(CatBoostRegressor(verbose=False, depth=9), x_taxi, y_taxi)
# Batch size

BATCH_SIZE = 1024



# Number of training epochs

EPOCHS = 100



# Learning rate

L_RATE = 1e-4



# Proportion of samples to hold out

VAL_SPLIT = 0.2
# Multilayer dense neural network

D = x_taxi.shape[1]

model = Sequential([

    Dense(512, use_bias=False, input_shape=(D,)),

    BatchNormalization(),

    ReLU(),

    Dropout(0.1),

    Dense(128, use_bias=False),

    BatchNormalization(),

    ReLU(),

    Dropout(0.1),

    Dense(64, use_bias=False),

    BatchNormalization(),

    ReLU(),

    Dropout(0.1),

    Dense(32, use_bias=False),

    BatchNormalization(),

    ReLU(),

    Dropout(0.1),

    Dense(1)

])
# Compile the model with MAE loss

model.compile(tf.keras.optimizers.Adam(lr=L_RATE),

              loss='mean_absolute_error')
%%time



# Fit the model

history = model.fit(x_taxi, y_taxi,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_split=VAL_SPLIT,

                    verbose=0)
plt.plot(history.history['loss'], label='Train')

plt.plot(history.history['val_loss'], label='Val')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('MAE')

plt.show()
# Make predictions (tautological)

preds = model.predict(x_taxi)



# Plot true vs predicted durations

plt.figure(figsize=(6.4, 8))

for i in range(8):

    plt.subplot(4,2,i+1)

    plt.axvline(preds[i], label='Pred')

    plt.axvline(y_taxi[i], ls=':', color='gray', label='True')

    plt.xlim([-5, 5])

    plt.legend()

    plt.gca().get_yaxis().set_ticklabels([])

    if i<6:

        plt.gca().get_xaxis().set_ticklabels([])
# Split data randomly into training + validation

tr_ind = np.random.choice([False, True],

                          size=x_taxi.shape[0],

                          p=[VAL_SPLIT, 1.0-VAL_SPLIT])

x_train = x_taxi[tr_ind].values

y_train = y_taxi[tr_ind].values

x_val = x_taxi[~tr_ind].values

y_val = y_taxi[~tr_ind].values

N_train = x_train.shape[0]

N_val = x_val.shape[0]



# Make y 2d

y_train = np.expand_dims(y_train, 1)

y_val = np.expand_dims(y_val, 1)



# Make a TensorFlow Dataset from training data

data_train = tf.data.Dataset.from_tensor_slices(

    (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)



# Make a TensorFlow Dataset from validation data

data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(N_val)
# Xavier initializer

def xavier(shape):

    return tf.random.truncated_normal(

        shape, 

        mean=0.0,

        stddev=np.sqrt(2/sum(shape)))
class BayesianDenseLayer(tf.keras.Model):

    """A fully-connected Bayesian neural network layer

    

    Parameters

    ----------

    d_in : int

        Dimensionality of the input (# input features)

    d_out : int

        Output dimensionality (# units in the layer)

    name : str

        Name for the layer

        

    Attributes

    ----------

    weight : tensorflow_probability.distributions.Normal

        Variational distributions for the network weights

    bias : tensorflow_probability.distributions.Normal

        Variational distributions for the network biases

    losses : tensorflow.Tensor

        Sum of the Kullback–Leibler divergences between

        the posterior distributions and their priors

        

    Methods

    -------

    call : tensorflow.Tensor

        Perform the forward pass of the data through

        the layer

    """



    def __init__(self, d_in, d_out, name=None):

        super(BayesianDenseLayer, self).__init__(name=name)

        self.w_loc = tf.Variable(xavier([d_in, d_out]), name='w_loc')

        self.w_std = tf.Variable(xavier([d_in, d_out])-6.0, name='w_std')

        self.b_loc = tf.Variable(xavier([1, d_out]), name='b_loc')

        self.b_std = tf.Variable(xavier([1, d_out])-6.0, name='b_std')

        

    

    @property

    def weight(self):

        return tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))

    

    

    @property

    def bias(self):

        return tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))

        

        

    def call(self, x, sampling=True):

        if sampling:

            return x @ self.weight.sample() + self.bias.sample()        

        else:

            return x @ self.w_loc + self.b_loc

            

            

    @property

    def losses(self):

        prior = tfd.Normal(0, 1)

        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +

                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))
class BayesianDenseLayer(tf.keras.Model):

    """A fully-connected Bayesian neural network layer

    

    Parameters

    ----------

    d_in : int

        Dimensionality of the input (# input features)

    d_out : int

        Output dimensionality (# units in the layer)

    name : str

        Name for the layer

        

    Attributes

    ----------

    losses : tensorflow.Tensor

        Sum of the Kullback–Leibler divergences between

        the posterior distributions and their priors

        

    Methods

    -------

    call : tensorflow.Tensor

        Perform the forward pass of the data through

        the layer

    """

    

    def __init__(self, d_in, d_out, name=None):

        

        super(BayesianDenseLayer, self).__init__(name=name)

        self.d_in = d_in

        self.d_out = d_out

        

        self.w_loc = tf.Variable(xavier([d_in, d_out]), name='w_loc')

        self.w_std = tf.Variable(xavier([d_in, d_out])-6.0, name='w_std')

        self.b_loc = tf.Variable(xavier([1, d_out]), name='b_loc')

        self.b_std = tf.Variable(xavier([1, d_out])-6.0, name='b_std')

    

    

    def call(self, x, sampling=True):

        """Perform the forward pass"""

        

        if sampling:

        

            # Flipout-estimated weight samples

            s = random_rademacher(tf.shape(x))

            r = random_rademacher([x.shape[0], self.d_out])

            w_samples = tf.nn.softplus(self.w_std)*tf.random.normal([self.d_in, self.d_out])

            w_perturbations = r*tf.matmul(x*s, w_samples)

            w_outputs = tf.matmul(x, self.w_loc) + w_perturbations

            

            # Flipout-estimated bias samples

            r = random_rademacher([x.shape[0], self.d_out])

            b_samples = tf.nn.softplus(self.b_std)*tf.random.normal([self.d_out])

            b_outputs = self.b_loc + r*b_samples

            

            return w_outputs + b_outputs

        

        else:

            return x @ self.w_loc + self.b_loc

    

    

    @property

    def losses(self):

        """Sum of the KL divergences between priors + posteriors"""

        weight = tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))

        bias = tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))

        prior = tfd.Normal(0, 1)

        return (tf.reduce_sum(tfd.kl_divergence(weight, prior)) +

                tf.reduce_sum(tfd.kl_divergence(bias, prior)))
class BayesianDenseNetwork(tf.keras.Model):

    """A multilayer fully-connected Bayesian neural network

    

    Parameters

    ----------

    dims : List[int]

        List of units in each layer

    name : str

        Name for the network

        

    Attributes

    ----------

    losses : tensorflow.Tensor

        Sum of the Kullback–Leibler divergences between

        the posterior distributions and their priors, 

        over all layers in the network

        

    Methods

    -------

    call : tensorflow.Tensor

        Perform the forward pass of the data through

        the network

    """

    

    def __init__(self, dims, name=None):

        

        super(BayesianDenseNetwork, self).__init__(name=name)

        

        self.steps = []

        self.acts = []

        for i in range(len(dims)-1):

            self.steps += [BayesianDenseLayer(dims[i], dims[i+1])]

            self.acts += [tf.nn.relu]

            

        self.acts[-1] = lambda x: x

        

    

    def call(self, x, sampling=True):

        """Perform the forward pass"""



        for i in range(len(self.steps)):

            x = self.steps[i](x, sampling=sampling)

            x = self.acts[i](x)

            

        return x

    

    

    @property

    def losses(self):

        """Sum of the KL divergences between priors + posteriors"""

        return tf.reduce_sum([s.losses for s in self.steps])
class BayesianDenseRegression(tf.keras.Model):

    """A multilayer fully-connected Bayesian neural network regression

    

    Parameters

    ----------

    dims : List[int]

        List of units in each layer

    name : str

        Name for the network

        

    Attributes

    ----------

    losses : tensorflow.Tensor

        Sum of the Kullback–Leibler divergences between

        the posterior distributions and their priors, 

        over all layers in the network

        

    Methods

    -------

    call : tensorflow.Tensor

        Perform the forward pass of the data through

        the network, predicting both means and stds

    log_likelihood : tensorflow.Tensor

        Compute the log likelihood of y given x

    samples : tensorflow.Tensor

        Draw multiple samples from the predictive distribution

    """    

    

    

    def __init__(self, dims, name=None):

        

        super(BayesianDenseRegression, self).__init__(name=name)

        

        # Multilayer fully-connected neural network to predict mean

        self.loc_net = BayesianDenseNetwork(dims)

        

        # Variational distribution variables for observation error

        self.std_alpha = tf.Variable([10.0], name='std_alpha')

        self.std_beta = tf.Variable([10.0], name='std_beta')



    

    def call(self, x, sampling=True):

        """Perform the forward pass, predicting both means and stds"""

        

        # Predict means

        loc_preds = self.loc_net(x, sampling=sampling)

    

        # Predict std deviation

        posterior = tfd.Gamma(self.std_alpha, self.std_beta)

        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))

        N = x.shape[0]

        if sampling:

            std_preds = transform(posterior.sample([N]))

        else:

            std_preds = tf.ones([N, 1])*transform(posterior.mean())

    

        # Return mean and std predictions

        return tf.concat([loc_preds, std_preds], 1)

    

    

    def log_likelihood(self, x, y, sampling=True):

        """Compute the log likelihood of y given x"""

        

        # Compute mean and std predictions

        preds = self.call(x, sampling=sampling)

        

        # Return log likelihood of true data given predictions

        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])

    

    

    @tf.function

    def sample(self, x):

        """Draw one sample from the predictive distribution"""

        preds = self.call(x)

        return tfd.Normal(preds[:,0], preds[:,1]).sample()

    

    

    def samples(self, x, n_samples=1):

        """Draw multiple samples from the predictive distribution"""

        samples = np.zeros((x.shape[0], n_samples))

        for i in range(n_samples):

            samples[:,i] = self.sample(x)

        return samples

    

    

    @property

    def losses(self):

        """Sum of the KL divergences between priors + posteriors"""

                

        # Loss due to network weights

        net_loss = self.loc_net.losses



        # Loss due to std deviation parameter

        posterior = tfd.Gamma(self.std_alpha, self.std_beta)

        prior = tfd.Gamma(10.0, 10.0)

        std_loss = tfd.kl_divergence(posterior, prior)



        # Return the sum of both

        return net_loss + std_loss
model1 = BayesianDenseRegression([7, 256, 128, 64, 32, 1])
# Adam optimizer

optimizer = tf.keras.optimizers.Adam(lr=L_RATE)
N = x_train.shape[0]



@tf.function

def train_step(x_data, y_data):

    with tf.GradientTape() as tape:

        log_likelihoods = model1.log_likelihood(x_data, y_data)

        kl_loss = model1.losses

        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)

    gradients = tape.gradient(elbo_loss, model1.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))

    return elbo_loss
%%time



# Fit the model

elbo1 = np.zeros(EPOCHS)

mae1 = np.zeros(EPOCHS)

for epoch in range(EPOCHS):

    

    # Update weights each batch

    for x_data, y_data in data_train:

        elbo1[epoch] += train_step(x_data, y_data)

        

    # Evaluate performance on validation data

    for x_data, y_data in data_val:

        y_pred = model1(x_data, sampling=False)[:, 0]

        mae1[epoch] = mean_absolute_error(y_pred, y_data)
# Plot the ELBO loss

plt.plot(elbo1)

plt.xlabel('Epoch')

plt.ylabel('ELBO Loss')

plt.show()
# Plot validation error over training

plt.plot(mae1)

plt.xlabel('Epoch')

plt.ylabel('Mean Absolute Error')

plt.show()
class BayesianDensityNetwork(tf.keras.Model):

    """Multilayer fully-connected Bayesian neural network, with

    two heads to predict both the mean and the standard deviation.

    

    Parameters

    ----------

    units : List[int]

        Number of output dimensions for each layer

        in the core network.

    units : List[int]

        Number of output dimensions for each layer

        in the head networks.

    name : None or str

        Name for the layer

    """

    

    

    def __init__(self, units, head_units, name=None):

        

        # Initialize

        super(BayesianDensityNetwork, self).__init__(name=name)

        

        # Create sub-networks

        self.core_net = BayesianDenseNetwork(units)

        self.loc_net = BayesianDenseNetwork([units[-1]]+head_units)

        self.std_net = BayesianDenseNetwork([units[-1]]+head_units)



    

    def call(self, x, sampling=True):

        """Pass data through the model

        

        Parameters

        ----------

        x : tf.Tensor

            Input data

        sampling : bool

            Whether to sample parameter values from their variational

            distributions (if True, the default), or just use the

            Maximum a Posteriori parameter value estimates (if False).

            

        Returns

        -------

        preds : tf.Tensor of shape (Nsamples, 2)

            Output of this model, the predictions.  First column is

            the mean predictions, and second column is the standard

            deviation predictions.

        """

        

        # Pass data through core network

        x = self.core_net(x, sampling=sampling)

        x = tf.nn.relu(x)

        

        # Make predictions with each head network

        loc_preds = self.loc_net(x, sampling=sampling)

        std_preds = self.std_net(x, sampling=sampling)

        std_preds = tf.nn.softplus(std_preds)

        

        # Return mean and std predictions

        return tf.concat([loc_preds, std_preds], 1)

    

    

    def log_likelihood(self, x, y, sampling=True):

        """Compute the log likelihood of y given x"""

        

        # Compute mean and std predictions

        preds = self.call(x, sampling=sampling)

        

        # Return log likelihood of true data given predictions

        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])

        

        

    @tf.function

    def sample(self, x):

        """Draw one sample from the predictive distribution"""

        preds = self.call(x)

        return tfd.Normal(preds[:,0], preds[:,1]).sample()

    

    

    def samples(self, x, n_samples=1):

        """Draw multiple samples from the predictive distribution"""

        samples = np.zeros((x.shape[0], n_samples))

        for i in range(n_samples):

            samples[:,i] = self.sample(x)

        return samples

    

    

    @property

    def losses(self):

        """Sum of the KL divergences between priors + posteriors"""

        return (self.core_net.losses +

                self.loc_net.losses +

                self.std_net.losses)
# Instantiate the model

model2 = BayesianDensityNetwork([7, 256, 128], [64, 32, 1])
# Use the Adam optimizer

optimizer = tf.keras.optimizers.Adam(lr=L_RATE)
N = x_train.shape[0]



@tf.function

def train_step(x_data, y_data):

    with tf.GradientTape() as tape:

        log_likelihoods = model2.log_likelihood(x_data, y_data)

        kl_loss = model2.losses

        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)

    gradients = tape.gradient(elbo_loss, model2.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))

    return elbo_loss
%%time



# Fit the model

elbo2 = np.zeros(EPOCHS)

mae2 = np.zeros(EPOCHS)

for epoch in range(EPOCHS):

    

    # Update weights each batch

    for x_data, y_data in data_train:

        elbo2[epoch] += train_step(x_data, y_data)

        

    # Evaluate performance on validation data

    for x_data, y_data in data_val:

        y_pred = model2(x_data, sampling=False)[:, 0]

        mae2[epoch] = mean_absolute_error(y_pred, y_data)
# Plot the ELBO Loss over training

plt.plot(elbo2)

plt.xlabel('Epoch')

plt.ylabel('ELBO Loss')

plt.show()
# Plot error over training

plt.plot(mae2)

plt.xlabel('Epoch')

plt.ylabel('Mean Absolue Error')

plt.show()
# Plot error vs epoch curves for all 3 models

plt.plot(mae1, label='No Error Estimation')

plt.plot(mae2, label='Density Network')

plt.plot(history.history['val_loss'], label='Non-Bayesian')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Mean Absolute Error')

plt.show()
# Make predictions on validation data

for x_data, y_data in data_val:

    resids1 = y_data[:, 0] - model1(x_data, sampling=False)[:, 0]

    resids2 = y_data[:, 0] - model2(x_data, sampling=False)[:, 0]

    

# Plot residual distributions

bins = np.linspace(-2, 2, 100)

plt.hist(resids1.numpy(), bins, alpha=0.5,

         label='No Error Estimation')

plt.hist(resids2.numpy(), bins, alpha=0.5,

         label='Density Network')

plt.legend()

plt.xlabel('Residuals')

plt.ylabel('Count')
%%time



# Sample from predictive distributions

for x_data, y_data in data_val:

    samples1 = model1.samples(x_data, 1000)

    samples2 = model2.samples(x_data, 1000)
# Plot predictive distributions

plt.figure(figsize=(6.4, 8))

for i in range(8):

    plt.subplot(4,2,i+1)

    sns.kdeplot(samples1[i,:], shade=True)

    sns.kdeplot(samples2[i,:], shade=True)

    plt.axvline(y_data.numpy()[i], ls=':', color='gray')

    plt.xlim([-5, 5])

    plt.ylim([0, 2.2])

    plt.title(str(i))

    plt.gca().get_yaxis().set_ticklabels([])

    if i<6:

        plt.gca().get_xaxis().set_ticklabels([])
%%time



def covered(samples, y_true, prc=95.0):

    """Whether each sample was covered by its predictive interval"""

    q0 = (100.0-prc)/2.0 #lower percentile 

    q1 = 100.0-q0        #upper percentile

    within_conf_int = np.zeros(len(y_true))

    for i in range(len(y_true)):

        p0 = np.percentile(samples[i,:], q0)

        p1 = np.percentile(samples[i,:], q1)

        if p0<=y_true[i] and p1>y_true[i]:

            within_conf_int[i] = 1

    return within_conf_int



# Compute what samples are covered by their 95% predictive intervals

covered1 = covered(samples1, y_data)

covered2 = covered(samples2, y_data)
print('No Unc Estimation: ', 100*np.mean(covered1))

print('Density Network: ', 100*np.mean(covered2))
# Compute hour of the day

hour = x_data[:,4].numpy()

hour = hour-hour.min()

hour = hour/hour.max()

hour = np.floor(23.99*hour)



# Compute coverage as a fn of time of day

covs1 = np.zeros(24)

covs2 = np.zeros(24)

for iT in range(0,24):

    ix = hour==iT

    covs1[iT] = 100.0*np.mean(covered1[ix])

    covs2[iT] = 100.0*np.mean(covered2[ix])

    

# Plot coverage as a fn of time of day

plt.plot(covs1, label='No Error Estimation')

plt.plot(covs2, label='Density Network')

plt.axhline(95.0, label='Ideal', ls=':', color='k')

plt.xlabel('Hour')

plt.ylabel('95% Predictive Interval Coverage')

plt.title('Coverage of 95% Predictive Interval by Hour')

plt.ylim([90, 100])

plt.legend()

plt.show()
# Create vaex df with predictive intervals

cov_by_loc = pd.DataFrame()

cov_by_loc['x'] = x_data[:, 0].numpy()

cov_by_loc['y'] = x_data[:, 1].numpy()

cov_by_loc['covered'] = covered1

vdf = vaex.from_pandas(cov_by_loc)



# Compute coverage of the predictive interval

lims = [[-3, 3.5],[-4, 4]]

cov = vdf.mean(vdf.covered, limits=lims, shape=250,

               binby=[vdf.x,

                      vdf.y])



# Plot coverage of the predictive interval

cmap = matplotlib.cm.PuOr

cmap.set_bad('black', 1.)

plt.imshow(cov.T, origin='lower', aspect='auto',

           vmin=0.9, vmax=1.0, cmap=cmap,

           extent=[lims[0][0], lims[0][1], 

                   lims[1][0], lims[1][1]])

ax = plt.gca()

ax.grid(False)

cbar = plt.colorbar()

cbar.set_label('Coverage of 95% predictive interval', 

               rotation=270)

plt.title('No Error Estimation')
# Create vaex df with predictive intervals

cov_by_loc['covered'] = covered2

vdf = vaex.from_pandas(cov_by_loc)



# Compute coverage of the predictive interval

cov = vdf.mean(vdf.covered, limits=lims, shape=250,

               binby=[vdf.x,

                      vdf.y])



# Plot coverage of the predictive interval

plt.imshow(cov.T, origin='lower', aspect='auto',

           vmin=0.9, vmax=1.0, cmap=cmap,

           extent=[lims[0][0], lims[0][1], 

                   lims[1][0], lims[1][1]])

ax = plt.gca()

ax.grid(False)

cbar = plt.colorbar()

cbar.set_label('Coverage of 95% predictive interval', 

               rotation=270)

plt.title('Density Network')