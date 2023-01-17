import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from sklearn.model_selection import train_test_split
# Load the data
train = pd.read_csv("../input/train.csv")
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 
# Normalize the data
X_train = X_train / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
# create an anomaly hold out group and then only train with the remaining digits
anom_mask = (Y_train==9)
anomaly_test = X_train[anom_mask]
X_train = X_train[~anom_mask]
Y_train = Y_train[~anom_mask]
# make a test set the same size as the anomaly set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=anomaly_test.shape[0], random_state=42)
print('Training Images', X_train.shape, 'Testing Images', X_test.shape, 'Anomaly Images', anomaly_test.shape)
batch_size = 256
original_shape = X_train.shape[1:]
original_dim = np.prod(original_shape)
latent_dim = 4
intermediate_dim = 128
final_dim = 64
epochs = 50
epsilon_std = 1.0
in_layer = Input(shape=original_shape)
x = Flatten()(in_layer)
h = Dense(intermediate_dim, activation='relu')(x)
h = Dense(final_dim, activation = 'relu')(h)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_f = Dense(final_dim, activation='relu')
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

f_decoded = decoder_f(z)
h_decoded = decoder_h(f_decoded)
x_decoded_mean = decoder_mean(h_decoded)
x_decoded_img = Reshape(original_shape)(x_decoded_mean)

# instantiate VAE model
vae = Model(in_layer, x_decoded_img)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
vae.fit(X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(anomaly_test, None))
encoder = Model(in_layer, z_mean)
# display a 2D plot of the digit classes in the latent space
X_test_encoded = encoder.predict(X_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c=Y_test)
plt.colorbar()
plt.show()
anomaly_encoded = encoder.predict(anomaly_test, batch_size=batch_size)
fig, m_axs = plt.subplots(latent_dim,latent_dim, figsize=(latent_dim*5, latent_dim*5))
if latent_dim == 1:
    m_axs = [[m_axs]]
for i, n_axs in enumerate(m_axs, 0):
    for j, c_ax in enumerate(n_axs, 0):
        c_ax.scatter(np.concatenate([X_test_encoded[:, i], anomaly_encoded[:,i]],0), 
                           np.concatenate([X_test_encoded[:, j], anomaly_encoded[:,j]],0),
            c=(['g']*X_test_encoded.shape[0])+['r']*anomaly_encoded.shape[0], alpha = 0.5)
from sklearn.manifold.t_sne import TSNE
latent_space_tsne = TSNE(2, verbose = True, n_iter = 500)
xa_tsne = latent_space_tsne.fit_transform(np.concatenate([X_test_encoded[:, :], anomaly_encoded[:,:]],0))

plt.figure(figsize=(6, 6))
plt.scatter(xa_tsne[:,0], xa_tsne[:,1],
            c=(['g']*X_test_encoded.shape[0])+['r']*anomaly_encoded.shape[0], alpha = 0.5)
plt.show()
model_mse = lambda x: np.mean(np.square(x-vae.predict(x, batch_size = batch_size)), (1,2,3))
fig, ax1 = plt.subplots(1,1, figsize = (8,8))
ax1.hist(model_mse(X_train), bins = np.linspace(0, .1, 50), label = 'Training Digits', normed = True, alpha = 1.0)
ax1.hist(model_mse(X_test), bins = np.linspace(0, .1, 50), label = 'Testing Digits', normed = True, alpha = 0.5)
ax1.hist(model_mse(anomaly_test), bins = np.linspace(0, .1, 50), label = 'Anomaly Digits', normed = True, alpha = 0.5)
ax1.legend()
ax1.set_xlabel('Reconstruction Error');
from sklearn.neighbors import KernelDensity
kd = KernelDensity()
kd.fit(encoder.predict(X_train))
test_score = [kd.score(x.reshape(1, -1)) for x in X_test_encoded]
anom_score = [kd.score(x.reshape(1, -1)) for x in anomaly_encoded]
fig, ax1 = plt.subplots(1,1, figsize = (8,8))
ax1.hist(test_score, label = 'Test Digits', normed = True, alpha = 1.0)
ax1.hist(anom_score, label = 'Anomaly Digits', normed = True, alpha = 0.5)
ax1.legend()
print('Test data score', np.mean(test_score))
print('Anomaly score', np.mean(anom_score))
from sklearn.metrics import roc_auc_score, roc_curve
mse_score = np.concatenate([model_mse(X_test), model_mse(anomaly_test)],0)
true_label = [0]*X_test.shape[0]+[1]*anomaly_test.shape[0]
if roc_auc_score(true_label, mse_score)<0.5:
    mse_score *= -1
fpr, tpr, thresholds = roc_curve(true_label, mse_score)
auc_score = roc_auc_score(true_label, mse_score)
fig, ax1 = plt.subplots(1, 1, figsize = (8, 8))
ax1.plot(fpr, tpr, 'b.-', label = 'ROC Curve (%2.2f)' %  auc_score)
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend();
anom_exam = np.mean(xa_tsne[-3:,:],0)
mse_score_train = np.sqrt(np.square(xa_tsne[:,0]-anom_exam[0])+np.square(xa_tsne[:,1]-anom_exam[1]))
if roc_auc_score(true_label, mse_score_train)<0.5:
    mse_score_train *= -1

fpr_new, tpr_new, thresholds = roc_curve(true_label, mse_score_train)
auc_score_new = roc_auc_score(true_label, mse_score_train)
fig, ax1 = plt.subplots(1, 1, figsize = (8, 8))
ax1.plot(fpr, tpr, 'b.-', label = 'ROC Curve (%2.2f)' %  auc_score)
ax1.plot(fpr_new, tpr_new, 'g.-', label = 'Trained ROC Curve (%2.2f)' %  auc_score_new)
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend();
decoder_input = Input(shape=(latent_dim,))
_f_decoded = decoder_f(decoder_input)
_h_decoded = decoder_h(_f_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_img = Reshape(original_shape)(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_img)
from skimage.util.montage import montage2d
from itertools import product
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
n = 9
scale_d = norm.ppf(np.linspace(0.05, 0.95, n))

fig, m_axs = plt.subplots(latent_dim,latent_dim, figsize=(latent_dim*5, latent_dim*5))
if latent_dim == 1:
    m_axs = [[m_axs]]
for i, n_axs in enumerate(m_axs, 0):
    for j, c_ax in enumerate(n_axs, 0):
        test_dims = [[0.5]]*latent_dim
        test_dims[i] = scale_d
        test_dims[j] = scale_d
        stack_in = np.stack(product(*test_dims),0)
        stack_out = montage2d(generator.predict(stack_in)[:,:,:,0])
        c_ax.imshow(stack_out, cmap='Greys_r')
        c_ax.axis('off')
        c_ax.set_title('{} vs {}'.format(i, j))
fig, m_axs = plt.subplots(5,4, figsize=(20, 10))
[c_ax.axis('off') for c_ax in m_axs.ravel()]
for i, (axa_in, axa_vae, axt_in, axt_vae) in enumerate(m_axs):
    axa_in.imshow(anomaly_test[i,:,:,0])
    axa_in.set_title('Anomaly In')
    axa_vae.imshow(vae.predict(anomaly_test[i:i+1])[0,:,:,0])
    axa_vae.set_title('Anomaly\nDreamed/Reconstructed')
    axt_in.imshow(X_test[i,:,:,0])
    axt_in.set_title('Test In')
    axt_vae.imshow(vae.predict(X_test[i:i+1])[0,:,:,0])
    axt_vae.set_title('Test Reconstructed')
fig, m_axs = plt.subplots(2,3, figsize=(20, 10))
sg_anom = anomaly_test[:100]
sg_test = X_test[:100]
for (ax_tt, ax_an), (c_name, c_func) in zip(m_axs.T, [('Input', lambda x: x), 
                                                   ('VAE', vae.predict),
                                                    ('Error', lambda x: x-vae.predict(x))
                                                   ]):
    c_an_dig = c_func(sg_anom)
    if c_an_dig.min()<0:
        plt_kwargs = dict(vmin = -0.5, vmax = 0.5, cmap = 'RdBu')
    else:
        plt_kwargs = dict(cmap = 'bone')
    ax_an.imshow(montage2d(c_an_dig[:,:,:,0]), **plt_kwargs)
    ax_an.set_title('Anomalous Digits - {}'.format(c_name))
    ax_an.axis('off')
    ax_tt.imshow(montage2d(c_func(sg_test)[:,:,:,0]), **plt_kwargs)
    ax_tt.set_title('Test Digits - {}'.format(c_name))
    ax_tt.axis('off')
