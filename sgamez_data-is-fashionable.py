import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
#data reading
train_df = pd.read_csv('../input/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashion-mnist_train.csv')

#Nrows, Ncols @ train
train_df.shape
train_df.shape
label_dict = {0: 'tshirt',
              1: 'trouser',
              2: 'pullover',
              3: 'dress',
              4: 'coat',
              5: 'sandal',
              6: 'shirt',
              7: 'sneaker',
              8: 'bag',
              9: 'boot'}
#header
train_df.head()
#plot an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def get_pixel_cols():
    """
    This function returns the pixel column names
    """
    return ['pixel' + str(i) for i in range(1, 785)]
def idx_to_pixseries(df, idx):
    """
    Given a pandas dataframe, and an index, it returns the pixel series for that index
    """
    return df.iloc[idx][get_pixel_cols()]
def plot_image_pd(pixels_series):
    """
    This functions plots an image, given a series with all the pixels
    """
    pix_mat = pixels_series.values.reshape(28, 28)
    imgplot = plt.imshow(pix_mat, cmap='gray')
plot_image_pd(idx_to_pixseries(train_df, 3))
labels = train_df.label.value_counts().index.values.tolist()
labels = sorted(labels)
plt.figure(figsize=(10,10))
plt.plot([4, 3, 11])
for lab in labels:
    ax = plt.subplot(4, 3, lab+1)
    ax.set_title(str(lab) + " - " + label_dict[lab])
    plt.axis('off')
    plot_image_pd(idx_to_pixseries(train_df, train_df[train_df.label == lab].index[0]))
#N images per row
N_im_lab = 6
N_labs = len(labels)
plt.figure(figsize=(11,11))
plt.plot([N_labs, N_im_lab, (N_im_lab * N_labs) + 1])

#for every label
for lab in labels:
    #show N_im_lab first samples
    for i in range(N_im_lab):
        ax = plt.subplot(N_labs, N_im_lab, 1 + (i + (lab*N_im_lab)))
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, train_df[train_df.label == lab].index[i]))
plt.figure(figsize=(10,10))
plt.plot([4, 3, 11])
for lab in labels:
    ax = plt.subplot(4, 3, lab+1)
    ax.set_title("Avg. " + str(lab) + " - " + label_dict[lab])
    plt.axis('off')
    avg_pixels = train_df.loc[train_df.label == lab][get_pixel_cols()].mean()
    plot_image_pd(avg_pixels)
#normalize data, so we get values between 0 and 1
train_df[get_pixel_cols()] = train_df[get_pixel_cols()] / 255.
test_df[get_pixel_cols()] = test_df[get_pixel_cols()] / 255.
#split train data in train-val
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df[get_pixel_cols()], train_df.label, test_size=0.25, random_state=4)
#train a logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 0.1, solver = 'sag')
% time lr.fit(X_train, y_train)
#class estimation
lr_y_val_pred = lr.predict(X_val)
#prints accuracy score
def print_acc(y_true, y_pred, set_str):
    print ("This model has a {0:.2f}% acc. score @ {1}".format(100*accuracy_score(y_true, y_pred), set_str))
#compute confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def plot_conf_matrix(y_true, y_pred, set_str):
    """
    This function plots a basic confusion matrix, and also shows the model
    accuracy score
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    df_conf = pd.DataFrame(conf_mat, index = [str(l) + '-' + label_dict[l] for l in labels],
                           columns = [str(l) + '-' + label_dict[l] for l in labels])

    plt.figure(figsize = (12, 12))
    sn.heatmap(df_conf, annot=True, cmap="YlGnBu")
    
    print_acc(y_true, y_pred, set_str)

plot_conf_matrix(y_val, lr_y_val_pred, 'Validation')
print_acc(y_train, lr.predict(X_train), 'Train')
def visual_err_inspection(y_true, y_pred, lab_eval, N_samples=6):
    """
    This function runs a visual error inspection. It plots two rows of images,
    the first row shows true positive predictions, while the second one shows
    flase positive predictions
    """
    
    df_y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    idx_y_eval_tp = df_y.loc[(df_y.y_true == lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    idx_y_eval_fp = df_y.loc[(df_y.y_true != lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    
    plt.figure(figsize=(12,5))
    plt.plot([2, N_samples, 2*N_samples + 1])

    for i in range(N_samples):
        ax = plt.subplot(2, N_samples, i+1)
        ax.set_title("OK: " + str(lab_eval) + " - " + label_dict[lab_eval])
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, idx_y_eval_tp[i]))

        ax2 = plt.subplot(2, N_samples, i+N_samples+1)
        lab_ = train_df.iloc[idx_y_eval_fp[i]].label
        ax2.set_title("KO: " + str(int(lab_)) + " - " + label_dict[lab_])
        plt.axis('off')
        plot_image_pd(idx_to_pixseries(train_df, idx_y_eval_fp[i]))
#run visual inspection for class 6 - shirts
visual_err_inspection(y_val, lr_y_val_pred, 6, 6)
#run visual inspection for class 4 - coats
visual_err_inspection(y_val, lr_y_val_pred, 4, 6)
#train a RF model on this data
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=25, n_jobs=4)
#train the model
%time rf.fit(X_train, y_train)

plot_conf_matrix(y_val, rf.predict(X_val), 'Validation')
print_acc(y_train, rf.predict(X_train), 'Train')
#train a xgboost model on this data
from xgboost import  XGBClassifier
xgb = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.03, n_jobs=4)
%time xgb.fit(X_train, y_train)
plot_conf_matrix(y_val, xgb.predict(X_val), 'Validation')
print_acc(y_train, xgb.predict(X_train), 'Train')
lr_val_probs, rf_val_probs, xgb_val_probs = lr.predict_proba(X_val), rf.predict_proba(X_val), xgb.predict_proba(X_val)
def show_probs_dist(prob_pred, label_dictionary = label_dict):
    '''
    Given the probabilities prediction of a model, for a set of labels, show how all look together 
    '''
    for i in range(prob_pred.shape[1]):
        plt.hist(prob_pred[:,i], alpha=0.4, label = label_dictionary[i])
    plt.legend(loc='upper right')
show_probs_dist(lr_val_probs)
show_probs_dist(rf_val_probs)
show_probs_dist(xgb_val_probs)
def show_probs_dist_label(prob_pred_list, preds_names_list, label_picked, label_dictionary = label_dict):
    '''
    Given a list of probablities prediction for different models, select one label an plot all models 
    probabilities prediction for that label
    '''
    for i in range(len(prob_pred_list)):
        plt.hist(prob_pred_list[i][:,label_picked], alpha=0.5, label = preds_names_list[i])
    plt.legend(loc='upper right')
    plt.title('Probability distribution for ' + label_dict[label_picked])
    plt.show()
list_probs = [lr_val_probs, rf_val_probs, xgb_val_probs]
list_names = ['LR', 'RF', 'XGB']
for i in label_dict.keys():
    show_probs_dist_label(list_probs, list_names, i)
#average all predictions
comb_val_probs = (lr_val_probs + rf_val_probs + xgb_val_probs) / 3
#pick the column idx with the max probability
comb_val_pred = comb_val_probs.argmax(axis=1)
plot_conf_matrix(y_val, comb_val_pred, 'Validation')
weighted_val_probs = (0.2*lr_val_probs + 0.3*rf_val_probs + 0.5*xgb_val_probs) / 3
weighted_val_pred = weighted_val_probs.argmax(axis=1)
plot_conf_matrix(y_val, weighted_val_pred, 'Validation')
from sklearn import decomposition
N_PCA_COMP = 50
pca_est = decomposition.PCA(n_components=N_PCA_COMP, svd_solver='randomized', whiten=True, random_state=42)
pca_est.fit(X_val)
pca_est.explained_variance_ratio_
plt.scatter([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_)
plt.plot([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_)
plt.scatter([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_.cumsum())
plt.plot([i for i in range(N_PCA_COMP)], pca_est.explained_variance_ratio_.cumsum())
plt.figure(figsize=(12,5))
plt.plot([2, 4])

for i in range(4):
    ax = plt.subplot(2, 4, i + 1)
    ax.set_title("Comp # " + str(i+1))
    plt.axis('off')
    plt.imshow(pca_est.components_[i, :].reshape(28, 28), cmap= 'gray')

    ax2 = plt.subplot(2, 4, i + 4 + 1)
    ax2.set_title("Comp # " + str(i + 4 + 1))
    plt.axis('off')
    plt.imshow(pca_est.components_[i + 4, :].reshape(28, 28), cmap= 'gray')
#transorm the original images in the PCA space
X_val_pca = pca_est.transform(X_val)
#work with just a sample
N_samps = 1000
y_val_ = y_val.reset_index(drop=True)
sample = X_val_pca[:N_samps, :3]
plt.figure(figsize=(12,5))
plt.subplot(111)
for lab in np.unique([a for a in label_dict.keys()]):
    ix = np.where(y_val_[:N_samps] == lab)
    plt.scatter(sample[:, 0][ix], sample[:, 1][ix], label = label_dict[lab])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.xlabel('1st PCA component')
plt.ylabel('2nd PCA component')
plt.show()
import plotly.offline as py
import plotly.graph_objs as go
%matplotlib inline
py.init_notebook_mode(connected=True)

%matplotlib inline
trace1 = go.Scatter3d(
    x= sample[:, 0],
    y= sample[:, 1],
    z= sample[:, 2],
    text= [label_dict[k] for k in y_val_[:N_samps].values],
    mode= 'markers',
    marker= dict(
        color=y_val_[:N_samps], 
        opacity=0.8
    )
)

data = [trace1]

layout = go.Layout(
    scene = dict(
    xaxis = dict(
        title='1st PCA component'),
    yaxis = dict(
        title='2nd PCA component'),
    zaxis = dict(
        title='3rd PCA component'),)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.layers.normalization import BatchNormalization

#CONV-> BatchNorm-> RELU block
def conv_bn_relu_block(X, n_channels, kernel_size=(3, 3)):
    X = Conv2D(n_channels, kernel_size)(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    return X

#simple keras model
def fashion_cnn_model(input_shape):
    X_input = Input(input_shape)
    #zeropad
    X = ZeroPadding2D((1, 1))(X_input)
    #run a CONV -> BN -> RELU block
    X = conv_bn_relu_block(X, 32)
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.3)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 64)
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.4)(X)
    #run another CONV -> BN -> RELU block
    #X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 128)
    #dropout
    X = Dropout(0.3)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(len(label_dict.keys()), activation='softmax')(X)
    #output model
    model = Model(inputs = X_input, outputs = X, name='fashion_cnn_model')

    return model
fashionModel = fashion_cnn_model((28, 28, 1,))
#show the model architecture summary
fashionModel.summary()
fashionModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#reshape the input data
X_train_ = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
X_val_ = X_val.values.reshape(X_val.shape[0], 28, 28, 1)
from keras.utils import to_categorical
#fit the model
fashionModel.fit(x = X_train_ , y = to_categorical(y_train) ,epochs = 50, batch_size = 64)
#evaluate the model performance in the validation set
evs = fashionModel.evaluate(x = X_val_, y = to_categorical(y_val))
#show the accuracy metric
print(evs[1])
#label prediction
cnn_y_val_pred = fashionModel.predict(X_val_).argmax(axis=-1)
#plot confusion matrix
plot_conf_matrix(y_val, cnn_y_val_pred, 'Validation')
from sklearn.metrics import classification_report
print(classification_report(y_val, cnn_y_val_pred, target_names=[v for v in label_dict.values()]))
#run visual inspection for all classes
for i in label_dict.keys():
    visual_err_inspection(y_val, cnn_y_val_pred, i, 6)
#get the output of the last conv layer
#we will use it as description vector
intermediate_layer_model = Model(inputs = fashionModel.input,
                                 outputs = fashionModel.get_layer('conv2d_3').output)
#generate the output for all observations in validation set
intermediate_output = intermediate_layer_model.predict(X_val_)
#the cosine distance is applied to 1D vectors
intermediate_output_res = intermediate_output.reshape(15000, 128*5*5)
import scipy

def get_N_rec(idx_eval, N_recoms):
    eval_row = X_val_[idx_eval, :, :, :].reshape((1, 28, 28, 1))
    eval_activ =  intermediate_layer_model.predict(eval_row)
    #apply the cosine distance to all rows
    distance = np.apply_along_axis(scipy.spatial.distance.cosine, 1, intermediate_output_res, eval_activ.reshape(128*5*5))
    #get the N_recoms with the lowest distance
    #drop the first, as it is the row to be evaluated
    idx_recoms = distance.argsort()[1:N_recoms+1]
    #pass this idx to the idx space of the original datset
    out = [X_val.index[i] for i in idx_recoms]
    #also convert the original idx
    original = X_val.index[idx_eval]
    return out, original

#give me 6 recommendations for idx 50
idx_rec, orig = get_N_rec(50, 6)
import math
def plot_recommendations(idx_eval, N_recoms):
    idx_rec, orig = get_N_rec(idx_eval, N_recoms)
    fig = plt.figure(figsize=(10,10))
    N_cols_rec = math.ceil(1 + len(idx_rec) / 2)
    ax1 = fig.add_subplot(2, N_cols_rec,1)
    ax1.set_title('Original item')
    plot_image_pd(idx_to_pixseries(train_df, orig))
    for i in range(len(idx_rec)):
        ax_ = fig.add_subplot(2, N_cols_rec, i+2)
        ax_.set_title('Recomendation #' + str(i+1))
        plot_image_pd(idx_to_pixseries(train_df, idx_rec[i]))

#draw 6 recommendations for idx 50
plot_recommendations(50, 6)
#draw 6 recommendations for idx 4
plot_recommendations(4, 6)
#draw 6 recommendations for idx 4242
plot_recommendations(4242, 6)
#draw 6 recommendations for idx 101
plot_recommendations(101, 6)
