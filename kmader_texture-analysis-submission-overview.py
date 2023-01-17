%matplotlib inline
import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns # nice visuals
from sklearn.model_selection import train_test_split # splitting data
# quantifying models
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
data_dir = '../input/'
def categories_to_indicators(in_df):
    new_df = in_df.copy()
    new_df['IsMale'] = in_df['PatientSex'].map(lambda x: 'M' in x).astype(float)
    new_df['IsAP'] = in_df['ViewPosition'].map(lambda x: 'AP' in x).astype(float)
    return new_df.drop(['PatientSex', 'ViewPosition'], axis=1)
full_train_df = categories_to_indicators(pd.read_csv(os.path.join(data_dir, 'train_all.csv')))
full_stack = imread(os.path.join(data_dir, 'train.tif')) # read all slices
full_train_df['image'] = full_train_df['slice_idx'].map(lambda x: full_stack[x]) # get the slice
full_train_df.sample(3)
sns.pairplot(full_train_df, hue='opacity')
submission_test_df = categories_to_indicators(pd.read_csv(os.path.join(data_dir, 'test_info.csv')))
test_stack = imread(os.path.join(data_dir, 'test.tif')) # read all slices
submission_test_df['image'] = submission_test_df['slice_idx'].map(lambda x: full_stack[x]) # get the slice
submission_test_df.sample(3)
from sklearn.preprocessing import RobustScaler
def fit_and_score(in_model, feature_maker, rescale=True):
    """
    Take a given model, set of features, and labels
    Break the dataset into training and validation
    Fit the model
    Show how well the model worked
    Input
        in_model: The model to fit (in scikit-learn model format)
        feature_maker: the function to run on the training data to turn it into a feature vector
        rescale: whether or not to rescale all features first
    """
    # compute features on the training data
    full_features = feature_maker(full_train_df)
    full_labels = full_train_df['opacity']
    # compute features on the test data
    submission_feat = feature_maker(submission_test_df)
    # split the training data into a training portion and a validation portion
    # so we can see how well the model works
    train_feat, valid_feat, train_lab, valid_lab = train_test_split(full_features, 
                                                                    full_labels,
                                                                    test_size=0.25,
                                                                    random_state=2018)
    
    if rescale:
        feature_scaler = RobustScaler()
        train_feat = feature_scaler.fit_transform(train_feat)
        valid_feat = feature_scaler.transform(valid_feat)
        submission_feat = feature_scaler.transform(submission_feat)
    # fit the model
    in_model.fit(train_feat, train_lab)
    # predict on the validation feature set
    predictions = in_model.predict_proba(valid_feat)[:, 1]
    # convert predictions to a class (opacity = 1, no-opacity = 0)
    predicted_class = predictions>0.5
    # make an ROC curve based on the known label for the validations et
    tpr, fpr, _ = roc_curve(valid_lab, predictions)
    # compute the AUC score
    auc = roc_auc_score(valid_lab, predictions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.plot(tpr, fpr, 'r.-', label='Prediction (AUC:{:2.2f})'.format(auc))
    ax1.plot(tpr, tpr, 'k-', label='Random Guessing')
    ax1.legend()
    ax1.set_title('ROC Curve')
    # make a full classification report
    print(classification_report(valid_lab, predicted_class, target_names=['opacity', 'no opacity']))
    # create a confusion matrix
    sns.heatmap(confusion_matrix(valid_lab, predicted_class), 
                annot=True, fmt='4d', ax=ax2)
    ax2.set_xlabel('Prediction')
    ax2.set_ylabel('Actual Value')
    ax2.set_title('Confusion Matrix ({:.1%})'.format(accuracy_score(valid_lab, predicted_class)))
    
    # create a submission file as a csv
    sub_df = submission_test_df[['tile_id']].copy()
    sub_df['opacity'] = in_model.predict_proba(submission_feat)[:, 1]
    
    sub_df[['tile_id', 'opacity']].to_csv(
        'm-{model}-f-{features}-s-{auc:2.0f}.csv'.format(
            model=in_model.__class__.__name__, # model name
            features=feature_maker.__name__, # feature name
            auc=100*auc, # add the auc score to the name
        ), index=False)
# dummy random guesser
from sklearn.dummy import DummyClassifier
dum_model = DummyClassifier(strategy='stratified', random_state=2018)
def justage(in_df):
    return in_df[['PatientAge']].values
fit_and_score(
    dum_model,
    justage
)
# nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(1) # one neighbor
def table_features(in_df):
    return in_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].values
fit_and_score(
    knn_model,
    table_features
)
def basic_image_features(in_df):
    out_df = in_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].copy()
    out_df['Mean_Intensity'] = in_df['image'].map(np.mean)
    out_df['Std_Intensity'] = in_df['image'].map(np.std)
    return out_df.values
knn_model = KNeighborsClassifier(2) # two neighbor
fit_and_score(
    knn_model,
    basic_image_features
)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs', random_state=2018)
fit_and_score(
    lr_model,
    basic_image_features
)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    basic_image_features
)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
fit_and_score(
    nb_model,
    basic_image_features
)
from sklearn.svm import SVC
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    basic_image_features
)
from skimage.feature import greycomatrix, greycoprops
grayco_prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
def calc_texture_features(in_slice):
    glcm = greycomatrix(in_slice, [5], [0], 256, symmetric=True, normed=True)
    out_row = {}
    for c_prop in grayco_prop_list:
        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]
    return pd.Series(out_row)
def texture_features(in_df):
    out_df = in_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].copy()
    out_df['Mean_Intensity'] = in_df['image'].map(np.mean)
    out_df['Std_Intensity'] = in_df['image'].map(np.std)
    # add the results to the current matrix
    aug_df = pd.concat([
        out_df,
        in_df.apply(lambda x: calc_texture_features(x['image']), axis=1)
    ], 1)
    return aug_df.values
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    texture_features
)
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from keras import models, layers
c_model = models.Sequential()
c_model.add(PTModel(include_top=False, 
                    # get the shape of the image
                    input_shape=full_train_df['image'].iloc[0].shape+(3,), 
                    weights='imagenet'))
c_model.add(layers.GlobalAvgPool2D())

def vgg_features(in_df):
    out_df = in_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].copy()
    full_image_stack = np.stack(in_df['image'], 0)
    color_image_stack = np.stack([full_image_stack, full_image_stack, full_image_stack], -1).astype(float)
    pp_color_image_stack = preprocess_input(color_image_stack)
    # add the results to the current matrix
    vgg_features = c_model.predict(pp_color_image_stack)
    return np.concatenate([out_df.values, vgg_features], 1)
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    vgg_features
)
from xgboost import XGBClassifier
xg_model = XGBClassifier()
fit_and_score(
    xg_model,
    vgg_features
)
trn_image, vld_image, trn_label , vld_label = train_test_split(full_train_df['image'], 
                                               full_train_df['opacity'],
                                               test_size=0.25,
                                               random_state=2018)
trn_image = np.stack(trn_image, 0)
vld_image = np.stack(vld_image, 0)
out_model = models.Sequential()
out_model.add(layers.Reshape((64, 64, 1), input_shape=trn_image.shape[1:]))
out_model.add(layers.Conv2D(16, (3, 3), padding='valid', activation='relu'))
out_model.add(layers.MaxPool2D((2, 2)))
out_model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
out_model.add(layers.MaxPool2D((2, 2)))
out_model.add(layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))
out_model.add(layers.MaxPool2D((2, 2)))
out_model.add(layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))
out_model.add(layers.MaxPool2D((2, 2)))
out_model.add(layers.GlobalAveragePooling2D())
out_model.add(layers.Dense(32, activation='relu'))
out_model.add(layers.Dense(1, activation='sigmoid'))
out_model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy'])
out_model.summary()
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image
model_rep = model_to_dot(out_model, show_shapes=True)
model_rep.set_rankdir('LR')
Image(model_rep.create_png())
from IPython.display import clear_output
fit_results = out_model.fit(trn_image, trn_label, 
                            validation_data=(vld_image, vld_label),
                            epochs=100)
clear_output()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(fit_results.history['loss'], label='Training')
ax1.plot(fit_results.history['val_loss'], label='Validation')
ax1.legend()
ax1.set_title('Loss History')
ax2.plot(100*np.array(fit_results.history['binary_accuracy']), label='Training')
ax2.plot(100*np.array(fit_results.history['val_binary_accuracy']), label='Validation')
ax2.legend()
ax2.set_title('Accuracy History')
def custom_cnn_features(in_df):
    out_df = in_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].copy()
    full_image_stack = np.stack(in_df['image'], 0)
    # add the results to the current matrix
    model_pred = out_model.predict(full_image_stack)
    return np.concatenate([out_df.values, model_pred], 1)
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    custom_cnn_features
)
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    custom_cnn_features
)
xg_model = XGBClassifier()
fit_and_score(
    xg_model,
    custom_cnn_features
)
!ls *.csv
