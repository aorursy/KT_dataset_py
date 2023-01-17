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
full_train_stack = imread(os.path.join(data_dir, 'train.tif'))
full_train_df.sample(5)
sns.pairplot(full_train_df, hue='opacity')
from sklearn.preprocessing import RobustScaler
def fit_and_score(in_model, full_features, full_labels, rescale=True):
    """
    Take a given model, set of features, and labels
    Break the dataset into training and validation
    Fit the model
    Show how well the model worked
    """
    train_feat, valid_feat, train_lab, valid_lab = train_test_split(full_features, 
                                                                    full_labels,
                                                                    test_size=0.25,
                                                                    random_state=2018)
    
    if rescale:
        feature_scaler = RobustScaler()
        train_feat = feature_scaler.fit_transform(train_feat)
        valid_feat = feature_scaler.transform(valid_feat)
    in_model.fit(train_feat, train_lab)
    predictions = in_model.predict_proba(valid_feat)[:, 1]
    predicted_class = predictions>0.5
    tpr, fpr, _ = roc_curve(valid_lab, predictions)
    auc = roc_auc_score(valid_lab, predictions)
    print(classification_report(valid_lab, predicted_class))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.plot(tpr, fpr, 'r.-', label='Prediction (AUC:{:2.2f})'.format(auc))
    ax1.plot(tpr, tpr, 'k-', label='Random Guessing')
    ax1.legend()
    ax1.set_title('ROC Curve')
    sns.heatmap(confusion_matrix(valid_lab, predicted_class), 
                annot=True, fmt='4d', ax=ax2)
    ax2.set_xlabel('Prediction')
    ax2.set_ylabel('Actual Value')
    ax2.set_title('Confusion Matrix ({:.1%})'.format(accuracy_score(valid_lab, predicted_class)))
# dummy random guesser
from sklearn.dummy import DummyClassifier
dum_model = DummyClassifier(strategy='stratified', random_state=2018)
fit_and_score(
    dum_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].values,
    full_train_df['opacity']
)
# nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(1) # one neighbor
fit_and_score(
    knn_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].values,
    full_train_df['opacity']
)
# logistic regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs', random_state=2018)
fit_and_score(
    lr_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP']].values,
    full_train_df['opacity']
)
full_train_df['Mean_Intensity'] = np.mean(full_train_stack, (1, 2))
full_train_df['Std_Intensity'] = np.std(full_train_stack, (1, 2))
knn_model = KNeighborsClassifier(2) # one neighbor
fit_and_score(
    knn_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP',
                   'Mean_Intensity', 'Std_Intensity']].values,
    full_train_df['opacity']
)
lr_model = LogisticRegression(solver='lbfgs', random_state=2018)
fit_and_score(
    lr_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']].values,
    full_train_df['opacity']
)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']].values,
    full_train_df['opacity']
)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
fit_and_score(
    nb_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']].values,
    full_train_df['opacity']
)
from sklearn.svm import SVC
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    full_train_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']].values,
    full_train_df['opacity']
)
from skimage.feature import greycomatrix, greycoprops
grayco_prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
def calc_texture_features(in_slice):
    glcm = greycomatrix(in_slice, [5], [0], 256, symmetric=True, normed=True)
    out_row = {}
    for c_prop in grayco_prop_list:
        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]
    return pd.Series(out_row)
# add the results to the current matrix
aug_df = pd.concat([
    full_train_df,
    full_train_df.apply(lambda x: calc_texture_features(full_train_stack[x['slice_idx']]), axis=1)
], 1)
aug_df.sample(3)
sns.pairplot(aug_df[['opacity', 'Mean_Intensity', 'Std_Intensity']+grayco_prop_list], hue='opacity')
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    aug_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']+grayco_prop_list].values,
    aug_df['opacity']
)
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    aug_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']+grayco_prop_list].values,
    aug_df['opacity']
)
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from keras import models, layers
color_image_stack = np.stack([full_train_stack, full_train_stack, full_train_stack], -1).astype(float)
pp_color_image_stack = preprocess_input(color_image_stack)
c_model = models.Sequential()
c_model.add(PTModel(include_top=False, 
                    input_shape=pp_color_image_stack.shape[1:], 
                    weights='imagenet'))
c_model.add(layers.GlobalAvgPool2D())
vgg_features = c_model.predict(pp_color_image_stack)
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    np.concatenate([aug_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']+grayco_prop_list].values,
                    vgg_features], 1),
    aug_df['opacity']
)
trn_image, _, trn_label , _ = train_test_split(full_train_stack, 
                                               full_train_df['opacity'],
                                               test_size=0.25,
                                               random_state=2018)
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
out_model.add(layers.Dense(16, activation='relu'))
out_model.add(layers.Dense(1, activation='sigmoid'))
out_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
out_model.summary()
from IPython.display import clear_output
fit_results = out_model.fit(trn_image, trn_label, epochs=100)
clear_output()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(fit_results.history['loss'])
ax1.set_title('Loss History')
ax2.plot(100*np.array(fit_results.history['binary_accuracy']))
ax2.set_title('Accuracy History')
rf_model = RandomForestClassifier(random_state=2018)
fit_and_score(
    rf_model,
    np.concatenate([aug_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']+grayco_prop_list].values,
                    out_model.predict(full_train_stack)], 1),
    aug_df['opacity']
)
svm_model = SVC(probability=True)
fit_and_score(
    svm_model,
    np.concatenate([aug_df[['PatientAge', 'opacity_prior', 'IsMale', 'IsAP', 
                   'Mean_Intensity', 'Std_Intensity']+grayco_prop_list].values,
                    out_model.predict(full_train_stack)], 1),
    aug_df['opacity']
)
