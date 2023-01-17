import os
import warnings
warnings.simplefilter("ignore")

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
df_test  = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
df_train.sample(5)
df_test.sample(5)
unique_patient_train = df_train.patient_id.unique()
unique_patient_test = df_test.patient_id.unique()

print("Unique patients in training set:", unique_patient_train.shape[0])
print("Unique patients in test set:", unique_patient_test.shape[0])
print("No of patients common in train and test sets:", np.intersect1d(unique_patient_train, unique_patient_test).shape[0])
df_train.diagnosis.value_counts()
# Set the value of "unknown" as nan
df_train.diagnosis = df_train.diagnosis.apply(lambda x: np.nan if x == "unknown" else x)
df_train.diagnosis.value_counts()
df_train.benign_malignant.value_counts()
df_train.benign_malignant.isnull().sum()
df_train.target.value_counts()
ax = sns.countplot(x = "benign_malignant", hue = "target", data = df_train)
df_train.drop("benign_malignant", axis = 1, inplace = True)
df_train.isnull().sum()
# Let's drop the diagnosis as well as it has many null entry and is not present in the test set
df_train.drop("diagnosis", axis = 1, inplace = True)
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
df_train.isnull().sum().to_frame()
df_test.isnull().sum().to_frame()
(df_train.sex.value_counts() * 100 / df_train.shape[0]).to_frame()
(df_test.sex.value_counts() * 100 / df_test.shape[0]).to_frame()
# Since male and female ratio is almost same, with mens are more in number, let's fill those 65 missing items as male
df_train.sex = df_train.sex.fillna("male")
(df_train.sex.value_counts() * 100 / df_train.shape[0]).to_frame()
ax = sns.countplot(x = "sex", hue = "target", data = df_train)
# For age_approx, fill the na values with 0.0
df_train.age_approx = df_train.age_approx.fillna(0.0)
fig = plt.figure(figsize = (10, 5))
ax = sns.countplot(x = "age_approx", hue = "target", data = df_train)
(df_train.anatom_site_general_challenge.value_counts() * 100 / df_train.shape[0]).to_frame()
(df_test.anatom_site_general_challenge.value_counts() * 100 / df_test.shape[0]).to_frame()
# anatom_site_general_challenge needs to be imputed for both train and test. Let's use something called unknown for now
df_train.anatom_site_general_challenge = df_train.anatom_site_general_challenge.fillna("unknown")
df_test.anatom_site_general_challenge = df_test.anatom_site_general_challenge.fillna("unknown")
df_train.info()
# Let's do one hot encoding on sex column
df_train["sex_cat_m"] = 0
df_train.loc[df_train.sex == "male", "sex_cat_m"] = 1
df_train.drop("sex", axis = 1, inplace = True)
# Let's do one hot encoding on anatom_site_general_challenge column
# torso, lower extremity, upper extremity, head/neck, unknown, palms/soles, oral/genital

temp = pd.get_dummies(df_train.anatom_site_general_challenge, prefix = "location")
temp.drop("location_unknown", axis = 1, inplace = True)
temp.columns = [("location_cat_" + str(i)) for i in range(1, 7)]

df_train = pd.concat([df_train, temp], axis = 1)
df_train.drop("anatom_site_general_challenge", axis = 1, inplace = True)
temp = pd.get_dummies(df_test.anatom_site_general_challenge, prefix = "location")
temp.drop("location_unknown", axis = 1, inplace = True)
temp.columns = [("location_cat_" + str(i)) for i in range(1, 7)]

df_test = pd.concat([df_test, temp], axis = 1)
df_test.drop("anatom_site_general_challenge", axis = 1, inplace = True)
values = df_train.target.values
df_train.drop("target", axis = 1, inplace = True)
df_train["target"] = values
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
df_train.corr().style.background_gradient(cmap = "RdBu")
df_train.target.value_counts().to_frame()
(df_train.target.value_counts() * 100 / df_train.shape[0]).to_frame()
df_train
def plot_images(df, n_rows = 5, n_cols = 5, figsize = (20, 20), resize = (1024, 1024), preprocessing = None, label = 0):
    query_string = "target == {}".format(label)
    df = df.query(query_string).reset_index(drop = True)
    fig = plt.figure(figsize = figsize)
    ax  = []
    base_path = "../input/siim-isic-melanoma-classification/jpeg/train/"

    for i in range(n_rows * n_cols):
        img = plt.imread(base_path + df.loc[i, "image_name"] + ".jpg")
        img = cv2.resize(img, resize)

        if preprocessing:
            img = preprocessing(img)

        ax.append(fig.add_subplot(n_rows, n_cols, i + 1) )
        plot_title = "Image {}: {}".format(str(i + 1), "Benign" if label == 0 else "Malignant") 
        ax[-1].set_title(plot_title)
        plt.imshow(img, alpha = 1, cmap = "gray")

    plt.show()
# Training images - Benign
plot_images(df_train, label = 0, resize = (224, 224))
# Training images - Malignant
plot_images(df_train, label = 1, resize = (224, 224))