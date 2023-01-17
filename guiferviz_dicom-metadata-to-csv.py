import glob



import pandas as pd



import pydicom



import tqdm
# Paths

input_path = "../input/rsna-intracranial-hemorrhage-detection"

train_imgs = glob.glob(f"{input_path}/stage_1_train_images/*")
def save_value(img_data, name, value):

    if type(value) == pydicom.multival.MultiValue:

        for i, j in enumerate(value):

            save_value(img_data, f"{name}_{i}", j)

    else:

        if type(value) == pydicom.uid.UID:

            value = str(value)

        elif type(value) == pydicom.valuerep.DSfloat:

            value = float(value)

        img_data[name] = value



def get_data_dict(img):

    img_data = {}

    for i in img.iterall():

        if i.name == "Pixel Data":

            continue

        name = i.name.replace(" ", "_").replace("(", "").replace(")", "").lower()

        save_value(img_data, name, i.value)

    return img_data



def get_list_data(imgs):

    list_data = []

    for i in tqdm.tqdm(imgs):

        img = pydicom.read_file(i)

        img_data = get_data_dict(img)

        list_data.append(img_data)

    return list_data



def get_df_data(imgs):

    list_data = get_list_data(imgs)

    return pd.DataFrame(list_data)



df_imgs = get_df_data(train_imgs)

df_imgs.head()
df_imgs.to_csv("df_dicom_metadata.csv", index=False)