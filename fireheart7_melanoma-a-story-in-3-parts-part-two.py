import os

for dirname, _, filenames in os.walk('/kaggle/input'):

        print(dirname)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go

import cv2

from tqdm import tqdm_notebook as tqdm

%matplotlib inline
train = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv"))

test = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv"))
train.shape, test.shape
train.head()
test.head()
train.info()
test.info()
len(train["patient_id"].unique()), len(test["patient_id"].unique())
print(train["target"].value_counts())
malignant = len(train[train["target"] == 1])

benign = len(train[train["target"] == 0])



labels = ["Malignant", "Benign"] 

size = [malignant, benign]



plt.figure(figsize = (8, 8))

plt.pie(size, labels = labels, shadow = True, startangle = 90, colors = ["r", "g"])

plt.title("Malignant VS Benign Cases")

plt.legend()
train_males = len(train[train["sex"] == "male"])

train_females  = len(train[train["sex"] == "female"])



test_males = len(test[test["sex"] == "male"])

test_females  = len(test[test["sex"] == "female"])



labels = ["Males", "Female"] 



size = [train_males, train_females]

explode = [0.1, 0.0]



plt.figure(figsize = (16, 16))

plt.subplot(1,2,1)

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["b", "g"])

plt.title("Male VS Female Training Set Count", fontsize = 18)

plt.legend()



print("Number of males in training set = ", train_males)

print("Number of females in training set= ", train_females)



size = [test_males, test_females]



plt.subplot(1,2,2)

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["b", "g"])

plt.title("Male VS Female Test Set Count", fontsize = 18)

plt.legend()



print("Number of males in testing set = ", test_males)

print("Number of females in testing set= ", test_females)
train_malignant  = train[train["target"] == 1]

train_malignant_males = len(train_malignant[train_malignant["sex"] == "male"])

train_malignant_females  = len(train_malignant[train_malignant["sex"] == "female"])



labels = ["Malignant Male Cases", "Malignant Female Cases"] 

size = [train_malignant_males, train_malignant_females]

explode = [0.1, 0.0]



plt.figure(figsize = (10, 10))

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["r", "c"])

plt.title("Malignant Male VS Female Cases", fontsize = 18)

plt.legend()

print("Malignant Male Cases = ", train_malignant_males)

print("Malignant Female Cases = ", train_malignant_females)
train_benign  = train[train["target"] == 0]



train_benign_males = len(train_benign[train_benign["sex"] == "male"])

train_benign_females  = len(train_benign[train_benign["sex"] == "female"]) 



labels = ["Benign Male Cases", "Benign Female Cases"] 

size = [train_benign_males, train_benign_females]

explode = [0.1, 0.0]



plt.figure(figsize = (10, 10))

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["g", "y"])

plt.title("Benign Male VS Benign Female Cases", fontsize = 18)

plt.legend()

print("Benign Male Cases = ", train_benign_males)

print("Benign Female Cases = ", train_benign_females)
cancer_versus_sex = train.groupby(["benign_malignant", "sex"]).size()

print(cancer_versus_sex)

print(type(cancer_versus_sex))
cancer_versus_sex = cancer_versus_sex.unstack(level = 1) / len(train) * 100

print(cancer_versus_sex)

print(type(cancer_versus_sex))
sns.set(style='whitegrid')

sns.set_context("paper", rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":18})   



plt.figure(figsize = (10, 6))

sns.heatmap(cancer_versus_sex, annot=True, cmap="icefire", cbar=True)

plt.title("Cancer VS Sex Heatmap Analysis Normalized", fontsize = 18)

plt.tight_layout()
sns.set(style='whitegrid')

sns.set_context("paper", rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":18})   



plt.figure(figsize = (10, 6))

sns.boxplot(train["benign_malignant"], train["age_approx"], palette="icefire")

plt.title("Age VS Cancer Boxplot Analysis")

plt.tight_layout()
print("################### Training set info ###################")

print(train["anatom_site_general_challenge"].unique())

print(train["anatom_site_general_challenge"].value_counts())



print("\n\n")



print("################### Test set info ###################")

print(test["anatom_site_general_challenge"].unique())

print(test["anatom_site_general_challenge"].value_counts())
# train

train_torso = len(train[train["anatom_site_general_challenge"] == "torso"])

train_lower_extremity = len(train[train["anatom_site_general_challenge"] == "lower extremity"])

train_upper_extremity = len(train[train["anatom_site_general_challenge"] == "upper extremity"])

train_head_neck = len(train[train["anatom_site_general_challenge"] == "head/neck"])

train_palms_soles = len(train[train["anatom_site_general_challenge"] == "palms/soles"])

train_oral_genital = len(train[train["anatom_site_general_challenge"] == "oral/genital"])



# test

test_torso = len(test[test["anatom_site_general_challenge"] == "torso"])

test_lower_extremity = len(test[test["anatom_site_general_challenge"] == "lower extremity"])

test_upper_extremity = len(test[test["anatom_site_general_challenge"] == "upper extremity"])

test_head_neck = len(test[test["anatom_site_general_challenge"] == "head/neck"])

test_palms_soles = len(test[test["anatom_site_general_challenge"] == "palms/soles"])

test_oral_genital = len(test[test["anatom_site_general_challenge"] == "oral/genital"])



################# DISPLAY #################



labels = ["Torso", "Lower Extremity", "Upper Extremity", "Head/Neck", "Palms/Soles", "Oral/Genital"] 



plt.figure(figsize = (16, 16))



plt.subplot(1,2,1)

size = [train_torso, train_lower_extremity, train_upper_extremity, train_head_neck, train_palms_soles, train_oral_genital]

explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1]

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90)

plt.title("Anatomy Sites In Training Set", fontsize = 18)

plt.legend()



plt.subplot(1,2,2)

size = [test_torso, test_lower_extremity, test_upper_extremity, test_head_neck, test_palms_soles, test_oral_genital]

explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1]

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90)

plt.title("Anatomy Sites In Testing Set", fontsize = 18)

plt.legend()



# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
train_ages_benign = train.loc[train["target"] == 0, "age_approx"]

train_ages_malignant = train.loc[train["target"] == 1 , "age_approx"]



plt.figure(figsize = (10, 8))

sns.kdeplot(train_ages_benign, label = "Benign", shade = True, legend = True, cbar = True)

sns.kdeplot(train_ages_malignant, label = "Malignant", shade = True, legend = True, cbar = True)

plt.grid(True)

plt.xlabel("Age Of The Patients", fontsize = 18)

plt.ylabel("Probability Density", fontsize = 18)

plt.grid(which = "minor", axis = "both")

plt.title("Probabilistic Age Distribution In Training Set", fontsize = 18)
train_image_stats_01 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_01"))

train_image_stats_02 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_02"))

train_image_stats_03 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_03"))

train_image_stats_04 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_04"))

train_image_stats_05 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_05"))

train_image_stats_06 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_06"))



print(train_image_stats_01.shape)

print(train_image_stats_02.shape)

print(train_image_stats_03.shape)

print(train_image_stats_04.shape)

print(train_image_stats_05.shape)

print(train_image_stats_06.shape)
train_image_statistics = pd.concat([train_image_stats_01, train_image_stats_02, train_image_stats_03,

                                   train_image_stats_04, train_image_stats_05, train_image_stats_06],

                                  ignore_index = True)

train_image_statistics.shape
train_image_statistics.info()
test_image_stats_01 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_test_01"))

test_image_stats_02 = pd.DataFrame(pd.read_csv("../input/melanoma-image-insights/melanoma_image_statistics_compiled_test_02"))



print(test_image_stats_01.shape)

print(test_image_stats_02.shape)
test_image_statistics = pd.concat([test_image_stats_01, test_image_stats_02], ignore_index = True)



test_image_statistics.shape
test_image_statistics.info()
train_image_statistics.head()
test_image_statistics.head()
image_names = train_image_statistics["image_name"].values

random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array

random_images 
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
plt.figure(figsize = (12, 8))

for i in range(4) : 

    plt.subplot(2, 2, i + 1) 

    image = cv2.imread(os.path.join(train_dir, random_images[i]))

    # cv2 reads images in BGR format. Hence we convert it to RGB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image, cmap = "gray")

    plt.grid(True)

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
benign_mean_red_value = []

benign_mean_green_value = []

benign_mean_blue_value = []



malignant_mean_red_value = []

malignant_mean_green_value = []

malignant_mean_blue_value = []



for image_name in tqdm(train_image_statistics["image_name"]) : 

    name = image_name[0:len(image_name)-4] # as .jpg are the appended at the end of the name

    extracted_section = train[train["image_name"] == name]

    r = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_red_value"])

    g = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_green_value"])

    b = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_blue_value"])

    if int(extracted_section["target"]) == 0 : # benign

        benign_mean_red_value.append(r)

        benign_mean_green_value.append(g)

        benign_mean_blue_value.append(b)

    else:

        malignant_mean_red_value.append(r)

        malignant_mean_green_value.append(g)

        malignant_mean_blue_value.append(b)
#red channel plot

range_of_spread = max(benign_mean_red_value) - min(benign_mean_red_value)



plt.figure(figsize = (12, 8))

plt.rc("font", weight = "bold")

sns.set_style("whitegrid")

fig = sns.distplot(benign_mean_red_value, hist = True, kde = True, label = "Mean Red Channel Intensities", color = "r")

fig.set(xlabel = "Mean red channel intensities observed in each image",

        ylabel = "Probability Density")

plt.title("Spread Of Red Channel In Benign Cases", fontsize = 18)

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
#green channel plot

range_of_spread = max(benign_mean_green_value) - min(benign_mean_green_value)



plt.figure(figsize = (12, 8))

plt.rc("font", weight = "bold")

sns.set_style("whitegrid")

fig = sns.distplot(benign_mean_green_value, hist = True, kde = True, label = "Mean Green Channel Intensities", color = "g")

fig.set(xlabel = "Mean green channel intensities observed in each image",

        ylabel = "Probability Density") 

plt.title("Spread Of Green Channel In Benign Cases", fontsize = 18)

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
#Blue channel plot

range_of_spread = max(benign_mean_blue_value) - min(benign_mean_blue_value)



plt.figure(figsize = (12, 8))

plt.rc("font", weight = "bold")

sns.set_style("whitegrid")

fig = sns.distplot(benign_mean_blue_value, hist = True, kde = True, label = "Mean Blue Channel Intensities", color = "b")

fig.set(xlabel = "Mean blue channel intensities observed in each image",

        ylabel = "Probability Density") 

plt.title("Spread Of Blue Channel In Benign Cases", fontsize = 18)

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
plt.figure(figsize = (12, 8))

plt.rc("font", weight = "bold")

sns.set_style("whitegrid")

fig = sns.distplot(benign_mean_blue_value, hist = False, kde = True, label = "Mean Blue Channel Intensities", color = "b")

fig = sns.distplot(benign_mean_red_value, hist = False, kde = True, label = "Mean Red Channel Intensities", color = "r")

fig = sns.distplot(benign_mean_green_value, hist = False, kde = True, label = "Mean Green Channel Intensities", color = "g")



fig.set(xlabel = "Mean channel intensities observed in each image",

        ylabel = "Probability Density") 

plt.title("Spread Of Channels In Benign Cases", fontsize = 18)

plt.legend()
# free up the memory

del benign_mean_red_value

del benign_mean_green_value

del benign_mean_blue_value
import gc

gc.collect()
plt.figure(figsize = (12, 8))

plt.rc("font", weight = "bold")

sns.set_style("whitegrid")

fig = sns.distplot(malignant_mean_blue_value, hist = False, kde = True, label = "Mean Blue Channel Intensities", color = "b")

fig = sns.distplot(malignant_mean_red_value, hist = False, kde = True, label = "Mean Red Channel Intensities", color = "r")

fig = sns.distplot(malignant_mean_green_value, hist = False, kde = True, label = "Mean Green Channel Intensities", color = "g")



fig.set(xlabel = "Mean channel intensities observed in each image",

        ylabel = "Probability Density") 

plt.title("Spread Of Channels In Malignant Cases", fontsize = 18)

plt.legend()
gc.collect() # free up the memory
train.head()
# visualizing missing values in "sex" column



missing = len(train[train["sex"].isna() == True])

available = len(train[train["sex"].isna() == False])



x = ["Availabe data", "Unavailable data"]

y = [np.log(available), np.log(missing)] # plotting log data as the extreme values will supressed and lower ones will shoot, making it eay to visualize



print("Count of missing data = ", missing)

print("Count of available data = ", available)



plt.figure(figsize = (12, 8))

plt.subplot(1,1,1)

plt.barh(x, y, color = "m")

plt.grid(True)

plt.title("Data On Patient's Sex")
train["sex"].fillna("male", inplace = True)
missing =  len(train[train["age_approx"].isna() == True]) 

available = len(train[train["age_approx"].isna() == False]) 



print("Missing age values = ", missing)

print("Available age data = ", available)



x = ["Availabe data", "Unavailable data"]

y = [np.log(available), np.log(missing)] # plotting log data as the extreme values will supressed and lower ones will shoot, making it eay to visualize



plt.figure(figsize = (12, 8))

plt.subplot(1,1,1)

plt.barh(x, y, color = "y")

plt.grid(True)

plt.title("Data On Patient's Age")
# train

anatomy_sites = ["torso", "upper extremity", "lower extremity"]



# first select the relevant part of the full dataframe satisfying either of the aforementioned three conditions.

relevant_dataframe_part = train[(train["sex"] == "male") &

                     (train["anatom_site_general_challenge"].isin(anatomy_sites)) &

                     (train["target"] == 0)]



# Now, we have the data frame. To calculate median, we need to specify the column along which we intend to calculate the median.

median_value = relevant_dataframe_part["age_approx"].median()



print("Median value = ", median_value)
train["age_approx"].fillna(median_value, inplace = True)
train["anatom_site_general_challenge"].fillna("torso", inplace = True)

test["anatom_site_general_challenge"].fillna("torso", inplace = True)
train.info()
test.info()
train.to_csv("updated_training_file", index = False)

test.to_csv("updated_test_file", index = False)
plt.figure(figsize = (15, 8))



plt.subplot(1,2,1)

x = train_image_statistics["rows"]

y = train_image_statistics["columns"]

plt.scatter(x, y, cmap = "magma")

plt.title("Shape Analysis Of Training Images", fontsize = 18)

plt.xlabel("Number Of Rows", fontsize = 18)

plt.ylabel("Number Of Columns", fontsize = 18)



plt.subplot(1,2,2)

x = test_image_statistics["rows"]

y = test_image_statistics["columns"]

plt.scatter(x, y, cmap = "magma")

plt.title("Shape Analysis Of Testing Images", fontsize = 18)

plt.xlabel("Number Of Rows", fontsize = 18)

plt.ylabel("Number Of Columns", fontsize = 18)





plt.tight_layout()
plt.figure(figsize = (12, 8))



x = train_image_statistics["rows"]

y = train_image_statistics["columns"]

plt.scatter(x, y, cmap = "plasma", label = "Training Image")

plt.title("Shape Analysis", fontsize = 18)



x = test_image_statistics["rows"]

y = test_image_statistics["columns"]

plt.scatter(x, y, cmap = "magma", label = "Testing Image")



plt.xlabel("Number Of Rows", fontsize = 18)

plt.ylabel("Number Of Columns", fontsize = 18)

plt.legend()
train_image_statistics.head()
fig = go.Figure(data = [go.Scatter3d(x = train_image_statistics["image_mean"],

                                    y = train_image_statistics["image_standard_deviation"], 

                                   z = train_image_statistics["image_skewness"],

                                    mode = "markers",

                                    marker = dict(size = 4, color = train_image_statistics["rows"],



                                                  colorscale = "jet", opacity = 0.4))] , 

               

                layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),

                                   scene = dict(xaxis = dict(title='Image Mean'),

                                                yaxis = dict(title='Image Standard Deviation'),

                                                zaxis = dict(title='Image Skewness'),),))

fig.show()
fig = go.Figure(data = [go.Scatter3d(x = test_image_statistics["image_mean"],

                                    y = test_image_statistics["image_standard_deviation"], 

                                   z = test_image_statistics["image_skewness"],

                                    mode = "markers",

                                    marker = dict(size = 4, color = test_image_statistics["rows"],



                                                  colorscale = "jet", opacity = 0.4))] , 

               

                layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),

                                   scene = dict(xaxis = dict(title='Image Mean'),

                                                yaxis = dict(title='Image Standard Deviation'),

                                                zaxis = dict(title='Image Skewness'),),))



fig.show()