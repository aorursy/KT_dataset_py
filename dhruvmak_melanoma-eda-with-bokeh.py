# importing libraries

import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import missingno as msno

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style="whitegrid")



#bokeh

from bokeh.models import ColumnDataSource, HoverTool, Panel, FactorRange

from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.palettes import Spectral6



import warnings

warnings.filterwarnings('ignore')
# set up directory and files path



base_dir = "../input/siim-isic-melanoma-classification/"

train_csv = os.path.join(base_dir + "train.csv")

test_csv = os.path.join(base_dir + "test.csv")

jpeg_train_images = os.path.join(base_dir + "jpeg/train")

jpeg_test_images = os.path.join(base_dir + "jpeg/test")



train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)



train_df.head(5)
train_df["benign_malignant"].value_counts()
benign = train_df[train_df['benign_malignant']=='benign']

malignant = train_df[train_df['benign_malignant']=='malignant']
# Extract 9 random images from benign lesions

random_images = [np.random.choice((benign['image_name'].values)+'.jpg') for i in range(9)]



print('Display benign Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Extract 9 random images from malignant lesions

random_images = [np.random.choice((malignant['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# check for missing values

print(train_df.isnull().any())



msno.matrix(train_df, color=(207/255, 196/255, 171/255), fontsize=10)
# Number of missing values in sex column

print("Number of missing values in sex column is {}".format(train_df.shape[0] - train_df['sex'].count()))

print("--------------------------------------------------")

# Number of missing values in age_approx column

print("Number of missing values in age_approx column is {}".format(train_df.shape[0] - train_df['age_approx'].count()))

print("--------------------------------------------------")

# Number of missing values in anatom_site_general_challenge column

print("Number of missing values in anatom_site_general_challenge column is {}".format(train_df.shape[0] - train_df['anatom_site_general_challenge'].count()))
print(test_df.isnull().any())



msno.matrix(train_df, color=(207/255, 196/255, 171/255), fontsize=10)
# Number of missing values in anatom_site_general_challenge column

print("Number of missing values in anatom_site_general_challenge column is {}".format(test_df.shape[0] - test_df['anatom_site_general_challenge'].count()))
# Total number of training and testing images

print("Total images in Train set:", train_df["image_name"].count())

print("Total images in Test set:", test_df["image_name"].count())
# unique number of patients

print("Total patients ids are {}".format(train_df["patient_id"].count()))

print("Unique patients ids are {}".format(len(train_df["patient_id"].unique())))
# exploring the target column

train_df["target"].value_counts()
# This function will plot different type of histogram with Bokeh. It takes dataframe, column for which we want 

# histogram, color palate, bins for axes and title and return histogram



# For more information on how histograms work follow this blog

# https://towardsdatascience.com/interactive-histograms-with-bokeh-202b522265f3



def hist_hover(dataframe, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):

    hist, edges = np.histogram(dataframe[column], bins = bins)

    

    hist_df = pd.DataFrame({column: hist,

                            "left": edges[:-1],

                            "right": edges[1:]})

    hist_df["interval"] = ["%d to %d" % (left, right) for left,

                           right in zip(hist_df["left"], hist_df["right"])]

    

    src = ColumnDataSource(hist_df)

    plot = figure(plot_height = 400, plot_width = 600,

                  title = title,

                  x_axis_label = column,

                  y_axis_label = "Count")    

    plot.quad(bottom = 0, top = column,left = "left",

              right = "right", source = src, fill_color = colors[0],

              line_color = "#35838d", fill_alpha = 0.7,

              hover_fill_alpha = 0.7, hover_fill_color = colors[1])

    

    hover = HoverTool(tooltips = [('Interval', '@interval'), ('Count', str("@" + column))])

    plot.add_tools(hover)

    output_notebook()

    show(plot)
# histogram of Target column in training set

hist_hover(train_df, 'target', bins=3, title='Distribution of the Target column in the training set')
# Gender wise Distribution of target in traing set



Sex = ["Female", "Male"]

Target = ['0', '1']



g = train_df.groupby(["target", "sex"]).size()

male = list(g[0].values)

female = list(g[1].values)



data = {'Sex':Sex,

        'Male':male,

        'Female':female}



x = [(sex, target) for sex in Sex for target in Target]

counts = sum(zip(data['Male'], data['Female']), ())



source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))



p = figure(x_range=FactorRange(*x), plot_height=400, plot_width=800, title="Location of Image site with respect of sex",

           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@x: @counts"))



p.vbar(x='x', top='counts', width=0.9, color='color', source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
# location of image anatom site

train_df["anatom_site_general_challenge"].value_counts(sort=True)
# Distribution of anatom site general challenge column in training set



Categories = ["torso", "lower extremity", "upper extremity", "head/neck", "palms/soles", "oral/genital"]

counts = list(train_df["anatom_site_general_challenge"].value_counts(sort=True))



source = ColumnDataSource(data=dict(Categories=Categories, counts=counts, color=Spectral6))



p = figure(x_range=Categories, y_range=(0,22000), plot_height=300, title="Distribution of the anatom_site_general_challenge in the training set",

           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@Categories: @counts"))



p.vbar(x='Categories', top='counts', width=0.9, color='color', legend_field="Categories", source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
# Gender wise distribution of anatom site column in training set

print(train_df.groupby(["sex", "anatom_site_general_challenge"]).size())
# Gender wise distribution of anatom site column in training set

Categories = ["head/neck", "lower extremity", "oral/genital", "palms/soles", "torso", "upper extremity"]

Sex = ["Male", "Female"]



g = train_df.groupby(["sex", "anatom_site_general_challenge"]).size()

male = list(g.male.values)

female = list(g.female.values)



data = {'Categories':Categories,

        'Male':male,

        'Female':female}



x = [(categories, sex) for categories in Categories for sex in Sex]

counts = sum(zip(data['Male'], data['Female']), ())



source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))



p = figure(x_range=FactorRange(*x), plot_height=400, plot_width=800, title="Location of Image site with respect of sex",

           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@x: @counts"))



p.vbar(x='x', top='counts', width=0.9, color='color', source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
# Extract 9 random images from malignant lesions with growth in torso



torsomale = train_df[(train_df['benign_malignant']=='malignant') & (train_df['anatom_site_general_challenge'] == 'torso') & (train_df['sex'] == 'male')]



random_images = [np.random.choice((torsomale['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant torso Images with Male')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Extract 9 random images from malignant lesions with growth in lower extremity



lowerextremity = train_df[(train_df['benign_malignant']=='malignant') & (train_df['anatom_site_general_challenge'] == 'lower extremity') & (train_df['sex'] == 'female')]



random_images = [np.random.choice((lowerextremity['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant lower extremity Images with Female')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Distribution of age_approx column in training set

# we have missing values in age_approx so we need to fix that before plotting histograms



training_df = train_df

training_df['age_approx'].fillna(45.0, inplace=True) # 45 is mode of age_approx

training_df['age_approx'].isnull().any()

hist_hover(training_df, 'age_approx', title='Age Distribution of patients')
hist_hover(test_df, 'age_approx', title='Age Distribution of patients')
# Extract 9 random images from malignant lesions with age less than 21



youngerage = train_df[(train_df['benign_malignant']=='malignant') & (train_df['age_approx'] <= 21)]



random_images = [np.random.choice((youngerage['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant younger age Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Extract 9 random images from malignant lesions with age gap 45-48



middleage = train_df[(train_df['benign_malignant']=='malignant') & (train_df['age_approx'] >= 45) & (train_df['age_approx'] <= 48)]



random_images = [np.random.choice((middleage['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant middle age Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Extract 9 random images from malignant lesions with age greater than 78



oldage = train_df[(train_df['benign_malignant']=='malignant') & (train_df['age_approx'] >= 78)]



random_images = [np.random.choice((oldage['image_name'].values)+'.jpg') for i in range(9)]



print('Display malignant old age Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# distribution of diagnosis column in training set

train_df['diagnosis'].value_counts()
# Gender wise distribution of diagnosis column in training set



Categories = ["unknown", "nevus", "melanoma", "seborrheic keratosis", "lentigo NOS", "lichenoid keratosis", 

              "solar lentigo", "cafe-au-lait macule", "atypical melanocytic proliferation"]

counts = list(train_df["diagnosis"].value_counts())



source = ColumnDataSource(data=dict(Categories=Categories, counts=counts, color=Spectral6))



p = figure(x_range=Categories, y_range=(0,300), plot_width=800, plot_height=300, title="Distribution of the diagnosis in the training set",

           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@Categories: @counts"))



p.vbar(x='Categories', top='counts', width=0.9, color='color', legend_field="Categories", source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"

show(p)
# Extract 9 random images with unknown diagnosis



unknown = train_df[train_df['diagnosis'] == 'unknown']



random_images = [np.random.choice((unknown['image_name'].values)+'.jpg') for i in range(9)]



print('Display unknown diagnosis Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
# Extract 9 random images with melanoma diagnosis



melanoma = train_df[train_df['diagnosis'] == 'melanoma']



random_images = [np.random.choice((melanoma['image_name'].values)+'.jpg') for i in range(9)]



print('Display melanoma diagnosis Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(jpeg_train_images, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()
train_df['sex'].fillna("male", inplace = True)

train_df['age_approx'].fillna(50, inplace = True)

train_df['anatom_site_general_challenge'].fillna('torso', inplace = True)
categorical = ['sex', 'anatom_site_general_challenge', 'diagnosis']



label_encoder = LabelEncoder()



for column in categorical:

    train_df[column] = label_encoder.fit_transform(train_df[column])

    

# we do not need benign_malignant column as information is already present in target

train_df.drop(['benign_malignant'], axis = 1, inplace = True)
test_df['anatom_site_general_challenge'].fillna('torso', inplace = True)
categorical = ['sex', 'anatom_site_general_challenge']



label_encoder = LabelEncoder()



for column in categorical:

    test_df[column] = label_encoder.fit_transform(test_df[column])
images_shape = []



for k, image_name in enumerate(train_df['image_name']):

    image = Image.open(jpeg_train_images + "/" + image_name + '.jpg')

    images_shape.append(image.size)



images_shape_df = pd.DataFrame(data = images_shape, columns = ['H', 'W'], dtype='object')

images_shape_df['Size'] = '[' + images_shape_df['H'].astype(str) + ',' + images_shape_df['W'].astype(str) + ']'
images_shape_df.head()
print("We have {} types of different shapes in training images".format(len(list(images_shape_df['Size'].unique()))))
# Distribution of shapes in training set



# We have 88 types of unique shapes but many of them contain only few samples. so we will plot only 10 with 

# highest number of samples



Categories = list(images_shape_df['Size'].value_counts().keys())[0:10]

counts = list(images_shape_df['Size'].value_counts().values)[0:10]



source = ColumnDataSource(data=dict(Categories=Categories, counts=counts, color=Spectral6))



p = figure(x_range=Categories, y_range=(0,22000), plot_width = 1000, plot_height=300, title="Images shape in training set",

           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@Categories: @counts"))



p.vbar(x='Categories', top='counts', width=0.9, color='color', legend_field="Categories", source=source)



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)