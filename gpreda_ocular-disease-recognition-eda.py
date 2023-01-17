import numpy as np

import pandas as pd

import os

import glob

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS
for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)
print(os.listdir("/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K"))
data_df = pd.read_excel(open("/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx", 'rb'), sheet_name='Sheet1')  
data_df.head()
data_df.columns = ["id", 'age', "sex", "left_fundus", "right_fundus", "left_diagnosys", "right_diagnosys", "normal",

                  "diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"]
data_df.head()
print(f"data shape: {data_df.shape}")

print(f"left fundus: {data_df.left_fundus.nunique()}")

print(f"right fundus: {data_df.right_fundus.nunique()}")
print(f"train images: {len(os.listdir('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Training Images'))}")

print(f"test images: {len(os.listdir('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Testing Images'))}")

print(f"train images - left eye: {len(glob.glob('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Training Images//*_left.jpg'))}")

print(f"train images - right eye: {len(glob.glob('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Training Images//*_right.jpg'))}")

print(f"test images - left eye: {len(glob.glob('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Testing Images//*_left.jpg'))}")

print(f"test images - right eye: {len(glob.glob('//kaggle//input//ocular-disease-recognition-odir5k//ODIR-5K//Testing Images//*_right.jpg'))}")
def plot_count(feature, title, df, size=1, show_all=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    if show_all:

        g = sns.countplot(df[feature], palette='Set3')

        g.set_title("{} distribution".format(title))

    else:

        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

        if(size > 2):

            plt.xticks(rotation=90, size=8)

            for p in ax.patches:

                height = p.get_height()

                ax.text(p.get_x()+p.get_width()/2.,

                        height + 0.2,

                        '{:1.2f}%'.format(100*height/total),

                        ha="center") 

        g.set_title("Number and percentage of {}".format(title))

    plt.show()    
plot_count("age", "Age", data_df, size=5, show_all=True)
plot_count("sex", "Sex", data_df, size=2)
plot_count("normal", "Normal", data_df, size=2)
plot_count("diabetes", "Diabetes", data_df, size=2)
plot_count("glaucoma", "Glaucoma", data_df, size=2)
plot_count("cataract", "Cataract", data_df, size=2)
plot_count("amd", "AMD", data_df, size=2)
plot_count("hypertension", "Hypertension", data_df, size=2)
plot_count("myopia", "Myopia", data_df, size=2)
plot_count("other", "Other", data_df, size=2)
plot_count("left_diagnosys", "Left eye diagnosys (first 20 values)", data_df, size=4)
plot_count("right_diagnosys", "Right eye diagnosys (first 20 values)", data_df, size=4)
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=40,

        max_font_size=40, 

        scale=3,

        random_state=1,

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(data_df['left_diagnosys'], title = 'Prevalent words in left eye diagnosys')
show_wordcloud(data_df['left_diagnosys'], title = 'Prevalent words in right eye diagnosys')
cataract_df = data_df.loc[data_df.cataract == 1]

show_wordcloud(cataract_df['left_diagnosys'], title = 'Prevalent words in left eye diagnosys for cataract')
show_wordcloud(cataract_df['right_diagnosys'], title = 'Prevalent words in right eye diagnosys for cataract')
def plot_feature_distribution_grouped(feature, title, df, hue, size=4):

    plt.figure(figsize=(size*5,size*2))

    plt.title(title)

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    g = sns.countplot(df[feature], hue=df[hue], palette='Set3')

    plt.xlabel(feature)

    plt.legend()

    plt.show()
plot_feature_distribution_grouped('sex', 'Normal diagnosys grouped by sex', data_df, 'normal', size=2)
plot_feature_distribution_grouped('sex', 'Diabetes diagnosys grouped by sex', data_df, 'diabetes', size=2)
plot_feature_distribution_grouped('sex', 'Glaucoma diagnosys grouped by sex', data_df, 'glaucoma', size=2)
plot_feature_distribution_grouped('sex', 'Cataract diagnosys grouped by sex', data_df, 'cataract', size=2)
plot_feature_distribution_grouped('sex', 'AMD diagnosys grouped by sex', data_df, 'amd', size=2)
plot_feature_distribution_grouped('sex', 'Hypertension diagnosys grouped by sex', data_df, 'hypertension', size=2)
plot_feature_distribution_grouped('sex', 'Myopia diagnosys grouped by sex', data_df, 'myopia', size=2)
plot_feature_distribution_grouped('sex', 'Other diseases diagnosys grouped by sex', data_df, 'other', size=2)
import imageio

IMAGE_PATH = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images"

def show_images(df, title="Diagnosys", eye_exam="left_fundus"):

    print(f"{title}; eye exam: {eye_exam}")

    f, ax = plt.subplots(3,3, figsize=(16,16))

    for i,idx in enumerate(df.index):

        dd = df.iloc[idx]

        image_name = dd[eye_exam]

        image_path = os.path.join(IMAGE_PATH, image_name)

        img_data=imageio.imread(image_path)

        ax[i//3, i%3].imshow(img_data)

        ax[i//3, i%3].axis('off')

    plt.show()
df = data_df.loc[(data_df.cataract==1) & (data_df.left_diagnosys=="cataract")].sample(9).reset_index()

show_images(df,title="Left eye with cataract",eye_exam="left_fundus")
df = data_df.loc[(data_df.cataract==1) & (data_df.right_diagnosys=="cataract")].sample(9).reset_index()

show_images(df,title="Right eye with cataract",eye_exam="right_fundus")
df = data_df.loc[(data_df.glaucoma==1) & (data_df.left_diagnosys=="glaucoma")].sample(9).reset_index()

show_images(df,title="Left eye with glaucoma",eye_exam="left_fundus")
df = data_df.loc[(data_df.glaucoma==1) & (data_df.right_diagnosys=="glaucoma")].sample(9).reset_index()

show_images(df,title="Right eye with glaucoma",eye_exam="right_fundus")
df = data_df.loc[(data_df.myopia==1)].sample(9).reset_index()

show_images(df,title="Left eye with myopia",eye_exam="left_fundus")
df = data_df.loc[(data_df.myopia==1)].sample(9).reset_index()

show_images(df,title="Right eye with myopia",eye_exam="right_fundus")