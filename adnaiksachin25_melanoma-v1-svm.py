#plotly

!pip install --upgrade pip --quiet

!pip install chart_studio --quiet



import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style="whitegrid")



#pydicom

import pydicom

# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()
os.listdir('../input/siim-isic-melanoma-classification/')
#os.listdir('../input/siim-isic-melanoma-classification/jpeg/train/')
BASE_PATH = '../input/siim-isic-melanoma-classification'







print('Reading data...')

train = pd.read_csv(f'{BASE_PATH}/train.csv')

test = pd.read_csv(f'{BASE_PATH}/test.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print('Reading data completed')
display(train.head())

print("Shape of train :", train.shape)
display(test.head())

print("Shape of test :", test.shape)
# checking missing data

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
# checking missing data

total = test.isnull().sum().sort_values(ascending = False)

percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)

missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test_data.head()
def plot_count(df, feature, title='', size=2.5):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(train, 'benign_malignant')
plot_count(train, 'sex')


plot_count(train, 'anatom_site_general_challenge')



train['diagnosis'].value_counts(normalize=True).sort_values().iplot(kind='barh',

                                                      xTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='blue',

                                                      theme='pearl',

                                                      bargap=0.2,

                                                      gridcolor='white',

                                                      title='Distribution in the training set'

                                                    )
def plot_relative_distribution(df, feature, hue, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_relative_distribution(

    df=train,

    feature='sex',

    hue='benign_malignant',

    title = 'relative count plot of sex with benign_malignant',

    size=2.8

)
plot_relative_distribution(

    df=train,

    feature='anatom_site_general_challenge',

    hue='benign_malignant',

    title = 'relative count plot of anatom_site_general_challenge with benign_malignant',

    size=3

)
train['age_approx'].iplot(

    kind='hist',

    bins=30,

    color='blue',

    xTitle='Age',

    yTitle='Count',

    title='Age Distribution'

)
import PIL

from PIL import Image, ImageDraw





def display_images(images, title=None): 

    f, ax = plt.subplots(5,3, figsize=(18,22))

    if title:

        f.suptitle(title, fontsize = 30)



    for i, image_id in enumerate(images):

        image_path = os.path.join(BASE_PATH, f'jpeg/train/{image_id}.jpg')

        image = Image.open(image_path)

        

        ax[i//3, i%3].imshow(image) 

        image.close()       

        ax[i//3, i%3].axis('off')



        benign_malignant = train[train['image_name'] == image_id]['benign_malignant'].values[0]

        ax[i//3, i%3].set_title(f"image_name: {image_id}\nSource: {benign_malignant}", fontsize="15")



    plt.show() 
benign = train[train.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title = 'benign images')
malignant = train[train.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images')
female_patients = train[train.sex == 'female']

benign = female_patients[female_patients.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title='benign images for female patients')
female_patients = train[train.sex == 'female']

malignant = female_patients[female_patients.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images for female patients')
male_patients = train[train.sex == 'male']

benign = male_patients[male_patients.benign_malignant == 'benign'].sample(n=15, random_state=42)

display_images(benign.image_name.values, title='benign images for male patients')
malignant = male_patients[male_patients.benign_malignant == 'malignant'].sample(n=15, random_state=42)

display_images(malignant.image_name.values, title='malignant images for male patients')
anatom_sites = [ site for site in list(train.anatom_site_general_challenge.unique()) if type(site) != float ]
for site in anatom_sites[:2]:

    site_df = train[train.anatom_site_general_challenge == site].sample(n=15, random_state=42)

    display_images(site_df.image_name.values, title = f'patient images for anatom_site == {site}')
#os.listdir('../input/siim-isic-melanoma-classification/jpeg/train/')

def imtocsv(path,resize):

    from PIL import Image

    r=resize

    A=list(os.listdir(path))

    D=np.zeros((len(A),r*r*3))

    for i in range(len(A)):

        image = Image.open(path+A[i])

        out=image.resize((r,r))

        out=np.array(out)

        out=out.flatten()

        D[i,]=out

    col_list = ['x' + str(x) for x in range(0,r*r*3)]

    df= pd.DataFrame(D,columns=col_list)

    image_name=[]

    for j in range(len(A)):

        image_name.append(A[j][:-4])

    df.insert(0,'image_name',image_name,True)

    return df
tr='../input/siim-isic-melanoma-classification/jpeg/train/'

ts='../input/siim-isic-melanoma-classification/jpeg/test/'
train_df=imtocsv(tr,28)

test_df=imtocsv(ts,28)

print(train_df.head())

print(test_df.head())
train.drop(['patient_id','sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant'],axis=1,inplace=True)

test.drop(['patient_id','sex','age_approx','anatom_site_general_challenge'],axis=1,inplace=True)
tr=pd.merge(train,train_df,how='left',on='image_name')

ts=pd.merge(test,test_df,how='left',on='image_name')
tr.to_csv("train_28.csv", index=False)

ts.to_csv("test_28.csv", index=False)
tr.drop('image_name',axis=1,inplace=True)

ts.drop('image_name',axis=1,inplace=True)
#tr[tr.image_name=='ISIC_7685852']
x_cols=tr.columns[tr.columns!='target']

x_cols
from sklearn.model_selection import train_test_split
X = pd.DataFrame(tr[x_cols])

y = pd.Series(tr.target.values)

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=.3,

                                                    random_state=1234123)



# look at the distrubution of labels in the train set

pd.Series(y_train).value_counts()
from sklearn.svm import SVC

# define support vector classifier

svm = SVC(kernel='rbf', probability=True, random_state=42)



# fit model

svm.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, auc, roc_curve



# generate predictions

y_pred = svm.predict(X_test)



# calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print('Model accuracy is: ', accuracy)
# predict probabilities for X_test using predict_proba

probabilities = svm.predict_proba(X_test)



# select the probabilities for label 1.0

y_proba = probabilities[:, 1]



# calculate false positive rate and true positive rate at different thresholds

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)



# calculate AUC

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

# plot the false positive rate on the x axis and the true positive rate on the y axis

roc_plot = plt.plot(false_positive_rate,

                    true_positive_rate,

                    label='AUC = {:0.2f}'.format(roc_auc))



plt.legend(loc=0)

plt.plot([0,1], [0,1], ls='--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate');
# generate predictions

y_pred = svm.predict(ts)

probabilities = svm.predict_proba(X_test)

print(probabilities.head())
predictions = pd.DataFrame(svm.predict_proba(ts))

sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

sample.loc[:, "target"] = predictions[1]



sample.to_csv("submission_svm.csv", index=False)