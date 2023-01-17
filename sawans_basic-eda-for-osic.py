import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (10,8)

import os

import glob

import pydicom
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



train_dir = '../input/osic-pulmonary-fibrosis-progression/train'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test'
#Checking dimensions of all datasets



train.shape, test.shape, sub.shape
# Exploring train data

train.head()
# Exploring test data

test
# Explore sample submission

sub.head()
# Checking the number of records in sample submission for a patient

sub_patient1 = sub[sub['Patient_Week'].str.contains('ID00419637202311204720264')]

sub_patient1.info()
# Checking the range of values for week 

sub_patient1.head(1), sub_patient1.tail(1)
# Checking for null values

train.isnull().sum()
# Check number of unique patients in train data

train['Patient'].nunique()
#Check types of features available in the dataset

train.dtypes
#plt.hist(train['FVC'])

sns.distplot(train['FVC'],hist=False,color='darkred')

plt.title('FVC Distribution')
df = train.groupby('Patient').count()['Weeks'].value_counts()

df
# Lets check number of males/females in the dataset

sizes = [len(train[train['Sex']=='Male']), len(train[train['Sex']=='Female'])]

explode = (0.1,0) #explde first slice

colors = ['blue','pink']



plt.pie(sizes, explode=explode, labels= train['Sex'].unique(), colors= colors, autopct= '%1.1f%%', 

        shadow=True, startangle=140)

plt.axis('equal')

plt.title('Pie Chart for Gender Distribution')

plt.show()
train['SmokingStatus'].value_counts()
sizes = [len(train[train['SmokingStatus']=='Ex-smoker']), len(train[train['SmokingStatus']=='Never smoked']), 

        len(train[train['SmokingStatus']=='Currently smokes'])]

explode = (0,0,0.1)

colors = ['blue','red','green']

plt.pie(sizes, explode=explode, colors = colors, labels= train['SmokingStatus'].unique(), autopct='%1.1f%%', 

        shadow=True, startangle=140)



plt.axis('equal')

plt.title('Pie Chart for Smoking Status distribution')

plt.show()
train['Age'].min(), train['Age'].max()
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))



#Patient age group

ageGroupLabel = 'Below 60', '60-70', '70-80', 'Above 80'



below60 = len(train[train.Age<60])

sixty_to_seventy = len(train[(train['Age']>=60) & (train['Age']<= 70)])

seventy_to_eighty = len(train[(train['Age']>70) & (train['Age']<= 80)])

above80 = len(train[train.Age>80])



patientNumbers = [below60, sixty_to_seventy, seventy_to_eighty,above80]

explode = (0,0,0,0.1)

colors = ['green','indigo','blue','red']



#Draw the pie chart

ax1.pie(patientNumbers, explode=explode, colors= colors, labels= ageGroupLabel, autopct = '%1.2f',startangle = 90)



#Aspect ratio

ax1.axis('equal')



# Distribution plot for age

sns.distplot(train['Age'],hist= False, color='red')

plt.suptitle('Age Distribution')

plt.show()

train['Weeks'].min(), train['Weeks'].max()
# Dividing weeks into bins of 10 to visualize



below10 = len(train[train['Weeks']<=10])

eleven_20 = len(train[(train['Weeks']>=11) & (train['Weeks']<= 20)])

twentyone_30 = len(train[(train['Weeks']>20) & (train['Weeks']<= 30)])

thirtyone_40 = len(train[(train['Weeks']>30) & (train['Weeks']<= 40)])

fortyone_50 = len(train[(train['Weeks']>40) & (train['Weeks']<= 50)])

fiftyone_60 = len(train[(train['Weeks']>50) & (train['Weeks']<= 60)])

sixtyone_70 = len(train[(train['Weeks']>60) & (train['Weeks']<= 70)])

seventyone_80 = len(train[(train['Weeks']>70) & (train['Weeks']<= 80)])

eightyone_90 = len(train[(train['Weeks']>80) & (train['Weeks']<= 90)])

ninetyone_100 = len(train[(train['Weeks']>90) & (train['Weeks']<= 100)])

hundredone_110 = len(train[(train['Weeks']>100) & (train['Weeks']<= 110)])

hundredten_120 = len(train[(train['Weeks']>110) & (train['Weeks']<= 120)])

above120 = len(train[train.Weeks>120])



sizes = [below10, eleven_20, twentyone_30, thirtyone_40, fortyone_50, fiftyone_60,sixtyone_70,

         seventyone_80,eightyone_90,ninetyone_100,hundredone_110,hundredten_120,above120]

labels =  ['below10','eleven_20', 'twentyone_30', 'thirtyone_40', 'fortyone_50', 'fiftyone_60','sixtyone_70',

'seventyone_80','eightyone_90',

'ninetyone_100','hundredone_110','hundredten_120','above120']



fig1, (ax1, ax2)= plt.subplots(1,2,figsize=(15, 10))

theme = plt.get_cmap('prism')

ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])

_, _ = ax1.pie(sizes, startangle=90)



ax1.axis('equal')

total = sum(sizes)

ax1.legend(

    loc='upper left',

    labels=['%s, %1.1f%%' % (

        l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],

    prop={'size': 11},

    bbox_to_anchor=(0.0, 1),

    bbox_transform=fig1.transFigure

)



# distribution plot for Weeks

sns.distplot(train['Weeks'], hist = False, color = "indigo")

plt.suptitle("Weeks Distribution")



plt.show()

# Heatmap

corrMatrix = train.corr()

mask = np.triu(corrMatrix)

sns.heatmap(corrMatrix,annot=True,cmap='coolwarm',

           linewidths=1, cbar=False, fmt='.1f',

           mask=mask)

plt.show()
# pair plot

sns.pairplot(train)
sns.pairplot(train,hue='SmokingStatus')

plt.show()
sns.pairplot(train,hue='Sex')

plt.show()
df_100 = train[train['Patient']==train['Patient'][100]]

sns.pairplot(df_100)

plt.show()
grp_sex  = train.groupby('Sex')



#draw a plot to display mean of FVC for males and females

splot = sns.barplot(x=train['Sex'].unique(),y= grp_sex['FVC'].mean())



plt.xlabel("Sex",fontsize=30)

plt.ylabel("Mean FVC",fontsize=30)

plt.title("Mean FVC Males vs Mean FVC Femailes")

plt.show()
grp_smoking = train.groupby('SmokingStatus')



splot = sns.barplot(x= train['SmokingStatus'].unique(),y=grp_smoking['FVC'].mean())



plt.xlabel("Smoking Status",fontsize=30)

plt.ylabel("Mean FVC",fontsize=30)

plt.title("Mean FVC for different Smoking categiries")

plt.show()
plt.figure(figsize=(12,8))



#Creating bins for weeks

train['Weeks_bins'] = pd.cut(train['Weeks'],13, duplicates='drop')



#Group the data by bins

grp_weeks = train.groupby('Weeks_bins')



#Drawing barplots

splot = sns.barplot(x=train['Weeks_bins'].unique(),y=grp_weeks['FVC'].mean())



plt.xlabel("Weeks Bins",fontsize=30)

plt.ylabel("Mean FVC",fontsize=30)

plt.title("Mean FVC for different Weeks bins")

plt.show()
#Creating bins

train['Age_bins'] = pd.cut(train['Age'],4,duplicates='drop')



#Group by Age bins

grp_Age = train.groupby('Age_bins')



#Drawing plot

splot = sns.barplot(x= train['Age_bins'].unique(),y=grp_Age['FVC'].mean())



plt.xlabel("Age Bins",fontsize=30)

plt.ylabel("Mean FVC",fontsize=30)

plt.title("Mean FVC for different Age bins")

plt.show()
p_id = list(train['Patient'].sample(3))

p_id
# Drawing the distribution of FVC over weeks for a randomly selected patient 1



plt.plot(train[train['Patient']==p_id[0]].Weeks, train[train['Patient']==p_id[0]].FVC,

            color='darkblue')

    

plt.xlabel('Weeks',fontsize=30)

plt.ylabel('FVC',fontsize=30)

plt.title('FVC for Patient: '+p_id[0])

plt.show()
# Drawing the distribution of FVC over weeks for a randomly selected patient 2



plt.plot(train[train['Patient']==p_id[1]].Weeks, train[train['Patient']==p_id[1]].FVC,

            color='darkgreen')

    

plt.xlabel('Weeks',fontsize=30)

plt.ylabel('FVC',fontsize=30)

plt.title('FVC for Patient: '+p_id[1])

plt.show()
# Drawing the distribution of FVC over weeks for a randomly selected patient 1



plt.plot(train[train['Patient']==p_id[2]].Weeks, train[train['Patient']==p_id[2]].FVC,

            color='purple')

    

plt.xlabel('Weeks',fontsize=30)

plt.ylabel('FVC',fontsize=30)

plt.title('FVC for Patient: '+p_id[2])

plt.show()
# Checking the contents of train directory

p_sizes = [] #list of number of dcm files present for each patient



for dir in os.listdir(train_dir):

    print('Patient {} has {} scans'.format(dir, len(os.listdir(train_dir+ "/"+ dir))))

    p_sizes.append(len(os.listdir(train_dir+"/"+dir)))

    

print('----------')

print('Total number of patients {}. Total DCM files {}'.format(len(os.listdir(train_dir)),sum(p_sizes)))



#Visualizing DICOM count per patient

p = sns.color_palette()

plt.hist(p_sizes,color=p[2])

plt.xlabel('Count of DCM files')

plt.ylabel('Number of patients')

plt.title('Histogram of DICOM count per patient-Training Data')
# Checking the contents of test directory

p_sizes = [] #list of number of dcm files present for each patient



for dir in os.listdir(test_dir):

    print('Patient {} has {} scans'.format(dir, len(os.listdir(test_dir+ "/"+ dir))))

    p_sizes.append(len(os.listdir(test_dir+"/"+dir)))

    

print('----------')

print('Total number of patients {}. Total DCM files {}'.format(len(os.listdir(test_dir)),sum(p_sizes)))
#Visualizing DICOM count per patient - Test Data

p = sns.color_palette()

plt.hist(p_sizes,color=p[4])

plt.xlabel('Count of DCM files')

plt.ylabel('Number of patients')

plt.title('Histogram of DICOM count per patient-Test Data')
sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob(train_dir+"/*/*.dcm")]



print('DCM File sizes: min {:.3} MB, max {:.3} MB, avg {:.3} MB, std {:.3} MB'.format(

        np.min(sizes),np.max(sizes),np.mean(sizes),np.std(sizes)))
#read a dcm file for patient ID00368637202296470751086

dcm = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00368637202296470751086/270.dcm'

print('Filename: {}'.format(dcm))

dcm = pydicom.read_file(dcm)
print(dcm)
#display the image read above

img = dcm.pixel_array

img[img==-2000] = 0



plt.axis('off')

plt.imshow(img)

plt.show()



plt.axis('off')

plt.imshow(-img) #invert colors with - 

plt.show()
#Helper function

def dicom_to_image(filename):

    dcm = pydicom.read_file(filename)

    img = dcm.pixel_array

    img[img==-2000] == 0

    return img
#Lets display 20 images at random

files = glob.glob(train_dir+"/*/*.dcm")



f,plots = plt.subplots(4,5, sharex='col', sharey='row',figsize=(10,8))



for i in range(20):

    plots[i//5, i%5].axis('off')

    plots[i//5, i%5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)
#function to sort patient dcm images

def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



#Return a list of images for given patient in ascending order of slice location

def load_patient(patient_id):

    files = glob.glob(train_dir+'/'+patient_id+'/*.dcm')

    imgs={}

    for f in files:

        dcm = pydicom.read_file(f)

        img = dcm.pixel_array

        img[img== -2000] =0

        sl= get_slice_location(dcm)

        imgs[sl] = img

        

    

    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]

    return sorted_imgs
#display all the images for patient ID00210637202257228694086

pat = load_patient('ID00210637202257228694086')

f, plots = plt.subplots(31, 10, sharex='col', sharey='row', figsize=(10, 31))

for i in range(303):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)

    
import matplotlib.animation as animation

from IPython.display import HTML
#Stack up all 2D slices to make up the 3D Volume

def load_scan(patient_name):

    

    patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}'), key=(lambda f: int(f.split('.')[0])))

    volume = np.zeros((len(patient_directory), 512, 512))



    for i, img in enumerate(patient_directory):

        img_slice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{img}')

        volume[i] = img_slice.pixel_array

            

    return volume

patient_scan = load_scan('ID00368637202296470751086')
fig = plt.figure(figsize=(8, 8))



imgs = []

for ps in patient_scan:

    img = plt.imshow(ps, animated=True, cmap=plt.cm.bone)

    plt.axis('off')

    imgs.append([img])
vid = animation.ArtistAnimation(fig, imgs, interval=25, blit=False, repeat_delay=1000)
# lets play the video 

HTML(vid.to_html5_video())