import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') # For plots

plt.rcParams['figure.figsize'] = (10, 8)

import numpy as np # linear algebra

import os

import glob

import pydicom
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

sub = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

train_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

test_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
# lets check dimensions of the train and test datasets

train.shape, test.shape,sub.shape
# check training data

train.head()
# check test dataset

test
# sample submission

sub.head()
# check for null values

train.isnull().sum()
# nº of unique patients in the traiing dataset

train.Patient.nunique()
# distribution plot for FVC

sns.distplot(train.FVC, hist = False,color = "darkred")

plt.title("FVC Distribution")
df =  train.groupby("Patient").count()["Weeks"].value_counts()

df
# lets check the nº of males/females in the dataset

sizes =  [len(train[train.Sex == "Male"]),len(train[train.Sex == "Female"])]

explode = (0.1,0)  # explode 1st slice

colors = ['Green',"Cyan"]

plt.pie(sizes, explode=explode, labels=train.Sex.unique(), colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title("Pie Chart for Gender Distribution")

plt.show()
# checking smoking status 

train.SmokingStatus.value_counts()
# lets check smoking distribution

train.SmokingStatus.unique()

sizes =  [len(train[train.SmokingStatus == "Ex-smoker"]),len(train[train.SmokingStatus == "Currently smokes"]),len(train[train.SmokingStatus == "Never smoked"])]

explode = (0,0,0.1)

colors = ['Cyan',"Green","Red"]

plt.pie(sizes, explode=explode, labels=train.SmokingStatus.unique(), colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.title("Pie Chart for SmokingStatus Distribution")

plt.show()
train.Age.min(),train.Age.max()
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



# Patient age group

ageGroupLabel = 'Below 60', '60-70', '70-80', 'Above 80'



below60 = len(train[train.Age<60])

sixty_to_seventy = len(train[(train['Age']>=60) & (train['Age']<= 70)])

seventy_to_eighty = len(train[(train['Age']>70) & (train['Age']<= 80)])

above80 = len(train[train.Age>80])



# Number of Guests expected in age group

patientNumbers     = [below60, sixty_to_seventy, seventy_to_eighty,above80] 



explode = (0, 0, 0, 0.1)

colors  = ("green","indigo","blue", "red")



# Draw the pie chart

ax1.pie(patientNumbers,explode = explode,colors = colors,labels = ageGroupLabel,autopct = '%1.2f',startangle = 90)



# Aspect ratio

ax1.axis('equal')





# distribution plot for Age

sns.distplot(train.Age, hist = False, color = "indigo")

plt.suptitle("Age Distribution")



plt.show()
train.Weeks.min(),train.Weeks.max()
# Patient age group



below10 = len(train[train.Weeks<10])

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



sizes = [below10, eleven_20, twentyone_30, thirtyone_40, fortyone_50, fiftyone_60,sixtyone_70,seventyone_80,eightyone_90,ninetyone_100,hundredone_110,hundredten_120,above120]

labels =  'below10','eleven_20', 'twentyone_30', 'thirtyone_40', 'fortyone_50', 'fiftyone_60','sixtyone_70','seventyone_80','eightyone_90','ninetyone_100','hundredone_110','hundredten_120','above120'





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

sns.distplot(train.Weeks, hist = False, color = "indigo")

plt.suptitle("Weeks Distribution")



plt.show()
# heat map

corrMatrix = train.corr()

mask = np.triu(corrMatrix)

sns.heatmap(corrMatrix,

            annot=True,

            fmt='.1f',

            cmap='coolwarm',            

            mask=mask,

            linewidths=1,

            cbar=False)

plt.show()
# Pair plot

sns.pairplot(train)
sns.pairplot(train,hue="SmokingStatus")

plt.show()
sns.pairplot(train,hue="Sex")

plt.show()
df_0 = train[train.Patient == train.Patient[0]]

sns.pairplot(df_0)

plt.show()
df_1 = train[train.Patient == train.Patient[1]]

sns.pairplot(df_1)

plt.show()
df_2 = train[train.Patient == train.Patient[2]]

sns.pairplot(df_2)

plt.show()
# lets group Male & Female data

grp = train.groupby("Sex")



# draw a plot to display mean of FVC for Males and Females

splot = sns.barplot(x=train.Sex.unique(),y= grp["FVC"].mean())



for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel("Sex",fontsize = 30)

plt.ylabel("Mean FCV",fontsize = 30)

plt.title ("FVC Mean for Male Vs Female",fontsize = 30) 

plt.show()
# lets create groupwise data for different categories of smoking status

grp = train.groupby("SmokingStatus")



# draw a barplot for different smoking categories vs mean FVC for individual categories

splot  = sns.barplot(x=train.SmokingStatus.unique(),y= grp["FVC"].mean())



for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel("Smoking Status",fontsize = 30)

plt.ylabel("Mean FCV",fontsize = 30)

plt.title ("FVC Mean for different Smoking Categories",fontsize = 30) 

plt.show()
plt.figure(figsize=(12,8))

# lets create bins for weeks

train["Weeks_Bins"] = pd.cut(train["Weeks"], 13, duplicates = 'drop') # creating bins     



# group the data for the bins created above

grp = train.groupby("Weeks_Bins")



# draw a barplot for different weeks bins and mean FVC

splot = sns.barplot(x=train.Weeks_Bins.unique(),y= grp["FVC"].mean())

splot.set_xticklabels(splot.get_xticklabels(),rotation = 45)



for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel("Weeks Bins",fontsize = 30)

plt.ylabel("Mean FCV",fontsize = 30)

plt.title ("FVC Mean for different Weeks Bins",fontsize = 30) 

plt.show()
train["Age_Bins"] = pd.cut(train["Age"], 4, duplicates = 'drop') # creating bins     

grp = train.groupby("Age_Bins")

splot = sns.barplot(x=train.Age_Bins.unique(),y= grp["FVC"].mean())

splot.set_xticklabels(splot.get_xticklabels(),rotation = 45)



for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.xlabel("Age Bins",fontsize = 30)

plt.ylabel("Mean FCV",fontsize = 30)

plt.title ("FVC Mean for different Age Bins",fontsize = 30) 

plt.show()
p_id = list(train.Patient.sample(3))

p_id
# lets draw a plot for a random patient, we will check how its FVC varies as value of Weeks is changed

plt.plot(train[train.Patient == p_id[0]].Weeks,train[train.Patient == p_id[0]].FVC,color = "darkblue")



plt.xlabel("Weeks",fontsize = 30)

plt.ylabel("FCV",fontsize = 30)

title = "FVC for patient:"+ p_id[0]

plt.title (title,fontsize = 25) 

plt.show()
# lets draw a plot for a random patient, we will check how its FVC varies as value of Weeks is changed

plt.plot(train[train.Patient == p_id[1]].Weeks,train[train.Patient == p_id[1]].FVC, color = "darkgreen")



plt.xlabel("Weeks",fontsize = 30)

plt.ylabel("FCV",fontsize = 30)

title = "FVC for patient:"+ p_id[1]

plt.title (title,fontsize = 25)  

plt.show()
# lets draw a plot for a random patient, we will check how its FVC varies as value of Weeks is changed

plt.plot(train[train.Patient == p_id[2]].Weeks,train[train.Patient == p_id[2]].FVC,color = "purple")



plt.xlabel("Weeks",fontsize = 30)

plt.ylabel("FCV",fontsize = 30)

title = "FVC for patient:"+ p_id[2]

plt.title (title,fontsize = 25) 

plt.show()
# review training directory

p_sizes = [] # list of no. of dcm files present for each patientx



for d in os.listdir(train_dir):

    print("Patient '{}' has {} scans".format(d, len(os.listdir(train_dir + d))))

    p_sizes.append(len(os.listdir(train_dir + d)))



print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir(train_dir)), 

                                                      len(glob.glob(train_dir+ "/*/*.dcm"))))
# lets visualize trainig data

p = sns.color_palette()

plt.hist(p_sizes, color=p[2])

plt.ylabel('Number of patients')

plt.xlabel('Count of DICOM files')

plt.title('Histogram of DICOM count per patient - Training Data')
# review test directory

p_sizes = [] # list of no. of dcm files present for each patientx



for d in os.listdir(test_dir):

    print("Patient '{}' has {} scans".format(d, len(os.listdir(test_dir + d))))

    p_sizes.append(len(os.listdir(test_dir + d)))



print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir(test_dir)), 

                                                      len(glob.glob(test_dir+ "/*/*.dcm"))))
# lets visualize image distribution per patient

plt.hist(p_sizes, color=p[3])

plt.ylabel('Number of patients')

plt.xlabel('Count of DICOM files')

plt.title('Histogram of DICOM count per patient - Test Data')
sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob(train_dir+ "/*/*.dcm")]

print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(np.min(sizes), 

                                                       np.max(sizes), np.mean(sizes), np.std(sizes)))
# read a dcm file for patient ID00368637202296470751086

dcm = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00368637202296470751086/270.dcm'

print('Filename: {}'.format(dcm))

dcm = pydicom.read_file(dcm)
print(dcm)
# display the image read above

img = dcm.pixel_array

img[img == -2000] = 0



plt.axis('off')

plt.imshow(img)

plt.show()



plt.axis('off')

plt.imshow(-img) # Invert colors with -

plt.show()
# helper function

def dicom_to_image(filename):

    dcm = pydicom.read_file(filename)

    img = dcm.pixel_array

    img[img == -2000] = 0

    return img
# lets display some 20 images at random

files = glob.glob(train_dir + "/*/*.dcm")



f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)
# function to sort patients dcm images

def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



# Returns a list of images for that patient_id, in ascending order of Slice Location

def load_patient(patient_id):

    files = glob.glob(train_dir + patient_id + "/*.dcm")

    imgs = {}

    for f in files:

        dcm = pydicom.read_file(f)

        img = dcm.pixel_array

        img[img == -2000] = 0

        sl = get_slice_location(dcm)

        imgs[sl] = img

        

    # Not a very elegant way to do this

    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]

    return sorted_imgs

# display  all dcm images for patient ID00210637202257228694086

pat = load_patient('ID00210637202257228694086')

f, plots = plt.subplots(31, 10, sharex='all', sharey='all', figsize=(10, 31))

for i in range(303):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)
# display  all dcm images for patient ID00368637202296470751086

pat = load_patient('ID00368637202296470751086')

f, plots = plt.subplots(35, 10, sharex='all', sharey='all', figsize=(10, 35))

for i in range(341):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)
# display  all dcm images for patient ID00169637202238024117706

pat = load_patient('ID00169637202238024117706')

f, plots = plt.subplots(12, 10, sharex='all', sharey='all', figsize=(10, 12))

for i in range(115): 

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)
import matplotlib.animation as animation

from IPython.display import HTML
# stack up all the 2D slices to make up a 3D volume

def load_scan(patient_name):

    

    patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}'), key=(lambda f: int(f.split('.')[0])))

    volume = np.zeros((len(patient_directory), 512, 512))



    for i, img in enumerate(patient_directory):

        img_slice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{img}')

        volume[i] = img_slice.pixel_array

            

    return volume
patient_scan = load_scan('ID00368637202296470751086')
%%capture

fig = plt.figure(figsize=(8, 8))



imgs = []

for ps in patient_scan:

    img = plt.imshow(ps, animated=True, cmap=plt.cm.bone)

    plt.axis('off')

    imgs.append([img])
vid = animation.ArtistAnimation(fig, imgs, interval=25, blit=False, repeat_delay=1000)
# lets play the video 

HTML(vid.to_html5_video())