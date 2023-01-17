from glob import glob

import os

import pandas as pd



#checnking the input files

print(os.listdir("../input/siim-acr-pneumothorax-segmentation-data"))



#reading all dcm files into train and text

train = sorted(glob("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm"))

test = sorted(glob("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm"))

print("train files: ", len(train))

print("test files: ", len(test))



pd.reset_option('max_colwidth')



#reading the csv

print("the csv with the labels: -1 means no Pneumothorax, othervise there is an encoding for the place of Pneumothorax")

masks = pd.read_csv("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv", delimiter=",")

masks.head()
import pydicom

import matplotlib.pyplot as plt



#displaying the image

img = pydicom.read_file(train[0]).pixel_array

plt.imshow(img, cmap='bone')

plt.grid(False)



#displaying metadata

data = pydicom.dcmread(train[0])

print(data)
#dataframe to ease the access

patients = []

missing = 0



pd.reset_option('max_colwidth')



for t in train:

    data = pydicom.dcmread(t)

    patient = {}

    patient["UID"] = data.SOPInstanceUID

    try:

        encoded_pixels = masks[masks["ImageId"] == patient["UID"]].values[0][1]

        patient["EncodedPixels"] = encoded_pixels

    except:

        missing = missing + 1

    patient["Age"] = data.PatientAge

    patient["Sex"] = data.PatientSex

    patient["Modality"] = data.Modality

    patient["BodyPart"] = data.BodyPartExamined

    patient["ViewPosition"] = data.ViewPosition

    patient["path"] = "../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/" + data.StudyInstanceUID + "/" + data.SeriesInstanceUID + "/" + data.SOPInstanceUID + ".dcm"

    patients.append(patient)



print("missing labels: ", missing)

#pd.set_option('display.max_colwidth', -1)

df_patients = pd.DataFrame(patients, columns=["UID", "EncodedPixels", "Age", "Sex", "Modality", "BodyPart", "ViewPosition", "path"])

print("images with labels: ", df_patients.shape[0])

df_patients.head()
import matplotlib as mpl

import numpy as np



#gender

men = df_patients[df_patients["Sex"] == "M"].shape[0]

women = df_patients.shape[0] - men

print(men, women)





#illness

healthy = df_patients[df_patients["EncodedPixels"] == " -1"].shape[0]

ill = df_patients.shape[0] - healthy

print(healthy, ill)



#gender + illness

men_h = df_patients[(df_patients["Sex"] == "M") & (df_patients["EncodedPixels"] == " -1")].shape[0]

men_ill = men - men_h

women_h = df_patients[(df_patients["Sex"] == "F") & (df_patients["EncodedPixels"] == " -1")].shape[0]

women_ill = women - women_h

print(men_h, men_ill, women_h, women_ill)



perc = [str(round(men_ill/107.12, 1)) + "% \n ill", "healthy \n" + str(round(men_h/107.12, 1)) + "%", "healthy \n" + str(round(women_h/107.12, 1)) + "%",str(round(women_ill/107.12, 1)) + "% \n ill"]



fig, ax = plt.subplots(1, 3, figsize=(15, 5))



fig.suptitle("Gender and Pneumothorax distributions", fontsize=24, y=1.1)



mpl.rcParams['font.size'] = 12.0



#circle for donut chart

circle0 = plt.Circle( (0,0), 0.6, color = 'white')

circle1 = plt.Circle( (0,0), 0.4, color = 'white')

circle2 = plt.Circle( (0,0), 0.6, color = 'white')



#men women

ax[0].pie([men, women], labels=["men", "women"], colors=["#42A5F5", "#E57373"], autopct='%1.1f%%', pctdistance=0.8, startangle=90)

ax[0].add_patch(circle0)

ax[0].axis('equal')



#gender healthy

mypie, _ = ax[2].pie([men, women], radius=1.3, labels=["men", "women"], colors=["#42A5F5", "#E57373"], startangle=90)

plt.setp( mypie, width=0.3, edgecolor='white')



mypie2, _ = ax[2].pie([ men_ill, men_h, women_h, women_ill], radius = 1.3 - 0.3, labels=perc, labeldistance=0.61,

                      colors = ["#FFB74D", "#9CCC65", "#9CCC65", "#FFB74D"], startangle=90)

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)



#healthy ill

ax[1].pie([healthy, ill], labels=["healthy", "ill"], colors=["#9CCC65", "#FFB74D"], autopct='%1.1f%%', pctdistance=0.8, startangle=135)

ax[1].add_patch(circle2)

ax[1].axis('equal')  



plt.tight_layout()

plt.show()
import numpy as np

#group into bins the same aged men and women with histogram --> all of them and ill of them



#convert he Age column to int

df_patients["Age"] = pd.to_numeric(df_patients["Age"])



sorted_ages = np.sort(df_patients["Age"].values)

print(sorted_ages)
import seaborn as sns

plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(17, 5))

plt.hist(sorted_ages[:-2], bins=[i for i in range(100)])

plt.title("All patients age histogram", fontsize=18, pad=10)

plt.xlabel("age", labelpad=10)

plt.xticks([i*10 for i in range(11)])

plt.ylabel("count", labelpad=10)

plt.show()
#calculating all and ill men and women histograms

bins = [i for i in range(100)]

plt.style.use('seaborn-whitegrid')



all_men = np.histogram(df_patients[df_patients["Sex"] == "M"]["Age"].values, bins=bins)[0]

all_women = np.histogram(df_patients[df_patients["Sex"] == "F"]["Age"].values, bins=bins)[0]



ill_men = np.histogram(df_patients[(df_patients["Sex"] == "M") & (df_patients["EncodedPixels"] != ' -1')]["Age"].values, bins=bins)[0]

ill_women = np.histogram(df_patients[(df_patients["Sex"] == "F") & (df_patients["EncodedPixels"] != ' -1')]["Age"].values, bins=bins)[0]



fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(17, 16))



fig.suptitle("The presence of Pneumothorax at particular ages and genders", fontsize=22, y=0.96)



axes[0].margins(x=0.1, y=0.01)

m1 = axes[0].barh(bins[:-1], all_men, color='#90CAF9')

m2 = axes[0].barh(bins[:-1], ill_men, color='#0D47A1')

axes[0].set_title('Men', fontsize=18, pad=15)

axes[0].invert_xaxis()

axes[0].set(yticks=[i*5 for i in range(20)])

axes[0].tick_params(axis="y", labelsize=14)

axes[0].yaxis.tick_right()

axes[0].xaxis.tick_top()

axes[0].legend((m1[0], m2[0]), ('healthy', 'with Pneumothorax'), loc=2, prop={'size': 16})



locs = axes[0].get_xticks()



axes[1].margins(y=0.01)

w1 = axes[1].barh(bins[:-1], all_women, color='#EF9A9A')

w2 = axes[1].barh(bins[:-1], ill_women, color='#B71C1C')

axes[1].set_title('Women', fontsize=18, pad=15)

axes[1].xaxis.tick_top()

axes[1].set_xticks(locs)

axes[1].legend((w1[0], w2[0]), ('healthy', 'with Pneumothorax'), prop={'size': 17})



#for i, v in enumerate(depos["ItemViewCount"].values):

   #print(i, v)

    #axes[1].text(int(v) + 3, int(i)-0.25, str(v))

plt.show()
bodypart = df_patients["BodyPart"].values

print("Body parts:", list(set(bodypart)))



modality = df_patients["Modality"].values

print("Modality:", list(set(modality)))



view = list(df_patients["ViewPosition"].values)

print("View Positions: ", list(set(view)))



pa = view.count("PA")

ap = view.count("AP")

print(pa, ap)


basic_palette = sns.color_palette()

plt.style.use('seaborn-whitegrid')

plt.pie([pa, ap], labels = ["PA", "AP"], colors=[basic_palette[-2], basic_palette[4]], autopct='%1.1f%%', startangle=70)

plt.title("Occurrences of View positions", fontsize=16)
#mask functions from sample dataset

import numpy as np



def mask2rle(img, width, height):

    rle = []

    lastColor = 0;

    currentPixel = 0;

    runStart = -1;

    runLength = 0;



    for x in range(width):

        for y in range(height):

            currentColor = img[x][y]

            if currentColor != lastColor:

                if currentColor == 255:

                    runStart = currentPixel;

                    runLength = 1;

                else:

                    rle.append(str(runStart));

                    rle.append(str(runLength));

                    runStart = -1;

                    runLength = 0;

                    currentPixel = 0;

            elif runStart > -1:

                runLength += 1

            lastColor = currentColor;

            currentPixel+=1;



    return " ".join(rle)



def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)
df_pneumo = df_patients[df_patients["EncodedPixels"] != ' -1']



#print(df_pneumo.values[3][2], df_pneumo.values[3][3])



mask = rle2mask(df_pneumo.values[3][1], 1024, 1024)

mask = np.rot90(mask, 3) #rotating three times 90 to the right place

mask = np.flip(mask, axis=1)

img = pydicom.read_file(df_pneumo.values[3][-1]).pixel_array



fig = plt.figure(figsize=(15, 10))

a = fig.add_subplot(1, 3, 1)

plt.imshow(img, cmap='bone') #original x-ray

a.set_title("Original x-ray image")

plt.grid(False)

plt.axis("off")



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mask, cmap='binary')



a.set_title("The mask")

plt.grid(False)

plt.xticks([])

plt.yticks([])



a = fig.add_subplot(1, 3, 3)

plt.imshow(img, cmap='bone')

plt.imshow(mask, cmap='binary', alpha=0.3)

a.set_title("Mask on the x-ray: air in the pleura")



plt.axis("off")



plt.grid(False)



mask = rle2mask(df_pneumo.values[6][1], 1024, 1024)

mask = np.rot90(mask, 3) #rotating three times 90 to the right place

mask = np.flip(mask, axis=1)

img = pydicom.read_file(df_pneumo.values[6][-1]).pixel_array



fig = plt.figure(figsize=(15, 10))

a = fig.add_subplot(1, 3, 1)

plt.imshow(img, cmap='bone') #original x-ray

a.set_title("Original x-ray image")

plt.grid(False)

plt.axis("off")



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mask, cmap='binary')



a.set_title("The mask")

plt.grid(False)

plt.xticks([])

plt.yticks([])



a = fig.add_subplot(1, 3, 3)

plt.imshow(img, cmap='bone')

plt.imshow(mask, cmap='binary', alpha=0.3)

a.set_title("Mask on the x-ray: air in the pleura")



plt.axis("off")



plt.grid(False)
area = []

pos = []

pa_area = []

ap_area = []



c = 0



for p in df_pneumo.values:

    try:

        mask = rle2mask(p[1], 1024, 1024)

        pixels = np.count_nonzero(mask)

        area.append(pixels)

        pos.append(p[6])

        if p[6] == "AP":

            ap_area.append(pixels)

        else:

            pa_area.append(pixels)

    except:

        c = c + 1



print("missing labels", c)

print("all area", np.sort(np.array(area)))

#print("ap area", np.sort(np.array(ap_area)))

print("pa area", np.sort(np.array(pa_area)))
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(17, 5))

plt.hist(area, bins=[i*500 for i in range(340)])

plt.title("The affected area by Pneumothorax", fontsize=18, pad=10)

plt.xlabel("area [pixels]", labelpad=10)

#plt.xticks([i*10 for i in range(1000)])

plt.ylabel("count of patient in groups", labelpad=10)

plt.show()
fig = plt.figure(figsize=(17, 5))

basic_palette = sns.color_palette()



ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)



sns.boxplot(x=area, y=pos, palette={"AP": basic_palette[4], "PA": basic_palette[-2]})#, height=[0.6, 0.4])

ax1.set_xlabel("area [pixels]", fontsize=14, labelpad=10)

ax1.set_ylabel("view position", fontsize=14, labelpad=10)

ax1.set_title("Affected area vs view position", fontsize=18, pad=10)



ax2 = plt.subplot2grid((1, 3), (0, 2))



ax2.pie([pa, ap], labels = ["PA", "AP"], colors=[basic_palette[-2], basic_palette[4]], autopct='%1.1f%%', startangle=70)

ax2.set_title("Occurrences of View positions", fontsize=18)

c = 0

ap_sum = np.array([[0 for i in range(1024)] for j in range(1024)])

pa_sum = np.array([[0 for i in range(1024)] for j in range(1024)])



ap = 0

pa = 0



for p in df_pneumo.values:

    try :

        mask = rle2mask(p[1], 1024, 1024)

        mask = np.rot90(mask, 3) #rotating three times 90 to the right place

        mask = np.flip(mask, axis=1)

        if p[6] == 'AP':

            ap_sum = ap_sum + mask

            ap = ap + 1

        else:

            pa_sum = pa_sum + mask

            pa = pa + 1

    except:

        c = c + 1
fig = plt.figure(figsize=(17, 5))

basic_palette = sns.color_palette()



ax1 = plt.subplot2grid((1, 3), (0, 0))

ax1.imshow(pa_sum, cmap='magma_r')

ax1.set_title("All PA positioned masks", fontsize=18, pad=15)



#colorbar

maxval = np.max(pa_sum)

cmap = plt.get_cmap('magma_r', maxval)



norm = mpl.colors.Normalize()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

sm.set_array([])



ticks=[1/(maxval+1)/2, 0.5, 1-1/(maxval+1)/2]



cb_1 = plt.colorbar(sm,ticks=[0, 1], fraction=0.046, ax=ax1)#ticks and boundaries

cb_1.ax.set_yticklabels(["no Pneu", str(int(maxval))]) #label of colormap

cb_1.ax.yaxis.set_label_position('left')



plt.grid(False)

plt.xticks([])

plt.yticks([])



ax2 = plt.subplot2grid((1, 3), (0, 1))

ax2.pie([pa, ap], labels = ["PA", "AP"], colors=[basic_palette[-2], basic_palette[4]], autopct='%1.1f%%', startangle=70)

ax2.set_title("Occurrences of View positions", fontsize=18)



ax3 = plt.subplot2grid((1, 3), (0, 2))

ax3.imshow(ap_sum, cmap='magma_r')

ax3.set_title("All AP positioned masks", fontsize=18, pad=15)



maxval = np.max(ap_sum)

cmap2 = plt.get_cmap('magma_r', maxval)



norm = mpl.colors.Normalize()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

sm.set_array([])



cb_2 = plt.colorbar(sm,ticks=[0, 1], fraction=0.046, ax=ax3)#ticks and boundaries

cb_2.ax.set_yticklabels(["no Pneu", str(int(maxval))]) #label of colormap

cb_2.ax.yaxis.set_label_position('left')



plt.grid(False)

plt.xticks([])

plt.yticks([])