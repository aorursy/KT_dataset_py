# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import plotly.express as px

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input//covidindiatravelhistory/Patient_Travel_History_Files/PatientMovementTrackingTemplate.xlsx')

df.head()
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Address"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Covid-19 Travelling History ')
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P35P36P37_Updated_Page_2_of_2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
travel_file = '../input/covidindiatravelhistory/Patient_Travel_History_Files/Kerala_UnknownPatientIDs.txt'

with open(travel_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P42_EnteredDataInSpreadsheet.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(travel_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'red', edgecolor = 'green',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (travel_file), fontsize=24)

    plt.show()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P51_EnteredDataInSpreadsheet.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
travel_file = '../input/covidindiatravelhistory/Patient_Travel_History_Files/Kerala_UnknownPatientIDs.txt'

plotWordFrequency(travel_file)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P42_EnteredDataInSpreadsheet.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P156.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
image = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P109.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Create our shapening kernel, we don't normalize since the 

# the values in the matrix sum to 1

kernel_sharpening = np.array([[-1,-1,-1], 

                              [-1,9,-1], 

                              [-1,-1,-1]])



# applying different kernels to the input image

sharpened = cv2.filter2D(image, -1, kernel_sharpening)





plt.subplot(1, 2, 2)

plt.title("Image Sharpening")

plt.imshow(sharpened)



plt.show()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P118.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
# Load our new image

image = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P175.jpeg', 0)



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



#Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.subplot(3, 2, 2)

plt.title("Threshold Binary")

plt.imshow(thresh1)



# It's good practice to blur images as it removes noise

image = cv2.GaussianBlur(image, (3, 3), 0)



# Using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.title("Otsu's Thresholding")

plt.imshow(th2)





plt.subplot(3, 2, 5)

# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3)

plt.show()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covidindiatravelhistory/Patient_Travel_History_Files/P140.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)