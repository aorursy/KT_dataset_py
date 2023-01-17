import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#pip install glob

import glob

#conda install -c menpo opencv

import cv2

#pip install PIL

from PIL import Image

from matplotlib import pyplot as plt

import os

from scipy.interpolate import griddata

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#The top three lines save the processed data from the training videos. For this I'm reading it back

#res = pd.DataFrame(X)

#res['label']=Y

#res.to_csv('/kaggle/working/power_average.csv', Index = True )

res = pd.read_csv("/kaggle/input/powerdata/power_average.csv")

F_mean = np.array(res[res['label']=='FAKE'].iloc[:,1:-1]).mean(0)

F_sd = np.array(res[res['label']=='FAKE'].iloc[:,1:-1]).std(0)

R_mean = np.array(res[res['label']=='REAL'].iloc[:,1:-1]).mean(0)

R_sd = np.array(res[res['label']=='REAL'].iloc[:,1:-1]).std(0)



x = np.arange(0, 300, 1)

fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(x, F_mean, alpha=0.5, color='red', label='fake', linewidth =2.0)

ax.fill_between(x, F_mean - F_sd, F_mean + F_sd, color='red', alpha=0.2)



ax.plot(x, R_mean, alpha=0.5, color='blue', label='real', linewidth = 2.0)

ax.fill_between(x, R_mean - R_sd, R_mean + R_sd, color='blue', alpha=0.2)



ax.set_title('1D Power Spectrum for real/fake faces taken from the sample videos ',size=20)

plt.xlabel('Spatial Frequency', fontsize=20)

plt.ylabel('Power Spectrum', fontsize=20)

plt.tick_params(axis='x', labelsize=20)

plt.tick_params(axis='y', labelsize=20)

ax.legend(loc='best', prop={'size': 20})

plt.show()
def ROI(img):

    

    offset = 50

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=5) 

    

    for (x,y,w,h) in face_rects: 

        roi = face_img[y-offset:y+h+offset,x-offset:x+w+offset] 

           

    return roi

def azimuthalAverage(image, center=None):

    """

    Calculate the azimuthally averaged radial profile.

    image - The 2D image

    center - The [x,y] pixel coordinates used as the center. The default is 

             None, which then uses the center of the image (including 

             fracitonal pixels).

    

    """

    

    # Calculate the indices from the image

    y, x = np.indices(image.shape)



    if not center:

        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])



    r = np.hypot(x - center[0], y - center[1])



    # Get sorted radii

    ind = np.argsort(r.flat)

    r_sorted = r.flat[ind]

    i_sorted = image.flat[ind]



    # Get the integer part of the radii (bin size = 1)

    r_int = r_sorted.astype(int)



    # Find all pixels that fall within each radial bin.

    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented

    rind = np.where(deltar)[0]       # location of changed radius

    nr = rind[1:] - rind[:-1]        # number of radius bin

    

    # Cumulative sum to figure out sums for each radius bin

    csim = np.cumsum(i_sorted, dtype=float)

    tbin = csim[rind[1:]] - csim[rind[:-1]]



    radial_prof = tbin / nr



    return radial_prof
#train_filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')

test_filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

#json = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.json')

#metadata = pd.read_json(json[0])



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



with open("/kaggle/input/pre-trained-balanced/logreg_model_bal.sav", 'rb') as file:

   lr_classifier = pickle.load(file)



# Number of frames to sample (evenly spaced) from each video

n_frames = 10

submission = []

X = []

#Y = []
for i, filename in enumerate(test_filenames): #change to train_filenames for training

    try:

        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')



        # Create video reader and find length

        v_cap = cv2.VideoCapture(filename)

        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))



        # Pick 'n_frames' evenly spaced frames to sample

        sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)

        faces = []



        for j in range(v_len):

            success, vframe = v_cap.read()

            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            

            if j in sample:

                try: 

                    face = ROI(vframe)

                    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

                    faces.append(face)

                except: 

                    pass 

        v_cap.release()



        prob = []

        for face in faces:

            f = np.fft.fft2(face) #Create the FFT

            fshift = np.fft.fftshift(f) #Shift the FFT

            magnitude_spectrum = 20*np.log(np.abs(fshift)) #Magnitude spectrum

            psd1D = azimuthalAverage(magnitude_spectrum)



        # Interpolation

            points = np.linspace(0,300,num=psd1D.size) 

            xi = np.linspace(0,300,num=300) 

            interpolated = griddata(points,psd1D,xi,method='cubic')



            # Normalization

            interpolated /= interpolated[0]

            X.append(interpolated)

            #Y.append(metadata.at['label',os.path.basename(filename)]) #training labels

            prob.append(lr_classifier.predict_proba(interpolated.reshape(1,-1))[:,0][0])

        

        submission.append([os.path.basename(filename), sum(prob)/len(prob)]) #average probs over all images

            

    except:

        submission.append([os.path.basename(filename), 0.5]) #sometimes bad things can happen, if that occurs just guess 0.5!

        #pass

submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)
'''

#This segment is used on the trainings set. Cross validation to estimate accuracy then training set.

LR = 0

for i in range(5):



    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(solver='liblinear', max_iter=1000, class_weight = 'balanced')

    logreg.fit(X_train, y_train)

    LR+=logreg.score(X_test,y_test)

    prob = logreg.predict_proba(np.array(X_test[0]).reshape(1,-1))[:,0]



model = logreg.fit(X,Y)

filename = '/kaggle/working/logreg_model_bal.sav'

pickle.dump(model, open(filename, 'wb'))

print(F'Accuracy is {round(LR/5,2)}')

'''
