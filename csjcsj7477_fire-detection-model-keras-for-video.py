import numpy as np

import cv2

%matplotlib inline

import matplotlib.pyplot as plt
tot_video_cnt = 0

width = 400

height = 240



ds_x = []

ds_y = [] 



ts_x = []

ts_y = [] 



class wildfire:



    #num_pt : 1 => fire, 0 => no_fire

    #cnt : frame cut count

    #FileName : Video file name 

    def generate_dataset(main,frCnt,num_pt,capture_cnt,FileName):

        

        cap_cnt = 0

        frame_cnt = 0 #프레임갯수

        cap = cv2.VideoCapture(FileName)

        mog = cv2.createBackgroundSubtractorMOG2()

        

        while True:

        

            if cap_cnt == capture_cnt:

                break

            print("filename:{}, cnt:{}".format(FileName,cap_cnt+1)) 

            

            frame_cnt += 1 # 프레임 갯수

            

            if (frame_cnt%frCnt) == 0:             

            

                ret, frame = cap.read()

                if ret == False:

                    break



                dst = cv2.fastNlMeansDenoisingColored(frame,None,25,25,7,21)    

                fgmask = mog.apply(dst)

                ds_x.append(fgmask)

                ds_y.append(num_pt)



                cap_cnt += 1



        cap.release()      



    #num_pt : 1 => fire, 0 => no_fire

    #cnt : frame cut count

    #FileName : Video file name 

    def test_dataset(main,frCnt,num_pt,capture_cnt,FileName):

        

        cap_cnt = 0

        frame_cnt = 0 #프레임갯수

        cap = cv2.VideoCapture(FileName)

        mog = cv2.createBackgroundSubtractorMOG2()

        

        while True:

        

            if cap_cnt == capture_cnt:

                break

            print("filename:{}, cnt:{}".format(FileName,cap_cnt+1)) 

            

            frame_cnt += 1 # 프레임 갯수

            

            if (frame_cnt%frCnt) == 0:             

            

                ret, frame = cap.read()

                if ret == False:

                    break



                dst = cv2.fastNlMeansDenoisingColored(frame,None,25,25,7,21)    

                fgmask = mog.apply(dst)

                ts_x.append(fgmask)

                ts_y.append(num_pt)



                cap_cnt += 1



        cap.release()   
wf = wildfire()

video_cnt = 10

wf.generate_dataset(1,20,video_cnt,'dalma_400240.mp4') # param : 메인변수, 건너뛸 프레임갯수, 캡처화면수, video명

tot_video_cnt += video_cnt
video_cnt = 30

wf.generate_dataset(1,80,video_cnt,'gwanak_400240.mp4') # param : 메인변수, 건너뛸 프레임갯수, 캡처화면수, video명

tot_video_cnt += video_cnt
X_train = np.array(ds_x).reshape(tot_video_cnt,width,height,1)

y_train = np.array(ds_y).reshape(tot_video_cnt, 1)
X_train.shape
X_val = np.array(ds_x).reshape(tot_video_cnt,width,height,1)

y_val = np.array(ds_y).reshape(tot_video_cnt, 1)
video_cnt1 = 10

wf.test_dataset(1,20,video_cnt1,'no_fire_1.mp4') # param : 메인변수, 건너뛸 프레임갯수, 캡처화면수, video명
X_test = np.array(ts_x).reshape(video_cnt1,width,height,1)

y_test = np.array(ts_y).reshape(video_cnt1, 1)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout



#2. 모델1 구성하기

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# 180/180 [==============================] - 17s - loss: 0.7322 - acc: 0.9556  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)

print(score)