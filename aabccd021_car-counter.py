import cv2

import os



# List semua gambar pada dataset

image_files = os.listdir('./input/Image-Traffic/')



# Gambar pertama

image1 = cv2.imread('./input/Image-Traffic/00020 0001.jpg')



# Gambar pertama dengan nilai 1/50

# 1/100 + 1/100 = 1/50

image1 = cv2.addWeighted(image1, 1/100, image1, 1/100, 0)



# List gambar average dari setiap batch

batch_avg = []



# Membuat gambar average dari setiap batch

for index, image_file in enumerate(image_files):

    if index % 50==0:

        batch_avg.append(image1)

        image1 = cv2.imread('./input/Image-Traffic/'+image_file)

        image1 = cv2.addWeighted(image1, 1/100, image1, 1/100, 0)

    image = cv2.imread('./input/Image-Traffic/'+image_file)

    image1 = cv2.addWeighted(image1, 1, image, 1/50, 0)
# Membuat gambar average dari setiap gambar average batch

batches_len = len(batch_avg)

result = cv2.addWeighted(batch_avg[0], 1/(2*batches_len), batch_avg[0], 1/(2*batches_len), 0)

batch_avg.pop(0)



for batch_now in batch_avg:

    result = cv2.addWeighted(result, 1, batch_now, 1/batches_len, 0)

    

# menuliskan output

cv2.imwrite('./output/Mean_Average_BG.png', result)
import cv2

import os

import imutils

import numpy as np

import pandas as pd

from skimage.measure import compare_ssim

from random import randint



# preprocessing gambar background menjadi grayscale

bg = cv2.imread('./output/Mean_Average_BG.png')

bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)



# list untuk diisi data-data

# setiap data adalah data yang terdeteksi beda pada gambar dengan background

# setiap data adalah tuple seperti berikut (x, y, w, h, score)

# x, y adalah titik kiri bawah dari kotak. w, h adalah lebar dan tinggi kotak

rows = []



BANYAK_GAMBAR_TRAIN = 40



# list semua gambar

image_files = os.listdir('./input/Image-Traffic/')



for index, image_file in enumerate(image_files):

    #membuat gambar untuk labeling dengan jumlah yang ditentukan, pada kode ini digunakan 50

    if index > BANYAK_GAMBAR_TRAIN:

        break

    

    # preprocessing gambar menjadi grayscale

    image_now_color = cv2.imread('./input/Image-Traffic/'+image_file)

    image_now = cv2.cvtColor(image_now_color, cv2.COLOR_BGR2GRAY)

    

    # menghitung score selisih gambar dengan background

    # semakin besar score, semakin sedikit foreground

    (score, diff) = compare_ssim(bg, image_now, full=True)

    diff = (diff * 255).astype("uint8")



    # angka 9 ini adalah magic number yang saya temukan dapat mendeteksi mobil dengan ukuran yang baik

    # dapat dikatakan saya mendapat angka 9 ini dari human training

    diffr = np.add(diff, 9)

    thresh = cv2.threshold(diffr, 0, 255, 

                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)



    

    detect_counter = 0

    for c in cnts:

        

        # mendapatkan x, y, w, h dari setiap kontur

        (x, y, w, h) = cv2.boundingRect(c)

        

        # hanya menampilkan kontur dengan constraint tertentu, untuk mempermudah labeling

        # angka 30, 500, dan 900 disini saya tentukan sendiri dengan asumsi tidak ada gambar mobil

        # yang tidak memenuhi constrain tersebut

        if w>30 and h>30 and h<900 and w<500:

            

            detect_counter+=1

            

            # nama kontur

            name = str(index)+'#'+str(detect_counter)

            

            # warna kotak dan tulisan untuk setiap kontur dirandom untuk mempermudah labeling

            color = (randint(100, 255),randint(100, 255),randint(100, 255))

            cv2.rectangle(image_now_color, (x, y), (x + w, y + h), color, 2)

            cv2.putText(image_now_color, name, (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

            

            # mencari score ssim untuk setiap kontur

            bg_crop = bg[y:y+h, x:x+w]

            image_now_crop = image_now[y:y+h, x:x+w]

            (score_crop, _) = compare_ssim(bg_crop, image_now_crop, full=True)

            

            

            # menyimpan data setiap kontur

            # 0 disini adalah default value, artinya tidak terdeteksi mobil

            rows.append((name,0,x,y,w,h,score_crop))

            

    # menyimpan gambar

    cv2.imwrite('./output/Training/'+str(index)+'_training.png', image_now_color)



df = pd.DataFrame(rows)

df.to_csv('./output/training.csv')
import pandas as pd

names = ['index','name','car_count','x','y','w','h','score']

train_df = pd.read_csv('./input/train_labeled.csv', names=names)



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import numpy as np



X = train_df[['x','y','w','h','score']]

X['area']= np.multiply(X['w'],X['h'])

X['aspect_ratio']= np.divide(X['w'],X['h'])

y = train_df['car_count']

X.head()



import cv2

import os

import numpy as np

from skimage.measure import compare_ssim

import imutils

from random import randint

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



clf = RandomForestRegressor()

clf.fit(X,Y)



bg = cv2.imread('./output/Mean_Average_BG.png')

bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)



LIMIT = 40



a = [str(x).zfill(3) for x in range(52,400)]

for index, image_file in enumerate(a):

    if index>LIMIT:

        break

    

    # preprocessing gambar menjadi grayscale

    image_now_color = cv2.imread('./input/Image-Traffic/00020 0'+image_file+'.jpg')

    image_now = cv2.cvtColor(image_now_color, cv2.COLOR_BGR2GRAY)

    

    # menghitung score selisih gambar dengan background

    # semakin besar score, semakin sedikit foreground

    (score, diff) = compare_ssim(bg, image_now, full=True)

    diff = (diff * 255).astype("uint8")



    # angka 9 ini adalah magic number yang saya temukan dapat mendeteksi mobil dengan ukuran yang baik

    # dapat dikatakan saya mendapat angka 9 ini dari human training

    diffr = np.add(diff, 9)

    thresh = cv2.threshold(diffr, 0, 255, 

                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)



    

    car_count = 0

    for c in cnts:

        # mendapatkan x, y, w, h dari setiap kontur

        (x, y, w, h) = cv2.boundingRect(c)

        

        # hanya menampilkan kontur dengan constraint tertentu, untuk mempermudah labeling

        # angka 30, 500, dan 900 disini saya tentukan sendiri dengan asumsi tidak ada gambar mobil

        # yang tidak memenuhi constrain tersebut

        if w>30 and h>30 and h<500 and w<500:

            

            # mencari score ssim untuk setiap kontur

            bg_crop = bg[y:y+h, x:x+w]

            image_now_crop = image_now[y:y+h, x:x+w]

            (score_crop, _) = compare_ssim(bg_crop, image_now_crop, full=True)

            data = pd.DataFrame([(x,y,w,h,score_crop,w*h,w/h)])

            car_count_predict = int(round(clf.predict(data)[0]))

            if car_count_predict>=1:

                color = (0,0,255)

                # menampilkan hasil deteksi dan prediksi jumlah mobil

                cv2.rectangle(image_now_color, (x, y), (x + w, y + h), color, 3)

                cv2.putText(image_now_color, str(car_count_predict), (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

                car_count += car_count_predict

                

    cv2.putText(image_now_color, 'car count:'+str(car_count), (0, 0+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), thickness=3)

    cv2.imwrite('./output/Prediction/prediction_'+image_file+'.png',image_now_color)

#     cv2.imshow('prediction result', image_now_color)

#     cv2.waitKey(1)

            

            

cv2.destroyAllWindows()
import cv2

import os

import imutils

import numpy as np

import pandas as pd

import time

from random import randint

from skimage.measure import compare_ssim

from random import randint

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor

from sklearn.svm import SVR



labeled = pd.read_csv('./input/labeled.csv',header=None)

labeled.head()



names = ['index','name','car_count','x','y','w','h','score']

train_df = pd.read_csv('./input/train_labeled.csv', names=names)

train_df.head()



X = train_df[['x','y','w','h','score']]

X['area']= np.multiply(X['w'],X['h'])

X['aspect_ratio']= np.divide(X['w'],X['h'])

Y = train_df['car_count']



bg = cv2.imread('./output/Mean_Average_BG.png')

bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)



models = {

    'Linear Regression':LinearRegression(),

    'Ridge':Ridge(),

    'Lasso':Lasso(),

    'ElasticNet':ElasticNet(),

    'Random Forest Regressor':RandomForestRegressor(),

    'Ada Boost Regressor':AdaBoostRegressor(),

    'Extra Tree Regressor':ExtraTreesRegressor(),

    'SVR':SVR(),

}



model_errors = []

rand = randint(1,300)

for name, model in models.items():

    start = time.time()

    model.fit(X,Y)

    end = time.time()

    train_time = (end-start)

    errors = []

    frame_times = []

    for index, image_file in enumerate(os.listdir('./input/Image-Traffic/')):

        if index>rand+50:

            break

        if index<rand:

            continue

        start = time.time()

        # preprocessing gambar menjadi grayscale

        image_now_color = cv2.imread('./input/Image-Traffic/'+image_file)

        image_now = cv2.cvtColor(image_now_color, cv2.COLOR_BGR2GRAY)



        # menghitung score selisih gambar dengan background

        # semakin besar score, semakin sedikit foreground

        (score, diff) = compare_ssim(bg, image_now, full=True)

        diff = (diff * 255).astype("uint8")



        # angka 9 ini adalah magic number yang saya temukan dapat mendeteksi mobil dengan ukuran yang baik

        # dapat dikatakan saya mendapat angka 9 ini dari human training

        diffr = np.add(diff, 9)

        thresh = cv2.threshold(diffr, 0, 255, 

                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)





        car_count = 0

        for c in cnts:

            # mendapatkan x, y, w, h dari setiap kontur

            (x, y, w, h) = cv2.boundingRect(c)



            # hanya menampilkan kontur dengan constraint tertentu, untuk mempermudah labeling

            # angka 30, 500, dan 900 disini saya tentukan sendiri dengan asumsi tidak ada gambar mobil

            # yang tidak memenuhi constrain tersebut

            if w>30 and h>30 and h<900 and w<500:



                # mencari score ssim untuk setiap kontur

                bg_crop = bg[y:y+h, x:x+w]

                image_now_crop = image_now[y:y+h, x:x+w]

                (score_crop, _) = compare_ssim(bg_crop, image_now_crop, full=True)

                data = pd.DataFrame([(x,y,w,h,score_crop,w*h,w/h)])

                car_count_predict = (model.predict(data)[0])

                

                if car_count_predict>0:

                    car_count += car_count_predict

                    

        end = time.time()

        frame_time = (end-start)

                    

        yang_bener = labeled[labeled[0]==image_file].iloc[0,1]

        error = int(round(car_count))-yang_bener

        errors.append((error,))

        frame_times.append((frame_time,))



    errors = pd.DataFrame(errors)

    frame_time_avg = pd.DataFrame(frame_times)[0].mean()

    rmse = np.sqrt(errors[0].pow(2).mean())

    akurasi = len(errors[errors[0]==0])/len(errors)

    mae = errors[0].abs().mean()

    model_errors.append((name, rmse, mae, akurasi, train_time, frame_time_avg))

    

model_errors = pd.DataFrame(model_errors, columns=['Model Name','RMSE','MAE','Akurasi','Train Time','Frame Time Average'])

model_errors
from sklearn.model_selection import RandomizedSearchCV



names = ['index','name','car_count','x','y','w','h','score']

train_df = pd.read_csv('./input/train_labeled.csv', names=names)

train_df.head()



X = train_df[['x','y','w','h','score']]

X['area']= np.multiply(X['w'],X['h'])

X['aspect_ratio']= np.divide(X['w'],X['h'])

Y = train_df['car_count']



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



def evaluate(model):

    model.fit(X,Y)

    errors = []

    for index, image_file in enumerate(os.listdir('./input/Image-Traffic/')):

        if index>rand+30:

            break

        if index<rand:

            continue

        # preprocessing gambar menjadi grayscale

        image_now_color = cv2.imread('./input/Image-Traffic/'+image_file)

        image_now = cv2.cvtColor(image_now_color, cv2.COLOR_BGR2GRAY)



        # menghitung score selisih gambar dengan background

        # semakin besar score, semakin sedikit foreground

        (score, diff) = compare_ssim(bg, image_now, full=True)

        diff = (diff * 255).astype("uint8")



        # angka 9 ini adalah magic number yang saya temukan dapat mendeteksi mobil dengan ukuran yang baik

        # dapat dikatakan saya mendapat angka 9 ini dari human training

        diffr = np.add(diff, 9)

        thresh = cv2.threshold(diffr, 0, 255, 

                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)





        car_count = 0

        for c in cnts:

            # mendapatkan x, y, w, h dari setiap kontur

            (x, y, w, h) = cv2.boundingRect(c)



            # hanya menampilkan kontur dengan constraint tertentu, untuk mempermudah labeling

            # angka 30, 500, dan 900 disini saya tentukan sendiri dengan asumsi tidak ada gambar mobil

            # yang tidak memenuhi constrain tersebut

            if w>30 and h>30 and h<900 and w<500:



                # mencari score ssim untuk setiap kontur

                bg_crop = bg[y:y+h, x:x+w]

                image_now_crop = image_now[y:y+h, x:x+w]

                (score_crop, _) = compare_ssim(bg_crop, image_now_crop, full=True)

                data = pd.DataFrame([(x,y,w,h,score_crop,w*h,w/h)])

                car_count_predict = (model.predict(data)[0])

                if car_count_predict>0:



                    color = (0,0,255)

                    car_count += car_count_predict

                    

        yang_bener = labeled[labeled[0]==image_file].iloc[0,1]

        error = int(round(car_count))-yang_bener

        errors.append((error,))



    errors = pd.DataFrame(errors)

    rmse = np.sqrt(errors[0].pow(2).mean())

    akurasi = len(errors[errors[0]==0])/len(errors)

    mae = errors[0].abs().mean()

    return (rmse, mae, akurasi)



base_model =RandomForestRegressor()

base_model.fit(X,Y)

result = evaluate(base_model)



print("Base Random Forest Regressor")

print("RMSE:",result[0])

print("MAE:",result[1])

print("Akurasi:",result[2])

print()



rf_random = RandomizedSearchCV(

    estimator=RandomForestRegressor(),

    param_distributions=random_grid,

    n_iter = 100,

    cv = 3,

    verbose = 2,

    random_state = 42,

    n_jobs = -1

)



rf_random.fit(X,Y)

best_random = rf_random.best_estimator_

result = evaluate(base_model)

print("Bast Random Forest Regressor")

print("RMSE:",result[0])

print("MAE:",result[1])

print("Akurasi:",result[2])

print()



print(rf_random.best_params_)