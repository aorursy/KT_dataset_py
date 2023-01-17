from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

from tensorflow.keras.models import load_model
img_path = '../input/thai-mnist-classification/train/'

test_df = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')

test_df = test_df[:200]

test_df['img_path'] = img_path + test_df['id']

model = load_model('../input/thaimnist/thai-mnist2.h5')

test_df.category = test_df.category.astype('str')

#try to predict

data_gen = ImageDataGenerator()

test = data_gen.flow_from_dataframe(test_df,x_col='img_path',y_col='category',class_mode='sparse',shuffle=False,target_size=(224, 224))

pred = model.predict_generator(test)

pred_list = pred.argmax(axis=1)
from sklearn.metrics import accuracy_score

print("show accuracy score " , accuracy_score(test.classes,pred_list))

#np.sum(pred.argmax(axis=1) == test.classes)/len(test.classes)
def predict_one(img_path,display=False,test=False):

    path = '../input/thai-mnist-classification/train/'

    if test:

        path = '../input/thai-mnist-classification/test/'

    if img_path != 'nan':

        #print("process image ", path + img_path)

        img = tf.keras.preprocessing.image.load_img(path + img_path, target_size=(224, 224))

        img_array = tf.keras.preprocessing.image.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        result = np.argmax(score)

        if display:

            plt.title(result)

            plt.imshow(img)

        return result

#try to predict 

predict_one('00525ebe-79fc-4424-bd6e-b966c8b6ab0e.png',True)

test_df
test_df.loc[0]['img_path']
f_test = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')

f_test = f_test[:200]
f_test
''' เลข 5 ตัวเล็ก '''

''' เลข 7 ตัวเล็ก '''

''' เลข 4 เข้มเป็น 5 '''

''' เลข 8 ตัวเล็ก '''

''' ตัวอะไรไม่รู้ ทายยาก'''

wrong_img = ['12f76bd6-dee2-4385-85a0-72d8048d9667.png',

             'ebfaaa31-59dc-40d3-80d3-bc0bd0419464.png',

             '7b56183c-01ac-4b84-9aa8-379856869996.png',

             'eeedf24f-c907-4134-982f-0e022890df9f.png',

             '12f76bd6-dee2-4385-85a0-72d8048d9667.png',

             '0b83d49b-436f-4580-9a32-2afb3595bbcc.png'] 
def show(list_img):

    c = 0

    fig, ax = plt.subplots(3,2, figsize=(15,15))

    for i in range(3):

        for j in range(2):

            test_path = '../input/thai-mnist-classification/test/'

            print(test_path + list_img[c])

            img = cv2.imread(test_path +list_img[c])

            ax[i][j].imshow(img)

            c += 1
#predict muti by subplot 

fig, ax = plt.subplots(6,6, figsize=(15,15))

for i in range(6):

    for j in range(6):

        img_index = np.random.randint(178,200)

        img_path = f_test.loc[img_index]['feature2']

        print(img_index,i,j,'path', img_path)

        img = cv2.imread('../input/thai-mnist-classification/test/' + img_path)

        predict = predict_one(img_path,display=False,test=True)

        ax[i][j].imshow(img)

        ax[i][j].set_title(predict)

        ax[i][j].set_axis_off()
f_train = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')

f_test = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')

f_submit = pd.read_csv('../input/thai-mnist-classification/submit.csv')
predict_one('1b2e1b3d-46f2-4c3f-b741-a8d5099ce710.png',True)
print(len(f_train))

f_train.head()

print(len(f_test))

f_test.head()
print(len(f_submit))

f_submit.head()
from tqdm import tqdm

tqdm.pandas()

#ten_fold = [f_test[f_test.index / i] for i in range(10)]
batch_size = 2000

ten_fold = []

for i in range(10):

    ten_fold.append(f_test[i*batch_size:batch_size * (i+1)])

    

    
ten_fold[9]
print("all data test =* ",len(f_test))

print("=* ",len(ten_fold)," fold")

print('=* each of fold', len(ten_fold[0]))
def convert_df_predict(df):

    df.feature1 = df.feature1.astype('str')

    df.feature1 = df.progress_apply(lambda x: predict_one(x.feature1,test=True),axis=1)

    df.feature2 = df.progress_apply(lambda x: predict_one(x.feature2,test=True),axis=1)

    df.feature3 = df.progress_apply(lambda x: predict_one(x.feature3,test=True),axis=1)

    return df

def case_answer(f1,f2,f3):

    #print("==* f1", f1)

    ans = 0

    f1 = float(f1)

    if f1 == 'nan' or 'NaN':

        ans =  f2 + f3

    if f1 == 0:

        ans = f2 * f3

    if f1 == 1:

        ans = abs(f2-f3)

    if f1 == 2:

        ans = (f2+f3) * abs(f2-f3)

    if f1 == 3:

        q = (((f2^2)+1)*f2) + f3*(f3+1)

        if q > 99:

            q = (q % 99)

        ans = q

    if f1 == 4:

        ans = 50+(f2-f3)

    if f1 == 5:

        ans = min(f2,f3)

    if f1 == 6:

        ans = max(f2,f3)

    if f1 == 7:

        ans = ((f2*f3)%9)*11

    if f1 == 8:

        q = f3*(f3 +1) - f2*(f2-1)

        ans = abs(q/2)

    if f1 == 9:

        ans = 50 + f2

    return ans



def predict_answer(df):

    predict_dic = []

    index_list = []

    for index, row in df.iterrows():

        predict_dic.append(case_answer(row['feature1'],row['feature2'],row['feature3']))

        index_list.append(index)

        

    df.predict = pd.DataFrame(predict_dic,index=index_list)

    return df.predict

sample_fold = ten_fold[9]
sample_fold[:200]
fold_ans = []

count = 1

for fold in ten_fold:

    print("procress fold :",count)

    fold = convert_df_predict(fold)

    fold.feature1 = fold.feature1.astype('str')

    fold.predict = predict_answer(fold)

    fold_ans.append(fold)

    count += 1



print("end procress fold..")

print("Concat !!")

fold_concat = pd.concat(fold_ans)
fold_concat
fold_concat
submit_fold = fold_concat
submit_fold
submit_ans = submit_fold.drop(columns=['feature1','feature2','feature3'])
submit_ans
submit_ans.to_csv('submit.csv',index=False)