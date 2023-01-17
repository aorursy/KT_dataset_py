import numpy as np

import pickle

# 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 파일로 저장하는 것 

# 10GB 짜리 raw text 파일을 파싱하여 필요한 부분을 빼내서 사용하는 것과 같다. 빠르다.



import cv2



from os import listdir

# 특정 폴더에 있는 특정 파일 리스트를 가져와서 사용해야하는 경우 read_csv와 비슷한 역할.



from sklearn.preprocessing import LabelBinarizer

# 텍스트 범주에서 숫자형 범주로, 숫자형 범주에서 원핫인코딩으로



from keras.models import Sequential

# 케라스 모델에서의 선형 모델 , Sequential 모델은 레이어를 선형으로 연결하여 구성

# 케라스 모델링의 예 

# model = Sequential([

#     Dense(32, input_shape=(784,)),  from keras.layers(Dense)  기존의 머신러닝의 하이퍼파라미터 대신에 레이어가 들어가 있다고 생각하면 될듯?

#     Activation('relu'),             from keras.layers(Activation)

#     Dense(10),                      from keras.layers(Dense)

#     Activation('softmax'),          from keras.layers(Activation)

# ])



from keras.layers.normalization import BatchNormalization

# normalization 을 위와 같이 설명하였다. mini-batch 의 평균과, 분산을 이용해서 

# normalize 후, scale and shift 를 γ, β 를 통해 실행한다. 이 때 γ와 β 는 학습 가능한 변수이다.

# 이렇게 normalization이 된 값을 activation function 에 입력으로 사용하고 최종 출력물을 다음 층에 입력으로 사용하는 것이다.

#                           여기서 activation은 레이어의 하이퍼파라미터 형태와 비슷하다.



from keras.layers.convolutional import Conv2D,MaxPooling2D

# 레이어 기법의 종류, 각 레이어마다 특징과 역할이 조금씩 다르다.



from keras.layers.core import Activation, Flatten, Dropout, Dense

# activation : 활성화 함수 설정합니다.

# ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.

# ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.

# ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.

# ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

#  https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/

# Dense 레이어는 입력과 출력을 모두 연결해줍니다. 예를 들어 입력 뉴런이 4개, 

# 출력 뉴런이 8개있다면 총 연결선은 32개(4*8=32) 입니다. 각 연결선에는 가중치(weight)를 포함하고 있는데, 

# 이 가중치가 나타내는 의미는 연결강도라고 보시면 됩니다. 현재 연결선이 32개이므로 가중치도 32개입니다.

# 가중치가 높을수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 적다.



from keras import backend as K

#  케라스는 거의 모든 종류의 딥러닝 모델을 간편하게 만들고 훈련시킬 수 있는 파이썬을 위한 딥러닝 프레임워크

#  딥러닝 모델을 만들기 위한 고수준의 구성 요소를 제공하는 모델 수준의 라이브러리

#  백엔드 엔진backend engine에서 제공하는 최적화되고 특화된 텐서 라이브러리를 사용

#  백엔드, 사용자 눈에 보이지 않는 뒤에서 이루어지는 작업 , 서버나 클라이언트 작업



from keras.preprocessing.image import ImageDataGenerator

# 데이터를 이리저리 변형시켜서 새로운 학습 데이터를 만들어줍니다. 변형의 예시는 회전, 이동 등등 매우 다양

# https://neurowhai.tistory.com/158



from keras.optimizers import Adam

# 경사 하강법의 종류  최적의 파라미터를 찾고 학습률을 높게하기 위한 기능

# https://twinw.tistory.com/247



from keras.preprocessing import image

# 케라스방식의 사진 불러오기기능



from keras.preprocessing.image import img_to_array

# 케라스에서 이미지어레이



from sklearn.preprocessing import MultiLabelBinarizer

# 이터 러블의 이터 러블과 멀티 라벨 형식 간 변환



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
EPOCHS = 25                                      # 학습횟수 지정

INIT_LR = 1e-3                                   # 러닝레이트, 학습깊이,단위

BS = 32                                          # ?      

default_image_size = tuple((256, 256))           # 디포트 이미지 사이즈

image_size = 0                                   # 이미지사이즈 숫자데이터0

directory_root = '../input/plantvillage/'        # 파일 경로

width=256                                        # 가로

height=256                                       # 높이

depth=3                                          # 깊이
def convert_image_to_array(image_dir):   # 함수를 만들겠다. 이미지파일디렉토리 어레이만큼

    try:                                 # try문 성공시 이루어질 작업

        image = cv2.imread(image_dir)   # 이미지는 이미지디렉토리 파일을 cv2방식으로 불어와라 

        if image is not None :          # 이미지가 널이 아니라면

            image = cv2.resize(image, default_image_size)   #이미지는 cv2방식으로 resize해라(이미지(0),디포트 이미지 싸이즈) (위에서 지정해준 변수)

            return img_to_array(image)                      # 이미지의 어레이 반환하여라

        else :                          # 그게 아니라면

            return np.array([])         # 어레이는 빈 리스트가 된다/

    except Exception as e:              # 오류가 떳을때

        print(f"Error : {e}")           # 이문구를 출력해라

        return None                     # 널값을 반환해라
image_list, label_list = [], []                  # 빈 리스트 2개를 만든다.

try:                                             # 트라이문 진행

    print("[INFO] Loading images ...")           # 문구를 출력해라

    root_dir = listdir(directory_root)           # 변수지정 listdir방식으로 디렉토리파일을 불러와라  

    for directory in root_dir :                  # 반복문을 실행해라 root_dir 크기만큼?

        # remove .DS_Store from list 

        if directory == ".DS_Store" :           # 만약 디렉토리가 ".DS_Store"라면

            root_dir.remove(directory)          # listdir방식으로 불러온 root_dir를 삭제해라 



    for plant_folder in root_dir :              # 반복문을 실행해라, root_dir크기만큼?

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")  # 변수에  리스트dir에 저장해라(f"{directory_root}/{plant_folder}"(형식코드)) >>이러한 형식으로

        

        for disease_folder in plant_disease_folder_list :                   # 반복문안에서 또 반복문을 돌린다 위에 반복문에서 만들어진 plant_disease_folder_list에 대한

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :                              # 만약 disease_folder디렉토리가 ".DS_Store"라면

                plant_disease_folder_list.remove(disease_folder)            # 삭제해라



        for plant_disease_folder in plant_disease_folder_list:              # 또 반복문을 돌린다. 맨위 반복문에서 만들어진  plant_disease_folder_list에 대한,

            print(f"[INFO] Processing {plant_disease_folder} ...")          # 돌리는 동안 문구 출력

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/") #변수 생성 후 listdir방식으로 불러온다 형식은 괄호안에 형식으로

                

            for single_plant_disease_image in plant_disease_image_list :   # 반복문 안에 또 반복문을 돌린다.맨위 반복문에서 만들어진 plant_disease_image_list에 대한.

                if single_plant_disease_image == ".DS_Store" :             # 만약 single_plant_disease_image가 ".DS_Store"라면

                    plant_disease_image_list.remove(single_plant_disease_image) # 또 삭제해라



            for image in plant_disease_image_list[:200]:                    # 또 반복문을돌린다. 맨위 반복문에서 만들어진 plant_disease_image_list[조건]에 대한.

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True: #만약image_directory.endswith(".jpg"),image_directory.endswith(".JPG") 형식이 맞다면

                    image_list.append(convert_image_to_array(image_directory))                             # 리스트에 추가해라

                    label_list.append(plant_disease_folder)                                                 # 리스트에 추가해라.

    print("[INFO] Image loading completed")                        #완료문구출력

except Exception as e:                                            #오류가 나왔다면

    print(f"Error : {e}")                                          #문구를 출력해라
image_size = len(image_list)  # 이미지 사이즈 개수 출력
label_binarizer = LabelBinarizer()          ## 텍스트 범주에서 숫자형 범주로, 숫자형 범주에서 원핫인코딩으로

image_labels = label_binarizer.fit_transform(label_list) # LabelBinarizer() fit해준다.

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb')) # 파이썬 객체를 파일에 저장하는 과정, 피클링(pickling) ()LabelBinarizer() 에 맞게 객체파일을 열어라.

n_classes = len(label_binarizer.classes_)  # 개수 출력
print(label_binarizer.classes_)  # 라벨 확인 
np_image_list = np.array(image_list, dtype=np.float16) / 225.0  # ?
print("[INFO] Spliting data to train, test")  # 나누는 문구 출력

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(      # # 데이터를 이리저리 변형시켜서 새로운 학습 데이터를 만들어줍니다. 변형의 예시는 회전, 이동 등등 매우 다양# https://neurowhai.tistory.com/158

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential()    # 모델링

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))   # 모델에 레이어추가

model.add(Activation("relu"))                                          # 모델 레이어에 액티베이션 추가.

model.add(BatchNormalization(axis=chanDim))                            # 모델 레이어에 BatchNormalization 해준다.

model.add(MaxPooling2D(pool_size=(3, 3)))                              # 모델에 레이어추가

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))                          # 모델에 레이어추가

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation("softmax"))



# activation : 활성화 함수 설정합니다.

# ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.

# ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.

# ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.

# ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

#  https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/
model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# distribution

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network

print("[INFO] training network...")



# 모델의 정확도 지표



# 학습률 지정 , 기존의 모델의 eport,running_rate와 비슷한 역할로써 여기서는 옵티마이저라는 새로운 기능으로 쓰인다.



# https://forensics.tistory.com/28
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1

    )

# 모델 학습.

# __data_generation는 generation process에서 core한 역할인 데이터의 batch를 생성함

# data generation동안에 이 코드는 ID.npy에 상응하는 example를 NumPy 배열로 만들어냄

# 코드가 multicore friendly 하기 때문에 차후에 더 복잡한 연산도 가능하다.(예: source 파일로 부터 계산)
acc = history.history['acc']                      # 정확도

val_acc = history.history['val_acc']              # 정확도 검증

loss = history.history['loss']                    # 오차율

val_loss = history.history['val_loss']            # 오차율 검증

epochs = range(1, len(acc) + 1)                   # 학습횧수

#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')  #그래프로

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
print("[INFO] Calculating model accuracy")  # 문구 출력 

scores = model.evaluate(x_test, y_test)   # 테스트데이터 오차율,정확도

print(f"Test Accuracy: {scores[1]*100}")
# save the model to disk

!pip install coremltools



#  coremltools = 훈련 된 모델을 널리 사용되는 기계 학습 도구에서 Core ML 형식 (.mlmodel)으로 변환합니다.

# 간단한 API로 모델을 Core ML 형식으로 작성하십시오.

# Core ML 프레임 워크 (일부 플랫폼)를 사용하여 예측하여 전환을 확인합니다.

import coremltools



output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



coreml_model = coremltools.converters.keras.convert(model,

input_names='image',

image_input_names='image',

output_names='output')



coreml_model.author = 'Utkarsh Sharma and Prateek Sawhney'

coreml_model.short_description = 'Model to classify hand written digit'

coreml_model.input_description['image'] = 'Color image of plant leaf'

coreml_model.output_description['output'] = 'Disease Detection'



coreml_model.save('plantDiseaseModel.mlmodel')



print("[INFO] Saving model...")   # 문구 출력

pickle.dump(model,open('cnn_model.pkl', 'wb'))