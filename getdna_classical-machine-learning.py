# Setup Code

from IPython.display import display

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
Data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
# Print top 5 rows of Data

print("This is 'Data'")

display(Data.head())



# Feature Extraction

DataX = Data[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]



# Display Table with Extracted Feature

print("This is Data with Extracted Feature")

display(DataX.head())
import cv2

Images = []

for x in range(1,4):

    Image = cv2.imread(f"../input/covid19-image-dataset/Covid19-dataset/train/Covid/0{x}.jpeg")

    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)

    Image = cv2.resize(Image, (224, 224))

    Images.append(Image)

Images = np.array(Images) / 255.0
from tensorflow.keras.applications import VGG16



# Show one of the Images

plt.imshow(Images[0])



# Initialize VGG16 Model

# Note: VGG16 is the model that extracts feature from the image

FE = VGG16()



# Get the feature

DataX = FE.predict(Images)



display(DataX)
Data = pd.read_csv('../input/home-data-for-ml-course/train.csv')



DataX = Data[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]



DataY = Data['SalePrice']
from sklearn.model_selection import train_test_split



TrainX, TrainY, ValX, ValY = train_test_split(DataX, DataY, train_size = 0.9)



print("TrainX")

display(TrainX)



print("ValX")

display(ValX)



print("TrainY")

display(TrainY)



print("ValY")

display(ValY)



# กดปุ่ม Output เพื่อดูข้อมูลใน TrainX, ValX, TrainY, และ ValY
i = round(len(DataX) * 0.9)



TrainX = DataX[:i]

print("TrainX")

display(TrainX)



ValX = DataX[i:]

print("ValX")

display(ValX)



TrainY = DataY[:i]

print("TrainY")

display(TrainY)



ValY = DataY[i:]

print("ValY")

display(ValY)



# กดปุ่ม Output เพื่อดูข้อมูลใน TrainX, ValX, TrainY, และ ValY
from sklearn.tree import DecisionTreeRegressor



# สร้าง Model

SKLDecisionTreeRegModel = DecisionTreeRegressor(random_state=1);



# Fit Model (หรือ Train Model) โดยใช้ TrainX กับ TrainY

SKLDecisionTreeRegModel.fit(TrainX,TrainY)



# Predict Value โดยใช้ ValX ให้ได้ Prediction (Predict ค่า Y)

Prediction = SKLDecisionTreeRegModel.predict(ValX)
# สมมุติว่าทำ Model แล้วได้ค่าออกมาดังนี้

Prediction = [1,1,2,3,4,5,3,1,2,4,100] # 100 = Outlier



# ข้อมูล Validation ที่ใช้ในการเช็ค

ValY       = [2,1,2,4,2,3,4,3,2,3,  3]
from sklearn.metrics import mean_absolute_error



Score = mean_absolute_error(ValY,Prediction)



print(Score)
from sklearn.metrics import mean_squared_error



Score = mean_squared_error(ValY,Prediction)



print(Score)
# Load Data Again



Data = pd.read_csv('../input/home-data-for-ml-course/train.csv')



DataX = Data[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]



DataY = Data['SalePrice']



i = round(len(DataX) * 0.9)



TrainX = DataX[:i]

print("TrainX")

display(TrainX)



ValX = DataX[i:]

print("ValX")

display(ValX)



TrainY = DataY[:i]

print("TrainY")

display(TrainY)



ValY = DataY[i:]

print("ValY")

display(ValY)



# กดปุ่ม Output เพื่อดูข้อมูลใน TrainX, ValX, TrainY, และ ValY
from sklearn.tree import DecisionTreeRegressor



# สร้าง Model

SKLDecisionTreeRegModel = DecisionTreeRegressor(random_state=1);



# Fit Model (หรือ Train Model) โดยใช้ TrainX กับ TrainY

SKLDecisionTreeRegModel.fit(TrainX,TrainY)



# Predict Value โดยใช้ ValX ให้ได้ Prediction (Predict ค่า Y)

Prediction = SKLDecisionTreeRegModel.predict(ValX)
# สามารถดู Code หรือ Output ได้ เพื่อดูผลลัพท์ของ Model



print("Prediction")

display(Prediction)

print("Validation Y")

display(ValY)
# สร้าง Data จำลองมาเพื่อใช้เป็นตัวอย่าง

TrainX = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

ValX = np.array([[0, 0], [12, 3], [2, 2], [8, 3]])



# Graph TrainX

print("Blue = TrainX, Orange = ValX")

plt.title("All Points")

plt.scatter(TrainX[:,0],TrainX[:,1],c="blue")

plt.scatter(ValX[:,0],ValX[:,1],c="orange")

plt.show()
# Example Copied From https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

from sklearn.cluster import KMeans



# สร้าง Model

kmeans = KMeans(n_clusters=2, random_state=0)



# Fit Model (หรือ Train Model)

kmeans.fit(TrainX)



# Predict Model

Prediction = kmeans.predict(ValX)



display(Prediction)
# Load Data Again



Data = pd.read_csv('../input/home-data-for-ml-course/train.csv')



DataX = Data[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]



DataY = Data['SalePrice']



i = round(len(DataX) * 0.9)



TrainX = DataX[:i]

print("TrainX")

display(TrainX)



ValX = DataX[i:]

print("ValX")

display(ValX)



TrainY = DataY[:i]

print("TrainY")

display(TrainY)



ValY = DataY[i:]

print("ValY")

display(ValY)



# กดปุ่ม Output เพื่อดูข้อมูลใน TrainX, ValX, TrainY, และ ValY
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn.metrics import mean_absolute_error



for m in [Ridge, Lasso, ElasticNet]:

    Model = m(random_state=0);

    

    Model.fit(TrainX,TrainY)

    

    Prediction = Model.predict(ValX)

    

    print(f"Model = {m}, Score = {mean_absolute_error(ValY, Prediction)}")
from sklearn.datasets import load_iris

DataX, DataY = load_iris(return_X_y=True)



TrainX = DataX[:-2,:]

TrainY = DataY[:-2]

ValX = DataX[-2:,:]

ValY = DataY[:-2]



print("TrainX")

display(TrainX)



print("TrainY")

display(TrainY)



print("ValX")

display(ValX)



print("ValY")

display(ValY)
# ใช้ Data จาก sklearn.datasets (กดปุ่ม Code หรือ Output ด้านบนเพื่อดู)



from sklearn.linear_model import LogisticRegression



Model = LogisticRegression(random_state=0);



Model.fit(TrainX,TrainY)



Prediction = Model.predict(ValX)



display(Prediction)
# สร้าง Data จำลองมาเพื่อใช้เป็นตัวอย่าง

TrainX = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

ValX = np.array([[0, 0], [12, 3], [2, 2], [8, 3]])



# Graph TrainX

print("Blue = TrainX, Orange = ValX")

plt.title("All Points")

plt.scatter(TrainX[:,0],TrainX[:,1],c="blue")

plt.scatter(ValX[:,0],ValX[:,1],c="orange")

plt.show()
# Example Copied From https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

from sklearn.cluster import KMeans



# สร้าง Model

kmeans = KMeans(n_clusters=2, random_state=0)



# Fit Model (หรือ Train Model)

kmeans.fit(TrainX)



# Predict Model

Prediction = kmeans.predict(ValX)



display(Prediction)
# Copied from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]



from sklearn.neighbors import NearestNeighbors



neigh = NearestNeighbors(n_neighbors=1)

neigh.fit(samples)



print(neigh.kneighbors([[1., 1., 1.]]))