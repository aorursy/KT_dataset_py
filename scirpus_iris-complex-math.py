import numpy as np

import warnings

warnings.simplefilter("ignore", category=np.ComplexWarning)



import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

%matplotlib inline
train=pd.read_csv('../input/iris/Iris.csv')

le = LabelEncoder()

train.Species = le.fit_transform(train.Species)-1

train.head()

complextrain = train[train.columns[1:]].copy()

complextrain[train.columns[1:-1]] = complextrain[train.columns[1:-1]].astype('complex')
def GPComplex(data):

    return (1.0*np.real(np.tanh((np.sinh(((((data["PetalLengthCm"]) + (((((complex(-2.0)) * 2.0)) / (np.cosh((((np.sin((np.sin((complex(1.14559674263000488)))))) * (((((np.sin((((data["PetalWidthCm"]) * 2.0)))) * (data["PetalWidthCm"]))) * (((data["PetalWidthCm"]) * (np.sin((((data["PetalLengthCm"]) - (complex(1.14559674263000488)))))))))))))))))/2.0)))))) +

            0.926925*np.real(((np.tanh((((((((((np.cosh((data["PetalWidthCm"]))) - (data["SepalWidthCm"]))) * 2.0)) * 2.0)) * 2.0)))) / (((((np.sin((np.cosh((data["PetalLengthCm"]))))) - (((np.sinh((data["PetalWidthCm"]))) * 2.0)))) + (((np.cosh((((complex(2.0)) - (np.sinh((data["PetalWidthCm"]))))))) * (complex(8.16992855072021484)))))))) +

            1.0*np.real((((((-((((np.tanh((data["SepalWidthCm"]))) - (np.sin((np.sqrt((((np.tanh((np.cosh((np.sin((data["SepalWidthCm"]))))))) - (((np.tanh((np.tanh((((((data["SepalWidthCm"]) - (np.sin((data["SepalWidthCm"]))))) - (np.sqrt((data["SepalWidthCm"]))))))))) - (complex(2.0))))))))))))))) * 2.0)) * 2.0)) +

            0.944119*np.real(((data["SepalLengthCm"]) / (((np.sinh((data["SepalLengthCm"]))) - (((complex(3.0)) + (((complex(8.0)) + (np.sqrt(((-(((((((np.sinh((data["SepalLengthCm"]))) + (np.sinh((data["SepalLengthCm"]))))) + (np.sinh(((-((((complex(14.77018547058105469)) + (np.sin((np.sinh((data["SepalLengthCm"])))))))))))))/2.0))))))))))))))) +

            0.915592*np.real((-((((((data["SepalLengthCm"]) - (complex(1.0)))) / (((np.sin((((data["SepalLengthCm"]) - (np.sin((np.sin((complex(-3.0)))))))))) - (np.tanh((np.sqrt((((complex(1.0)) - (((np.tanh((np.cos((((data["SepalLengthCm"]) - (np.sin((data["SepalLengthCm"]))))))))) - (complex(-3.0))))))))))))))))) +

            1.0*np.real(((data["PetalLengthCm"]) / (np.sinh((((((complex(-3.0)) * 2.0)) + (((((((((((np.cos((data["PetalLengthCm"]))) * 2.0)) * (np.sqrt((np.cos((complex(-3.0)))))))) / (np.cos((data["PetalWidthCm"]))))) + (np.cos((np.cos((data["PetalWidthCm"]))))))) + (np.cos((data["PetalWidthCm"]))))))))))) +

            1.0*np.real(((np.cos((np.sqrt(((((((np.cosh((data["PetalLengthCm"]))) / (np.sqrt((((data["SepalWidthCm"]) / (np.sqrt((complex(-3.0)))))))))) + (np.sinh((data["SepalLengthCm"]))))/2.0)))))) / ((((((((np.sinh((data["PetalLengthCm"]))) * 2.0)) / (((data["PetalLengthCm"]) / (complex(-3.0)))))) + (np.cosh((data["SepalLengthCm"]))))/2.0)))) +

            0.986323*np.real(((((((complex(2.0)) - (complex(2.0)))) - (np.sin((((np.sinh((((np.sinh((np.sin((((np.sinh((((np.sinh((data["SepalWidthCm"]))) * 2.0)))) / 2.0)))))) * 2.0)))) / 2.0)))))) / (((data["SepalWidthCm"]) - (((((np.sinh((data["SepalWidthCm"]))) * 2.0)) * 2.0)))))) +

            1.0*np.real(np.sin((((((np.cos((np.sinh((data["SepalLengthCm"]))))) - ((-((data["SepalLengthCm"])))))) / (np.sqrt(((-((np.cos((np.cosh((((((np.cosh((np.cos((np.cos((np.cos((np.sinh((((data["SepalLengthCm"]) / 2.0)))))))))))) * (np.sin((np.cosh((data["SepalLengthCm"]))))))) / 2.0))))))))))))))) +

            1.0*np.real((((((((complex(3.0)) + (np.cos((((((data["PetalWidthCm"]) * 2.0)) * 2.0)))))/2.0)) - (((np.sin((((np.cos((((((np.cos((data["PetalLengthCm"]))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)))) / (np.sinh((np.sinh((((complex(3.0)) + (np.sin((((np.cosh((data["PetalWidthCm"]))) * 2.0)))))))))))))
plt.figure(figsize=(15,15))

_ = plt.scatter(GPComplex(complextrain),complextrain.Species)
accuracy_score(complextrain.Species,np.round(GPComplex(complextrain)))