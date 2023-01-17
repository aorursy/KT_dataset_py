# -*- coding: utf-8 -*-

'''

Python 3.5.2

developer mauricio munoz, contact: yemauricio@gmail.com

'''
import pandas as pd

from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings("ignore")
# Importing dataset and split it according of the class i.e. class D or S

dataset = pd.read_csv("../input/SalesKaggle3.csv")

dataset_copy =  dataset.copy()
ClassD_product = dataset_copy[dataset_copy.MarketingType == "D"]

ClassD_product = ClassD_product[["SKU_number","ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice", "SoldFlag"]]

ClassS_product = dataset_copy[dataset_copy.MarketingType == "S"]

ClassS_product = ClassS_product[["SKU_number","ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice", "SoldFlag"]]

ClassD_product_forprediction = ClassD_product[ClassD_product.SoldFlag.isnull()]

ClassD_product_fortrain = ClassD_product[ClassD_product.SoldFlag.isnull()== False]

ClassS_product_forprediction = ClassS_product[ClassS_product.SoldFlag.isnull()]

ClassS_product_fortrain = ClassS_product[ClassS_product.SoldFlag.isnull()== False]
# Data for training and testing for class D products

X_train_D = ClassD_product_fortrain[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]

Y_train_D = ClassD_product_fortrain[["SoldFlag"]]

# test data is the data that will be forecasted

X_test_D = ClassD_product_forprediction[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]
# Creation of classfier and training

# The training step is developed using 90% of the data, the resting 10% is used to evaluate the Error

clf_D = MLPClassifier(activation="tanh",solver="lbfgs",hidden_layer_sizes=(6, 1))

clf_D.fit(X_train_D[:31608],Y_train_D[:31608])

prediction_D_train = clf_D.predict(X_train_D[31608:])
prediction_D_train = pd.DataFrame(prediction_D_train)

prediction_D_train.columns = ["Prediction D"]
# Error level of the classifier

accuracy = 0

for i in range(0, len(prediction_D_train)):

    if Y_train_D["SoldFlag"].ix[31608+i] == prediction_D_train["Prediction D"].ix[i]:

        accuracy += 1

accuracy/len(prediction_D_train)*100
# Forecasting for the target data and presenting 

prediction_D_test = clf_D.predict(X_test_D)

prediction_D_test = pd.DataFrame(prediction_D_test)

prediction_D_test.columns = ["Prediction SoldFlag D"]

ClassD_product_forprediction = ClassD_product_forprediction.reset_index()

ClassD_product_forprediction["SoldFlag"] = prediction_D_test["Prediction SoldFlag D"]

ClassD_product_forprediction = ClassD_product_forprediction.set_index("index")

ClassD_product_forprediction
# Products that would have 1.0 flag 

ClassD_product_forprediction[ClassD_product_forprediction["SoldFlag"]>0]
# Data for training and testing for class S products

X_train_S = ClassS_product_fortrain[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]

Y_train_S = ClassS_product_fortrain[["SoldFlag"]]

X_test_S = ClassS_product_forprediction[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]
# Creation of classifier and training

# The training step is developed using 90% of the data, the resting 10% is used to evaluate the Error

clf_S = MLPClassifier(activation="tanh",solver="lbfgs",hidden_layer_sizes=(6, 1))

clf_S.fit(X_train_S[:36790],Y_train_S[:36790])
prediction_S_train = clf_S.predict(X_train_S[36790:])
prediction_S_train = pd.DataFrame(prediction_S_train)

prediction_S_train.columns = ["Prediction S"]
# Error level of the classifier

accuracy_S = 0

for i in range(0, len(Y_train_S[36790:])):

    if Y_train_S["SoldFlag"].ix[71909+i] == prediction_S_train["Prediction S"].ix[i]:

        accuracy_S += 1

accuracy_S/len(prediction_S_train)*100
# Forecasting for the target data and presenting Results

prediction_S_test = clf_S.predict(X_test_S)

prediction_S_test= pd.DataFrame(prediction_S_test)

prediction_S_test.columns = ["Prediction S"]

ClassS_product_forprediction = ClassS_product_forprediction.reset_index()

ClassS_product_forprediction["SoldFlag"] = prediction_S_test["Prediction S"]

ClassS_product_forprediction = ClassS_product_forprediction.set_index("index")

ClassS_product_forprediction
# Products that would have 1.0 flag 

ClassS_product_forprediction[ClassS_product_forprediction["SoldFlag"]>0]