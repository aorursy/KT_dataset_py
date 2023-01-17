import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/diabetes.csv')

train.shape
train.head()
ss = StandardScaler()

train[train.columns[:-1]] = ss.fit_transform(train[train.columns[:-1]])
def Output(p):

    return 1.0/(1+np.exp(-p))



def GPI(data):

    return Output(-0.623621 +

                    1.0*np.tanh(((((((data["DiabetesPedigreeFunction"]) + (((data["DiabetesPedigreeFunction"]) / 2.0)))/2.0)) + (((((((data["Glucose"]) * 2.0)) + (np.minimum(((data["Age"])), ((np.maximum(((((data["DiabetesPedigreeFunction"]) * (np.minimum(((data["Pregnancies"])), ((data["DiabetesPedigreeFunction"]))))))), ((data["Glucose"]))))))))) + (data["BMI"]))))/2.0)) +

                    1.0*np.tanh(np.minimum(((((data["Age"]) + (((((((data["BMI"]) + (data["Glucose"]))/2.0)) > ((((((((data["Glucose"]) / 2.0)) / 2.0)) < (data["BloodPressure"]))*1.)))*1.))))), (((((data["Glucose"]) > (((((((data["Age"]) > (data["Glucose"]))*1.)) < (data["Age"]))*1.)))*1.))))) +

                    1.0*np.tanh(np.minimum(((np.maximum(((data["BMI"])), ((data["BloodPressure"]))))), (((((-2.0) > (np.minimum((((-1.0*((data["BMI"]))))), ((((((np.minimum(((data["BMI"])), ((data["Glucose"])))) - (np.minimum((((0.97150707244873047))), ((data["SkinThickness"])))))) - (data["DiabetesPedigreeFunction"])))))))*1.))))) +

                    1.0*np.tanh((((((((((np.minimum((((((((((((data["Glucose"]) / 2.0)) > (data["BloodPressure"]))*1.)) / (data["Glucose"]))) * (data["Pregnancies"])))), ((np.minimum(((data["Glucose"])), ((((data["BloodPressure"]) + (data["Age"]))))))))) < (-3.0))*1.)) * 2.0)) * 2.0)) * 2.0)) +

                    1.0*np.tanh((((((np.maximum(((((data["Glucose"]) * 2.0))), (((((data["BMI"]) < (((data["Age"]) * (((data["Glucose"]) + (-3.0))))))*1.))))) / (((data["Age"]) * (((data["DiabetesPedigreeFunction"]) + (-3.0))))))) > ((-1.0*((-3.0)))))*1.)) +

                    1.0*np.tanh((-1.0*((((((5.0)) < (((((-2.0) + (((data["Glucose"]) - ((1.52228271961212158)))))) * (((np.minimum((((((1.52228271961212158)) + (((data["BMI"]) + (((1.0) / (data["Insulin"])))))))), ((data["BMI"])))) * 2.0)))))*1.))))) +

                    1.0*np.tanh((((np.maximum((((((((((-1.0*((data["BMI"])))) * 2.0)) * 2.0)) - (2.0)))), ((np.maximum(((((2.0) - (data["Pregnancies"])))), (((((-1.0*((((((2.0) - (data["DiabetesPedigreeFunction"]))) * 2.0))))) * 2.0)))))))) < (data["DiabetesPedigreeFunction"]))*1.)) +

                    1.0*np.tanh((((np.minimum(((data["Insulin"])), (((((((data["BMI"]) > (((data["Glucose"]) / (data["Glucose"]))))*1.)) + (data["Age"])))))) > (np.maximum(((np.maximum(((((1.0) / 2.0))), (((((data["Age"]) > (2.0))*1.)))))), ((data["Glucose"])))))*1.)) +

                    1.0*np.tanh((((-1.0*(((((((((((((data["BMI"]) * (data["DiabetesPedigreeFunction"]))) * (data["DiabetesPedigreeFunction"]))) * (np.minimum(((data["BMI"])), (((((data["BloodPressure"]) + (((((1.0)) + ((2.0)))/2.0)))/2.0))))))) > ((2.0)))*1.)) * (data["DiabetesPedigreeFunction"])))))) * 2.0)) +

                    0.900762*np.tanh((-1.0*(((((np.minimum(((data["BMI"])), ((((data["Age"]) * 2.0))))) > (((((((data["BloodPressure"]) < (np.minimum((((((data["Age"]) < ((((data["BloodPressure"]) < (data["BMI"]))*1.)))*1.))), ((data["Insulin"])))))*1.)) < (data["BMI"]))*1.)))*1.))))))



def GPII(data):

    return Output(-0.623621 +

                    0.992381*np.tanh((((((data["BMI"]) + (((data["Glucose"]) * 2.0)))/2.0)) - ((-1.0*(((((data["DiabetesPedigreeFunction"]) + ((((np.minimum(((data["Glucose"])), ((np.minimum(((data["DiabetesPedigreeFunction"])), ((((data["BMI"]) * 2.0)))))))) + (data["Pregnancies"]))/2.0)))/2.0))))))) +

                    1.0*np.tanh((((((2.0) > (data["Age"]))*1.)) * (((data["Age"]) + (((np.maximum(((data["Glucose"])), ((-2.0)))) - (((np.minimum(((data["BloodPressure"])), ((((data["Glucose"]) - (((((data["BloodPressure"]) / 2.0)) * (data["SkinThickness"])))))))) / 2.0)))))))) +

                    0.943153*np.tanh((((-2.0) > (((((((((((data["Insulin"]) > (data["Glucose"]))*1.)) < (data["Glucose"]))*1.)) + (((((((((data["Glucose"]) > (data["Pregnancies"]))*1.)) < (data["Glucose"]))*1.)) + ((-1.0*((data["Pregnancies"])))))))) / ((-1.0*((data["BMI"])))))))*1.)) +

                    1.0*np.tanh(((((((((data["BMI"]) * ((((4.0)) + (data["BMI"]))))) * 2.0)) * 2.0)) * ((((np.maximum(((data["Glucose"])), (((((1.89036774635314941)) / (((data["Glucose"]) + ((3.05443954467773438))))))))) < ((-1.0*((data["BMI"])))))*1.)))) +

                    1.0*np.tanh((((((data["Glucose"]) < ((-1.0*(((((((data["Glucose"]) < ((-1.0*(((((data["SkinThickness"]) > (np.minimum(((data["DiabetesPedigreeFunction"])), ((np.minimum(((data["BloodPressure"])), ((data["Pregnancies"]))))))))*1.))))))*1.)) + (np.maximum(((data["Pregnancies"])), ((data["BMI"]))))))))))*1.)) * (data["BMI"]))) +

                    1.0*np.tanh(np.minimum(((((data["Age"]) * ((((np.maximum(((data["BMI"])), (((((data["Glucose"]) > (data["BMI"]))*1.))))) < (data["Insulin"]))*1.))))), ((((((data["Age"]) * ((10.0)))) * ((((((3.0) - (data["Insulin"]))) < (data["Insulin"]))*1.))))))) +

                    1.0*np.tanh((-1.0*(((((((((((data["Age"]) > (data["DiabetesPedigreeFunction"]))*1.)) > (data["Age"]))*1.)) < (((((data["Insulin"]) / (data["Pregnancies"]))) * ((((np.minimum(((data["Age"])), ((1.0)))) > ((((data["SkinThickness"]) < (data["BloodPressure"]))*1.)))*1.)))))*1.))))) +

                    1.0*np.tanh((((data["Pregnancies"]) > ((((((10.0)) - (data["BloodPressure"]))) - ((((6.0)) - ((-1.0*((((((data["DiabetesPedigreeFunction"]) + ((((((data["Pregnancies"]) * (data["Pregnancies"]))) > (data["Pregnancies"]))*1.)))) / ((-1.0*((data["Age"]))))))))))))))*1.)) +

                    0.881422*np.tanh(((data["SkinThickness"]) * ((((((data["DiabetesPedigreeFunction"]) / (data["BMI"]))) > (np.maximum(((((3.0) + (((((((data["SkinThickness"]) / (data["BMI"]))) - ((((data["BloodPressure"]) + ((4.0)))/2.0)))) * (data["Pregnancies"])))))), ((-3.0)))))*1.)))) +

                    1.0*np.tanh((((-1.0*((((data["Age"]) + (((((((3.0)) < (data["DiabetesPedigreeFunction"]))*1.)) + (((((((((data["Insulin"]) * 2.0)) > (data["BMI"]))*1.)) < (data["Insulin"]))*1.))))))))) * ((((data["DiabetesPedigreeFunction"]) > (np.maximum(((2.0)), ((data["Glucose"])))))*1.)))))
print(accuracy_score(train.Outcome,GPI(train)>0.5))
print(accuracy_score(train.Outcome,GPII(train)>0.5))