import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.model_selection import train_test_split

data = pd.read_csv("/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv")

data
# We will use some of the data for building a model, and some for evaluating the model

[training_data, testing_data] = train_test_split(data)

training_data = pd.DataFrame(training_data)

testing_data = pd.DataFrame(testing_data)

# 

def stayToNumber(stay):

    return {

        '0-10': 5,

        '11-20': 15,

        '21-30': 25,

        '31-40': 35,

        '41-50': 45,

        '51-60': 55,

        '61-70': 65,

        '71-80': 75,

        '81-90': 85,

        '91-100': 95,

        'More than 100 Days': 101

        

    }[stay]



def ageToNumber(age):

    return {

    '0-10': 5,

    '11-20': 15,

    '21-30': 25,

    '31-40': 35,

    '41-50': 45,

    '51-60': 55,

    '61-70': 65,

    '71-80': 75,

    '81-90': 85,

    '91-100': 95

}[age]



def severityToNumber(severity):

    return {

        'Extreme': 3,

        'Moderate': 2,

        'Minor': 1,

    }[severity]



training_data["Stay"] = training_data["Stay"].apply(stayToNumber)

training_data["Age"] = training_data["Age"].apply(ageToNumber)

training_data["Severity"] = training_data["Severity of Illness"].apply(severityToNumber)

testing_data["Stay"] = testing_data["Stay"].apply(stayToNumber)

testing_data["Age"] = testing_data["Age"].apply(ageToNumber)

testing_data["Severity"] = testing_data["Severity of Illness"].apply(severityToNumber)
# Here I'm ignoring some of the features to make a smaller example

training_data = training_data.drop(['case_id', 'Ward_Type','Admission_Deposit','City_Code_Hospital', 'Hospital_code', 'Hospital_type_code', 'Hospital_region_code', 'Ward_Facility_Code', 'City_Code_Patient', 'Visitors with Patient', 'Bed Grade',  'Available Extra Rooms in Hospital'], axis=1)
training_data.drop(['patientid'], axis=1).hist()
from matplotlib.pyplot import bar

import matplotlib.pyplot as plt

axes = plt.axes()

axes.set_ylim([0, 1])

trainingDataCorrelation = training_data.corr()["Stay"]

bar(trainingDataCorrelation.drop(["Stay"]).sort_values().index, trainingDataCorrelation.drop(["Stay"]).sort_values())

from matplotlib.pyplot import scatter

scatter(training_data["Age"].apply(lambda x: x + np.random.randn(1)*0.5), training_data["Stay"].apply(lambda x: x + np.random.randn(1)*0.5), alpha=0.1)
from sklearn.linear_model import LinearRegression

# A Linear Regression model will attempt to 'learn' the correlation in the training data and then use this to make predictions

# We take a sample to train the model as trying to use the whole dataset led to memory usage issues

training_data_sample = training_data

predictByAge = LinearRegression().fit(training_data_sample[["Age"]].values,training_data_sample[["Stay"]].values)

predictBySeverity = LinearRegression().fit(training_data_sample[["Severity"]].values,training_data_sample[["Stay"]].values)
from matplotlib.pyplot import plot





import numpy as np

import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_ylabel('Predicted Stay Length (Days)')

ax1.set_title('Prediction by Age')



ax1.scatter([20,30,40,50,60,70,80,90], predictByAge.predict([[20],[30],[40],[50],[60],[70],[80],[90]]).flatten())

ax1.set_xlabel('Age')

    

fig = plt.figure()

ax2.set_title('Prediction by Severity')

ax2.set_ylabel('Predicted Stay Length (Days)')



ax2.scatter([1,2,3], predictBySeverity.predict([[1],[2],[3]]).flatten())

ax2.set_xlabel('Severity')

def calculateError(classifier, classifierInput):

    error = (classifier.predict(classifierInput).flatten()  - testing_data["Stay"]).abs().describe()[["mean", "std"]]

    return "Mean Error is " +  str(round(error["mean"], 2)) + " with standard deviation of " + str(round(error["std"], 2))



{"Age Classifer Error": calculateError(predictByAge, testing_data[["Age"]]),

"Severity Error": calculateError(predictBySeverity, testing_data[["Severity"]])}



# What Next?

# We can experiment with different approaches to try to make a better model
predictByAgeAndSeverity = LinearRegression().fit(training_data_sample[["Age", "Severity"]].values,training_data_sample[["Stay"]].values)
import matplotlib.pyplot as plt

import numpy as np

from itertools import product



df = pd.DataFrame(list(product([20,30,40,50,60,70,80,90], [0,1,2])), columns=["Age", "Severity"])

df["Prediction"] = df.apply(lambda x:   predictByAgeAndSeverity.predict([x])[0][0], axis=1)



fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(df["Age"], df["Severity"], c=df["Prediction"] ,cmap='YlOrRd')

ax.set_xlabel('Age')

ax.set_ylabel('Severity')



{"Age Classifer Error": calculateError(predictByAge, testing_data[["Age"]]),

"Severity Error": calculateError(predictBySeverity, testing_data[["Severity"]]),

"Age & Severity Error": calculateError(predictByAgeAndSeverity, testing_data[["Age", "Severity"]]) }
