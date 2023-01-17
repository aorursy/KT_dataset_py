import matplotlib.pyplot as plt



import numpy as np

import pandas as pd



import scipy

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
#Reading the datasets

data_v1 = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

data_v0 = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")

data = pd.concat([data_v1, data_v0])



print(data.shape)



data.head()
data = data.drop_duplicates()

data.shape
#Removing the serial number column as it adds no correlation to any columns

data = data.drop(columns = ["Serial No."])



#The column "Chance of Admit" has a trailing space which is removed

data = data.rename(columns={"Chance of Admit ": "Chance of Admit"})



data.head()
def get_training_data(df):

    """

    This function splits the data into X and y variables and returns them

    """

    X = df.drop(columns = ["University Rating", "Chance of Admit"])

    y = df["Chance of Admit"]

    

    return X, y
def train_model(university_rating):

    """

    1. Takes the subset only for one university rating

    2. Invokes the get_training_data function,

    3. Fits a linear regression model

    4. Cross validates it

    5. Returns the model object and the metrics for cross validation

    """

    #Filtering for one university fromt the data dataframe

    df = data[data["University Rating"] == university_rating]

    print(df.shape)

    

    #Splitting into X and y for regression

    X, y = get_training_data(df)

    

    regressor = LinearRegression()

    regressor.fit(X, y)

    

    metric = cross_val_score(regressor, X, y, cv = 5)

    

    return regressor, metric
university_ratings = data["University Rating"].unique()



university_recommendations = {}



for u in university_ratings:

    regressor, metric = train_model(u)

    university_recommendations["University ranking " + str(u)] = {'model': regressor, 'metric': metric}
university_recommendations
test = data.sample(20)

test = test.drop(columns = ["Chance of Admit", "University Rating"])

test.head()
predictions = {}



for uni in university_recommendations.keys():

    model = university_recommendations[uni]["model"]

    

    predictions[uni] = model.predict(test)

    

pred = pd.DataFrame(predictions)

pred.head(10)