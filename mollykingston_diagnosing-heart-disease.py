%matplotlib inline

import math
import random
import matplotlib
import numpy as np
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from typing import *
def healthy_choices(food: str) -> bool:
    if food == "apples":
        return True
    else:
        return False
healthy_choices("apples")  # Expected True
healthy_choices("donuts")  # Expected False
healthy_choices("green beans")  # Expected False
def healthy_choices(food: str) -> bool:
    if food == "apples":
        return True
    elif food == "green beans":
        return True
    else:
        return False
healthy_choices("apples")  # Expected True
healthy_choices("donuts")  # Expected False
healthy_choices("green beans")  # Expected True
def healthy_choices(food: str, servings: int) -> bool:
    if food == "apples" and servings <= 10:
        return True
    elif food == "green beans" and servings <= 10:
        return True
    else:
        return False

healthy_choices("apples", 2)   # Expected True
healthy_choices("donut", 2)   # Expected False
healthy_choices("green beans", 40)   # Expected False
test = pd.read_csv("../input/cleveland-data/cleveland-testing.csv")
test_disease = test[test["heart_disease"] == True]
test_healthy = test[test["heart_disease"] == False]
train = pd.read_csv("../input/cleveland-data/cleveland-training.csv")
train_disease = train[train["heart_disease"] == True]
train_healthy = train[train["heart_disease"] == False]
train_disease
train_healthy
def analyze_file(filename: str, predict, axes):
    # Open the file to get the data

    data = pd.read_csv("../input/cleveland-data/cleveland-" + filename + ".csv")

    # Set up the storage of the results

    results = [[0, 0], 
               [0, 0]]
    
    for index, row in data.iterrows():

        # Make a prediction
        
        prediction = predict(row["age"], row["female"], row["chest_pain"], row["rest_bps"], 
                             row["cholesterol"], row["high_fasting_blood_sugar"], row["rest_ecg"], 
                             row["maximum_heart_rate"], row["exercise_angina"], row["vessels"])

        # Saving the results

        results[1 - int(prediction)][1 - row["heart_disease"]] += 1

    # Calculating the accuracy on all patients

    total = len(data)
    correct = results[0][0] + results[1][1]

    # Graph the results for the user
    
    df_cm = pd.DataFrame(results, ["Disease", "Healthy"], ["Disease", "Healthy"])
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", ax=axes)# font size
    axes.set_xlabel("Actual")
    axes.set_ylabel("Predicted")
    axes.set_title(filename.capitalize() + " Accuracy: " + "{:.2%}".format(correct / float(total)))
def analyze(predict):
    # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # https://matplotlib.org/users/text_intro.html

    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    analyze_file("training", predict, ax1)
    analyze_file("testing", predict, ax2)
    plt.tight_layout()

def default_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                    high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                    exercise_angina: bool, vessels: int) -> bool:
    
    ##################################
    # BEGIN PREDICTION CODE
    ##################################
    
    prediction = True
    
    ##################################
    # END PREDICTION CODE
    ##################################

    return prediction

analyze(default_predict)
def female_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                   high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                   exercise_angina: bool, vessels: int) -> bool:
    
    ##################################
    # BEGIN PREDICTION CODE
    ##################################
    
    if female:
        prediction = False
    else:
        prediction = True
  
    ##################################
    # END PREDICTION CODE
    ##################################

    return prediction

analyze(female_predict)
def age_female_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                       high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                       exercise_angina: bool, vessels: int) -> bool:
    if age >= 60 and female == False:
        return True
    else: 
        return False
    return prediction

analyze(age_female_predict)
def mhr_cp_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                   high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                   exercise_angina: bool, vessels: int) -> bool:
    if maximum_heart_rate <= 160 and chest_pain == "asymptomatic":
        return True
    else:
        return False
    return prediction

analyze(mhr_cp_predict)
def ver_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                exercise_angina: bool, vessels: int) -> bool:
    if vessels >= 1 and rest_ecg == "hypertrophy":
        return True
    elif exercise_angina == True:
        return True
    else:
        return False
    return prediction


analyze(ver_predict)
def overfit_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                    high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate: int, 
                    exercise_angina: bool, vessels: int) -> bool:
    
    ##################################
    # BEGIN PREDICTION CODE
    ##################################
    
    if age == 67 and not female and chest_pain == "asymptomatic" and rest_bps == 160 and \
    cholesterol == 286 and not high_fasting_blood_sugar and rest_ecg == "hypertrophy" and \
    maximum_heart_rate == 108 and exercise_angina and vessels == 3:
            prediction = True
    elif age == 67 and not female and chest_pain == "asymptomatic" and rest_bps == 120 and \
    cholesterol == 229 and not high_fasting_blood_sugar and rest_ecg == "hypertrophy" and \
    maximum_heart_rate == 129 and exercise_angina and vessels == 2:
            prediction = True
    elif age == 62 and female and chest_pain == "asymptomatic" and rest_bps == 140 and \
    cholesterol == 268 and not high_fasting_blood_sugar and rest_ecg == "hypertrophy" and \
    maximum_heart_rate == 160 and not exercise_angina and vessels == 2:
            prediction = True
    elif age == 63 and not female and chest_pain == "asymptomatic" and rest_bps == 130 and \
    cholesterol == 254 and not high_fasting_blood_sugar and rest_ecg == "hypertrophy" and \
    maximum_heart_rate == 147 and not exercise_angina and vessels == 1:
            prediction = True
    elif age == 53 and not female and chest_pain == "asymptomatic" and rest_bps == 140 and \
    cholesterol == 203 and high_fasting_blood_sugar and rest_ecg == "hypertrophy" and \
    maximum_heart_rate == 155 and exercise_angina and vessels == 0:
            prediction = True
    else:
        prediction = False
        
    ##################################
    # END PREDICTION CODE
    ##################################

    return prediction

analyze(overfit_predict)
def final_predict(age: int, female: bool, chest_pain: str, rest_bps: int, cholesterol: int, 
                  high_fasting_blood_sugar: bool, rest_ecg: str, maximum_heart_rate : int, 
                  exercise_angina: bool, vessels: int) -> bool:
    if vessels > 0 and chest_pain == "asymptomatic":
        return True
    
    elif vessels > 0 and chest_pain == "typical angina" and rest_bps <= 138:
        return True
    
    elif vessels > 0 and chest_pain == "non-anginal pain" and female == False:
        return True
    
    elif vessels > 0 and chest_pain == "atypical angina" and rest_ecg == "hypertrophy":
        return True
    
    elif vessels == 0 and exercise_angina == True:
        return True
    else:
        return False
    return prediction

analyze(final_predict)
test_healthy
test_disease
