import pandas as pd

import shap

import os

import pickle
'''



This program is attempting to classify a foster child's case into two categories:



0 - Low risk of multiple removals in their time in the foster care sytem.

1 - Elevated risk of multiple removals.



By entering in details about the case, the program will use a Random Forest model 

to make a prediction.  That prediction is nothing but a 0 or 1, however, we can extrapolate

how close to either value the model predicted.  The visualization at the bottom shows

different features and how they influenced the prediction.



With an experienced practitioner in the loop, more analysis could be done to better understand 

the risk of each child.  The model can only show us which characteristics influenced the 

prediction the most, and how they influenced this prediction.



Note that the model was built on approx. 2,300 data rows.  It is, overall, about 83% accurate.



'''
# Input case details

INPUT_VALUES = pd.Series({



'zip_count': 150.0,

# the number of total cases within child's zip code



'number_participants': 6.0,

# the total number of participants in child's case



'case_duration_yrs': 3.5,

# the total time between begin date and end date (in years)



'number_caregivers': 2.0,

# the number of unique caregivers in child's case



'age_child': 9.5,

# the age of the child



'avg_age_caregiver': 28.2,

# the average age of a caregiver in child's case



'avg_gross_income_zip': 70000.0,

# the average gross income per the zip code associated with the child's case



'first_placement': 3.0,

# the first placement of the child (4-Adoption, 3-Foster(Relative), 2-Foster(Non-Relative), 1-Others)



'gender': 1.0,

# 0 for male, 1 for female



'ethnicity': 1.0,

# child's ethnicity (3-African American/Black, 2-Hispanic/Latino, 1-Eastern European, 0-Other)



'perc_life': 15.0,

# the percentage of the child's life that they have spent in foster care cases



'first_place_duration': 0.3

# the length of the child's first placement (in years)



})



INPUT_ARRAY = INPUT_VALUES.values.reshape(1, -1)
# Open saved model and make prediction

with open('/kaggle/input/models/saved_rf_model.pkl', 'rb') as file:

    pickle_model = pickle.load(file)

    

prediction = pickle_model.predict(INPUT_ARRAY)



if prediction == 1:

    print("PREDICTION RESULT: Significant risk of multiple removals")

elif prediction == 0:

    print("PREDICTION RESULT: No significant risk of multiple removals")
# Output explainer plot

pickle_model.predict_proba(INPUT_ARRAY)

exp = shap.TreeExplainer(pickle_model)

shap_values = exp.shap_values(INPUT_VALUES)



shap.initjs()

shap.force_plot(exp.expected_value[1], shap_values[1], INPUT_VALUES)