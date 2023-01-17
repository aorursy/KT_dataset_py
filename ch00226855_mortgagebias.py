# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import roc_curve, auc  #Metrics

from sklearn.model_selection import train_test_split
# fillColor = "#FFA07A"

# fillColor2 = "#F1C40F"

loans = pd.read_csv('/kaggle/input/ny-home-mortgage/ny_hmda_2015.csv')

loans.head()
# Covert categorical variables into numbers

cols = [f_ for f_ in loans.columns if loans[f_].dtype != 'object']

features = cols



list_to_remove = ['action_taken','purchaser_type',

                  'denial_reason_1','denial_reason_2','denial_reason_3','sequence_number']



features= list(set(cols).difference(set(list_to_remove)))



X = loans[features]

y = loans['action_taken']
# We define a function in which we mark the Loans which are 

# originated as 1 and the Loans which are NOT originated as 0

def change_action_taken(y):

    if ( y == 1):

        return 1

    else:

        return 0
# Apply the above function to get the labels for each record

y = loans['action_taken'].apply(change_action_taken)



X = X.fillna(0)
from lightgbm import LGBMClassifier
# Construct a Gradient Boosting Classifier

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

first_model = LGBMClassifier(random_state=1).fit(train_X, train_y)
# Get predictions on the validation set

predictions =  first_model.predict_proba(val_X)
# Plot the ROC curve.

# It is close to the upper bound and the left bound, so its performance is reasonable.

fpr, tpr, thresholds = roc_curve(val_y, predictions[:,1])



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
# Idea1: What are the approval rates in the model predictions for each race?
loans.columns
loans['applicant_ethnicity_name'].value_counts()
loans['applicant_ethnicity'].value_counts()
# Get loan applications from white applicants

loans_not_latino = X[X['applicant_ethnicity'] == 2]



loans_latino = X[X['applicant_ethnicity'] == 1]
predictions_not_latino = first_model.predict(loans_not_latino)

print(predictions_not_latino)



predictions_latino = first_model.predict(loans_latino)

print(predictions_latino)
approval_rate_not_latino = np.sum(predictions_not_latino) / predictions_not_latino.shape

print("Approval rate for Non Latino's:", approval_rate_not_latino * 100)



approval_rate_latino = np.sum(predictions_latino) / predictions_latino.shape

print("Approval rate for Latino's:", approval_rate_latino * 100)
loans['applicant_race_name_1'].value_counts()
loans['applicant_race_1'].value_counts()
# Get loan applications from white applicants

loans_white = X[X['applicant_race_1'] == 5]

# loans_white.head()



# Get loan applications from black applicants

loans_black = X[X['applicant_race_1'] == 3]

# loans_black.head()



# Get loan applications from asian applicants

loans_asian = X[X['applicant_race_1'] == 2]

# loans_asian.head()



# Get loan applications from indian applicants

loans_indian = X[X['applicant_race_1'] == 1]

# loans_indian.head()



# Get loan applications from Hawaiian applicants

loans_hawaiian = X[X['applicant_race_1'] == 4]

# loans_hawaiian.head()
# Get model predictions on these applicants

predictions_white = first_model.predict(loans_white)

print(predictions_white)



# Get model predictions on these applicants

predictions_black = first_model.predict(loans_black)

print(predictions_black)



# Get model predictions on these applicants

predictions_asian = first_model.predict(loans_asian)

print(predictions_asian)



# Get model predictions on these applicants

predictions_indian = first_model.predict(loans_indian)

print(predictions_indian)



# Get model predictions on these applicants

predictions_hawaiian = first_model.predict(loans_hawaiian)

print(predictions_hawaiian)
# Calculate the approval rates among predictions

approval_rate_white = np.sum(predictions_white) / predictions_white.shape

print("Approval rate for Whites:", approval_rate_white * 100)



approval_rate_black = np.sum(predictions_black) / predictions_black.shape

print("Approval rate for Black:", approval_rate_black* 100)



approval_rate_asian = np.sum(predictions_asian) / predictions_asian.shape

print("Approval rate for Asian:", approval_rate_asian* 100)



approval_rate_indian = np.sum(predictions_indian) / predictions_indian.shape

print("Approval rate for Indian:", approval_rate_indian* 100)



approval_rate_hawaiian = np.sum(predictions_hawaiian) / predictions_hawaiian.shape

print("Approval rate for Hawaiian:", approval_rate_hawaiian* 100)



#Divide each racial group into different income groups and compare 