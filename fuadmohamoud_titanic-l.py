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

# Try and Clean more Data to use, this time were going to clean the Embark section a bit
def get_training_fnames_extra_fields():
    return ['PassengerId','Survived','Pclass','Name','Male', 'Female','Age','SibSp','Parch','Ticket','Fare','Cabin','Cherbourg', 'Queenstown', 'Southampton']

def validateOutPutFileName(outputFileName):
    return outputFileName.endswith('.csv')

def clean_training_data_extra_fields(trainingDataFileName, intendedOutPutFileName):
    # Sanity check, make sure we are creating an excel file
    res = validateOutPutFileName(intendedOutPutFileName)
    if not res:
        intendedOutPutFileName +='.csv'
    data_list = []
    # Read through the original training file, append to list
    with open(trainingDataFileName, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data_list.append(row)
    # loop through list and append values to new file
    with open(intendedOutPutFileName, 'w') as file:
        # headers of our new csv file
        fnames = get_training_fnames_extra_fields()
        writer = csv.DictWriter(file, fieldnames=fnames)
        writer.writeheader()
        # this is to skip the first index of our list, because it contains the training csv files header information
        iterate = iter(data_list)
        next(iterate)
        for rows in iterate:
            isMale = 0
            isFemale = 0
            if rows[4].lower() == 'male':
                isMale = 1
            else:
                isFemale = 1
            isCherb = 0
            isQueens = 0
            isSoutham = 0
            embarkPort = str(rows[11]).lower()
            if embarkPort == 'q':
                isQueens = 1
            elif embarkPort == 'c':
                isCherb = 1
            else:
                isSoutham = 1
            writer.writerow({'PassengerId': rows[0],'Survived': rows[1], 'Pclass':rows[2],'Name': rows[3],'Male': isMale ,'Female': isFemale,'Age': rows[5], 'SibSp': rows[6], 'Parch': rows[7], 'Ticket': rows[8], 'Fare': rows[9], 'Cabin': rows[10], 'Cherbourg': isCherb, 'Queenstown': isQueens,'Southampton': isSoutham })
clean_training_data_extra_fields('/kaggle/input/train.csv', '/kaggle/input/clean_training_data_2.csv')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

titanic_file_path = './clean_training_data_2.csv'

titanic_train_data = pd.read_csv(titanic_file_path)

y = titanic_train_data.Survived

titanic_features = ['PassengerId','Male','Female', 'Pclass','Cherbourg','Queenstown','Southampton']

x = titanic_train_data[titanic_features]


train_X1, val_X1, train_y1, val_y1 = train_test_split(x, y, random_state = 0)


titanicModel = DecisionTreeRegressor()
titanicModel.fit(train_X1, train_y1)

val_predictions1 = titanicModel.predict(val_X1)
mean_absolute_error (val_y1, val_predictions1)

# Link to run notebook
# https://cocalc.com/projects/7b5aa3b0-0afe-49c1-901d-feac42f96c76/files/TitanicPrediction.ipynb?session=default