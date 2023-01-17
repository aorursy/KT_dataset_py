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
indians = pd.read_csv("/kaggle/input/pima-indians-diabetes/pima-indians-diabetes.data.csv")



indians.head(5)
#Column name based

indians["triceps_sf_thickness"]



#Using .loc

indians.loc[3]

indians.loc[1,["bmi","diabetes"]]



#Using .iloc

indians.iloc[-2]

indians.iloc[:,3]
example_one = indians.drop(columns = ["diastolic_blood_pressure", "bmi", "years_of_age"])

example_one = indians.drop([2])

example_one.head(3)
#Creating a new dataframe. Disregard unless you would like to know how to build your own dataframe, 

#for which you may contact one of the directors

new_data = {"tribe": ["Pima Indian","Pima Indian", "Pima Indian"], 

            "gender": ["Unknown", "Unknown", "Unknown"]}

frame_data = pd.DataFrame(new_data)



data = [indians, frame_data]



appended_data = pd.concat(data, axis = 1)



appended_data.head(5)