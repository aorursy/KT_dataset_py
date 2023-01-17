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
wine=pd.read_csv("/kaggle/input/wine-reviews/wine reviews.csv")
!pip install dora #Install and load Dora

from Dora import Dora

dora = Dora()  #Intialize dora
dora.configure(output = 'Reviews Rating', data = wine)

dora.data

#Remove unwanted features from the data set



dora.remove_feature('Sl.No.')

dora.data
#Perform one-hot encoding for categorical variables



dora.extract_ordinal_feature('Reviews do Recommend')

dora.data
#Transform and create new features



dora.extract_feature('Weight', 'Ltr', lambda x: x * 2)

dora.data
import seaborn as sns

iris=sns.load_dataset("iris")
dora.configure(output="petal_width",data=iris)

dora.remove_feature('species')

dora.data


# render plots of each feature in the data set against the output variable



dora.explore()
## plot a single feature against the output variable



dora.plot_feature('sepal_width')
#partition training / validation data (80/20 split)

dora.set_training_and_validation()



# Assign training data 

X = dora.training_data[dora.input_columns()]

y = dora.training_data[dora.output]

from sklearn.linear_model import LinearRegression

acc=LinearRegression().fit(X,y)
# validate the model

X = dora.validation_data[dora.input_columns()]

y = dora.validation_data[dora.output]

acc.score(X, y)