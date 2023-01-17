# plotting
import pandas as pd # pandas
import numpy as np # numpy

# preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# classifier
from sklearn.ensemble import RandomForestClassifier

# other
from sklearn.model_selection import train_test_split
# read csv
data_frame = pd.read_csv(
    filepath_or_buffer = "../input/StudentsPerformance.csv", # file path of csv
    header = 0, # header row
)
data_frame.head(10) # top 10 rows from csv
data_frame.isnull().sum() # checking missing values
# get writing, reading, and math scores for a separate data frame
ML_DataPoints = pd.read_csv(
    filepath_or_buffer = "../input/StudentsPerformance.csv", # file path of csv
    header = 0, # header row
    usecols = ['math score',
               'reading score',
               'writing score'] # data points columns
)
# get test preparation course values
ML_Labels = pd.read_csv(
    filepath_or_buffer = "../input/StudentsPerformance.csv", # file path of csv
    header = 0, # header row
    usecols = ['test preparation course'] # data points labels
)
# min max scaler
MNScaler = MinMaxScaler()
MNScaler.fit(ML_DataPoints) # fit math, reading, and writing scores
T_DataPoints = MNScaler.transform(ML_DataPoints) # transform the scores
# label encoder
LEncoder = LabelEncoder()
LEncoder.fit(ML_Labels) # fit labels
T_Labels = LEncoder.transform(ML_Labels) # transform the labels
# split train test data set
XTrain, XTest, YTrain, YTest = train_test_split(T_DataPoints, T_Labels, random_state = 10)
RandomForest = RandomForestClassifier(
    n_estimators = 10,
    random_state = 3
) # load the classifier
RandomForest.fit(XTrain, YTrain) # fit data points and labels
RandomForest.score(XTrain, YTrain)
RandomForest.score(XTest, YTest)
data_points= np.array ([
    [72, 72, 74], [90, 95, 93], [47, 57, 44], [76, 78, 75], [71,83, 78], #none --> 1
    [69, 90, 88], [88, 95, 92], [64, 64, 67], [78, 72, 70], [46, 42, 46] # completed --> 0

])
T_Points = MNScaler.transform(data_points)
RandomForest.predict(T_Points)