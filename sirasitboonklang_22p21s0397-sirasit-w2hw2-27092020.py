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

import os # Create Directory
Train = pd.read_csv('/kaggle/input/titanic/train.csv',index_col="PassengerId")

Test = pd.read_csv('/kaggle/input/titanic/test.csv',index_col="PassengerId")

Submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv',index_col="PassengerId")
dtype={"Survived":np.int32,"Pclass":np.int32,"SibSp":np.int32,"Parch":np.int32,"Fare":np.float64}

Train.head()
Train.describe()
Train.Pclass = Train.Pclass.astype(int)

Train.Survived = Train.Survived.astype(int)
Test.head()
Test.describe()
Submission.head()
Train = Train.drop(["Name","Cabin","Ticket"], axis=1)

Test = Test.drop(["Name","Cabin","Ticket"], axis=1)
Train.Age = Train.Age.fillna(Train.Age.mean())

Test.Age = Test.Age.fillna(Test.Age.mean())
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
def OneHotEncode(DFTrain,DFTest,Columns):

    for Column in Columns:

        NewCol1 = pd.DataFrame(OHE.fit_transform(DFTrain[[Column]].replace(np.nan, 'N/A', regex=True)))

        NewCol2 = pd.DataFrame(OHE.fit_transform(DFTest[[Column]].replace(np.nan, 'N/A', regex=True)))

        

        NewCol1.index = DFTrain.index

        NewCol2.index = DFTest.index

        

        NewCol1 = NewCol1.rename(columns={x:f"{Column} {x}" for x in NewCol1.columns})

        NewCol2 = NewCol2.rename(columns={x:f"{Column} {x}" for x in NewCol2.columns})

        

        DFTrain = pd.concat([DFTrain.drop([Column], axis=1), NewCol1], axis=1)

        DFTest = pd.concat([DFTest.drop([Column], axis=1), NewCol2], axis=1)

    return DFTrain, DFTest
Train, Test = OneHotEncode(Train, Test,["Sex","Embarked"])
# DivideData From Question 1

def DivideData(Data,SetOfData=10):

    EmptySeries = pd.Series({x:None for x in Data.columns});

    while len(Data) % SetOfData != 0:

        Data = Data.append(EmptySeries,ignore_index = True)

    TmpData = np.array(Data).reshape(len(Data)//SetOfData,SetOfData,len(Data.columns))

    TenSetData = [pd.DataFrame(TmpData[:,i],columns=Data.columns).dropna(how='all') for i in range(SetOfData)]

    

    return TenSetData

# SimpleTTS From Question 1

def SimpleTTS(Data,TrainSize):

    TrainSize = round(TrainSize * len(Data))

    return Data.iloc[:TrainSize], Data.iloc[TrainSize:]
Train.describe()
Train.isnull().sum()
# Generate 5 Fold Cross Validation

FiveFoldTrain = DivideData(Train)
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

Models = [GaussianNB,DecisionTreeClassifier,MLPClassifier]

from sklearn.metrics import mean_absolute_error
def true(TrueY, PredY): return TrueY == PredY

def false(TrueY, PredY): return TrueY != PredY

def positive(PredY,Class=1): return PredY == Class

def negative(PredY,Class=1): return PredY != Class
def EvaluateModel(Data,idx):

    TrainX = Train[[x for x in Train.columns if x != "Survived"]]

    TrainY = Train['Survived']

    Results = [];

    for model in Models:

        TX, VX = SimpleTTS(TrainX, 0.9)

        TY, VY = SimpleTTS(TrainY.astype(int), 0.9)

        Model = model()

        Model.fit(TX,TY)

        Prediction = Model.predict(VX)

        Results.append((Model,Prediction,np.array(VY)))

        directory = f"{idx}/Model Prediction {Model}/".replace("(","").replace(")","")

        if not os.path.exists(directory):

            os.makedirs(directory)

        pd.DataFrame(VY).to_csv(directory+"Validation.csv")

        pd.DataFrame(Prediction, index=VX.index, columns=["Survived"]).to_csv(directory+"/Prediction.csv")

#         print(f"Model Prediction {Model} Prediction: {Prediction} True: {np.array(VY)}")

#         print("---------------------------------------------------------------------------------------------------------------------------------------")

    for i, (Model, PredVal, TrueVal) in enumerate(Results):

        PositiveClass = 0

        TP = sum( true(TrueVal, PredVal) & positive(PredVal,PositiveClass))

        TN = sum( true(TrueVal, PredVal) & negative(PredVal,PositiveClass))

        FP = sum(false(TrueVal, PredVal) & positive(PredVal,PositiveClass))

        FN = sum(false(TrueVal, PredVal) & negative(PredVal,PositiveClass))

        Precision = TP/(TP+FP);

        Recal = TP/(TP+FN);

        FMeasure1 = 2 * (Precision * Recal) / (Precision + Recal);

        print(f"""Result in {Model} (Index = {i})

    ------------------------------------------------------------------------------------------

    Class 0

    ---------------------------------------------

    True  Positive: {TP}

    True  Negative: {TN}

    False Positive: {FP}

    False Negative: {TN}

    ---------------------------------------------

    Precision: {Precision:.4f}

    Recal: {Recal:.4f}

    F-Measure: {FMeasure1:.4f}

    ------------------------------------------------------------------------------------------

    """);

        PositiveClass = 1

        TP = sum( true(TrueVal, PredVal) & positive(PredVal,PositiveClass))

        TN = sum( true(TrueVal, PredVal) & negative(PredVal,PositiveClass))

        FP = sum(false(TrueVal, PredVal) & positive(PredVal,PositiveClass))

        FN = sum(false(TrueVal, PredVal) & negative(PredVal,PositiveClass))

        Precision = TP/(TP+FP);

        Recal = TP/(TP+FN);

        FMeasure2 = 2 * (Precision * Recal) / (Precision + Recal);

        print(f"""Class 1

    ---------------------------------------------

    True  Positive: {TP}

    True  Negative: {TN}

    False Positive: {FP}

    False Negative: {TN}

    ---------------------------------------------

    Precision: {Precision:.4f}

    Recal: {Recal:.4f}

    F-Measure: {FMeasure2:.4f}

    ------------------------------------------------------------------------------------------""");

        print(f"Average F-Measure = {(FMeasure1+FMeasure2)/2:.4f}")

        print("---------------------------------------------------------------------------------------------------------------------------------------")
EvaluateModel(FiveFoldTrain[0],0)