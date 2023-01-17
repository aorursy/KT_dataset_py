import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#---- Input data files are available in the "../input/" directory.
from subprocess import check_output
print("Available files are:\n",check_output(["ls", "../input"]).decode("utf8"))

#---- Get the gender class model csv
genderClassModel = pd.read_csv("../input/genderclassmodel.csv")
print("Gender class model data:\n",genderClassModel)

#---- Get the gender model csv
genderModel = pd.read_csv("../input/gendermodel.csv")
print("Gender model data:\n",genderModel)
#---- Are these two files the same?
print("(genderclassmodel == gendermodel) =",genderClassModel.equals(genderModel))
#---- Strange because the two file look the same
modelDifference = genderModel-genderClassModel
nSurvivalDiff = sum(modelDifference["Survived"] > 0)
print("There are",nSurvivalDiff,"different predictions")
#---- Ok so the survival data is different for the following passengers
print("Gender class model:\n",genderClassModel[modelDifference["Survived"] > 0])
print("Gender model\n",genderModel[modelDifference["Survived"] > 0])