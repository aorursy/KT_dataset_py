#Predicting the category of crimes in San Francisco
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas.tools.rplot as rplot
from sklearn.ensemble import RandomForestClassifier
#Load the training data and test
dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")
#Resolution eliminate too many null data and be very significant for our study
#dftrain[dftrain.Resolution =='NONE']
#dftrain.Resolution.value_counts()
#Exploration data
dftrain.info()
dftrain.head()
dftrain.shape
dftrain.Category.value_counts()
#category = dftrain.pop("Category")
#category.describe()
dftrain.Category.value_counts().plot(kind = 'bar')
#Number of crimes per district
dftrain.PdDistrict.value_counts()
#Total number of suicides in a particular district
dftrain[(dftrain.Category =='SUICIDE') & (dftrain.PdDistrict =='SOUTHERN')].PdDistrict.value_counts()
fig, axs = plt.subplots(1,2)
dftrain[(dftrain.Category =='SUICIDE') ].PdDistrict.value_counts().plot(kind ='barh', ax=axs[0], title = 'Suicides')
dftrain[(dftrain.Category =='FAMILY OFFENSES')].PdDistrict.value_counts().plot(kind='bar', ax=axs[1], title = 'Family offenses', color = 'g')
dftrain.PdDistrict.value_counts().plot(kind = 'barh',title = 'Crimes by districts')
fig, axs = plt.subplots(1,2)
dftrain[dftrain.Category == 'DRUG/NARCOTIC'].PdDistrict.value_counts().plot(kind = 'bar', ax=axs[0], title = 'Drugs')
dftrain[dftrain.Category == 'ASSAULT'].PdDistrict.value_counts().plot(kind = 'bar', ax=axs[1],color = 'g', title = 'Assault')
#Specific category of crime district
dftrain[dftrain.PdDistrict == 'TENDERLOIN'].Category.value_counts()
dftrain[dftrain.PdDistrict == 'TENDERLOIN'].Category.value_counts().plot(kind = 'bar', title = 'Category of crimes in tenderloin')
dftrain[dftrain.Category == 'DRUG/NARCOTIC'].PdDistrict.value_counts().plot(kind = 'barh', title = 'Drugs in districts')