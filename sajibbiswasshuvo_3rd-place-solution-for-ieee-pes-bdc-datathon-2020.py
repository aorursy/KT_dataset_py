import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/train.csv')
df.head()
testdf1 = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/test.csv')
testdf1.head()
testdf = testdf1.drop(columns =['ID'])
testdf.head()
Y = df['global_horizontal_irradiance']
Y.head()
X = df.drop(columns =['global_horizontal_irradiance', 'ID'])
X.head()
from sklearn.ensemble import RandomForestRegressor

#just for ignoring some unimportant warnings

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
model1 = RandomForestRegressor(verbose = True ,n_estimators=5, n_jobs = -1, oob_score=True, random_state=100) # use n_estimators = 500
model1.fit(X,Y)
predicted_values1 = model1.predict(testdf)
Output1 = pd.DataFrame()
Output1["predicted_values1"] = predicted_values1
Output1.to_csv("Output1.csv", index = False)
Output1.head()
import os
os._exit(00)
import pandas as pd
df = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/train.csv')
df.head()
testdf1 = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/test.csv')
testdf1.head()
testdf = testdf1.drop(columns =['ID'])
testdf.head()
Y = df['global_horizontal_irradiance']
Y.head()
X = df.drop(columns =['global_horizontal_irradiance', 'ID'])
X.head()
#importing the model

from sklearn.ensemble import RandomForestRegressor

#just for ignoring some unimportant warnings

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#creating the model and fitting it to the data
model2 = RandomForestRegressor(verbose = True,n_estimators=6,n_jobs = -1, oob_score=True, random_state=200) # use n_estimators = 600
model2.fit(X,Y)
#predicting the output
predicted_values2 = model2.predict(testdf)

#saving the predictions to a csv file for averaging it later with other outputs 
Output2 = pd.DataFrame()
Output2["predicted_values2"] = predicted_values2
Output2.to_csv("Output2.csv", index = False)
Output2.head()
import os
os._exit(00)
import pandas as pd
df = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/train.csv')
df.head()
testdf1 = pd.read_csv('../input/ieee-pes-bdc-datathon-year-2020/test.csv')
testdf1.head()
testdf = testdf1.drop(columns =['ID'])
testdf.head()
Y = df['global_horizontal_irradiance']
Y.head()
X = df.drop(columns =['global_horizontal_irradiance', 'ID'])
X.head()
#importing the model

from sklearn.ensemble import RandomForestRegressor

#just for ignoring some unimportant warnings

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#creating the model and fitting it to the data
model3 = RandomForestRegressor(verbose = True,n_estimators=7,n_jobs = -1, oob_score=True, random_state=300) #Use n_estimators = 700
model3.fit(X,Y)
#predicting the output
predicted_values3 = model3.predict(testdf)
#saving the predictions to a csv file for averaging it later with other outputs 
Output3 = pd.DataFrame()
Output3["predicted_values3"] = predicted_values3
Output3.to_csv("Output3.csv", index = False)
Output3.head()
import os
os._exit(00)
import pandas as pd
predict1 = pd.read_csv("./Output1.csv")
predict2 = pd.read_csv("./Output2.csv")
predict3 = pd.read_csv("./Output3.csv")
avgPredict = (predict1.iloc[:,0]+ predict2.iloc[:,0] + predict3.iloc[:,0])/3
avgPredict
finalOutput = pd.DataFrame()
testdf = pd.read_csv("../input/ieee-pes-bdc-datathon-year-2020/test.csv")
finalOutput["ID"] = testdf.iloc[:,0]
finalOutput["global_horizontal_irradiance"] = avgPredict
finalOutput.to_csv("finalOutput.csv", index = False)
finalOutput