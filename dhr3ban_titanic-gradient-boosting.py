#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/titanic/train.csv')
data
!pip install pycaret
#here our target is survived class
#we will ignore passenger id,name and ticket
#importing environment
from pycaret.classification import*
%matplotlib inline
clf1=setup(data,target='Survived',ignore_features=['Name','Ticket','PassengerId'])
#Verifying data
compare_models()
#here the top performing model light GBM
#tuning best performing models
tuned_lightgbm=tune_model('lightgbm',optimize='AUC')
#so the performance of light GBM improved by 2%
#now we will evaluate the trained model
evaluate_model(tuned_lightgbm)
#predict the data
#finalize model fits the model on entire dataset
final_lightgbm=finalize_model(tuned_lightgbm)
print(final_lightgbm)
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
#applying predictions to test dataset
predictions=predict_model(final_lightgbm,data=test)
predictions.head(50)
#the label column added has the survival predictions with accuracy score beside it