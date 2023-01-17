!pip install sweetviz
!pip install pycaret
import sweetviz
import pandas as pd
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

train.head()
my_report = sweetviz.compare([train, "Train"], [test, "Test"], "SalePrice")
#it will create a html file in your working folder
my_report.show_html("my_report.html") # Not providing a filename will default to SWEETVIZ_REPORT.html
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from pycaret.regression import *

#intialize the setup
regression =  setup(data = train, target = 'SalePrice')
'''et = create_model('et')
catboost = create_model('catboost')
ada = create_model('ada')
ridge = create_model('ridge')
lightgbm = create_model('lightgbm')

# stack trained models
stacked_models = stack_models(estimator_list=[et,catboost,ada,ridge,lightgbm])

compare_models()'''
blend_models = blend_models(estimator_list = 'All',  fold = 10,  round = 4,  turbo = True, verbose = True)
final_model = finalize_model(blend_models)
pred = predict_model(final_model, data=test)
submission = pd.DataFrame({'Id': test_ID, 'SalePrice': pred['Label']})
submission.to_csv('submission.csv', index = False)
