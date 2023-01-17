from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split

train = '../input/train.csv'
iowa_data = pd.read_csv(train)
#iowa_data.describe()
train_y = iowa_data.SalePrice
#c = iowa_data.columns 
columns = ['LotArea','YearBuilt','1stFlrSF', '2ndFlrSF','FullBath','BedroomAbvGr', 
       'TotRmsAbvGrd']
train_X = iowa_data[columns]
#train_X,train_y,val_X,val_y = train_test_split(X,y,random_state = 0)
iowa_model = RandomForestRegressor()
iowa_model.fit(train_X,train_y)
test = pd.read_csv('../input/test.csv')
test_X = test[columns]
iowa_predict = iowa_model.predict(test_X)
print(iowa_predict)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': iowa_predict})
my_submission.to_csv('submission.csv', index=False)