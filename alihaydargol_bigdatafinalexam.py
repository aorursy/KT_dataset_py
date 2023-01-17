import pandas as pd
from sklearn.model_selection import train_test_split



test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

train_t, test_t = train_test_split(train, test_size=0.2)
train.head()
test.head()
sales_train_data = train.drop(columns=['item_price','date'])
train_t = train_t.drop(columns=['item_price','date'])
test_t = test_t.drop(columns=['item_price','date'])
sales_train_data.head()
monthly=sales_train_data.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()
train_t = train_t.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()
test_t = test_t.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()
monthly.head()
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini')
predictor = monthly[['date_block_num','shop_id','item_id']]
target = monthly['item_cnt_day']

model_t = tree.DecisionTreeClassifier(criterion='gini')
train_t_predictor = train_t[['date_block_num','shop_id','item_id']]
train_t_target = train_t['item_cnt_day']
model_t.fit(train_t_predictor,train_t_target)
test_t_predictor = test_t[['date_block_num','shop_id','item_id']]
test_t_target = test_t['item_cnt_day']
from sklearn.metrics import accuracy_score

test_t_prediction = model_t.predict(test_t_predictor)
train_accuracy = accuracy_score(test_t_target, test_t_prediction)
print("Accuracy = " + str(train_accuracy))
import gc

del model_t
gc.collect()
model.fit(predictor, target)
test_predictor = test[['shop_id','item_id']]
test_predictor.insert(loc=1, column='date_block_num', value='34')

test_predictor.head()
predicted_count = model.predict(test_predictor)
predictedDF = pd.DataFrame(predicted_count)
predictedDF.columns  = ['item_cnt_month']
predictedDF.insert(loc=0, column='ID', value=test['ID'])
predictedDF.head()
predictedDF.to_csv("submission.csv", sep=',' , index=False)