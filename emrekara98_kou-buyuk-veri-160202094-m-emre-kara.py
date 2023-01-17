import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
def create_folders():
    directories = []
    
    data_dir = 'data'
    directories.append(data_dir)
    
    for directory in directories: 
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory, 'succesfully created.')
        else:
            print(directory, 'already exists.')
def get_category_dict(item_categories): #items.csv to python dict
    item_category_dict = {}
    keys = item_categories['item_id']
    values = item_categories['item_category_id']
    item_category_dict = dict(zip(keys, values))
    return item_category_dict

def add_item_category_id_by_item_id(data):
    item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
    cat_dict = get_category_dict(item_categories)
    all_item_categories = []
    for item_id in data['item_id']:
        all_item_categories.append(cat_dict[item_id])
    data['item_category_id'] = all_item_categories
    return data
    
def write_csv_predicting_item_price_train():
    data = pd.read_csv('data/month_based_salesTrainData.csv')
    print('\nwrite_csv_predicting_item_price_train PRE:\n',data.head(5))
        
    data = data.drop(['item_cnt_day'],axis=1)
    data = add_item_category_id_by_item_id(data)
    data.to_csv('data/predicting_item_price_train.csv', index = False)
    
    print('\nwrite_csv_predicting_item_price_train POST:\n',data.head(5))
    
def write_csv_predicting_item_price_test():
    data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
    print('\nwrite_csv_predicting_item_price_test PRE:\n',data.head(5))
    
    data['date_block_num'] = 34
    data = add_item_category_id_by_item_id(data)
    data.to_csv('data/predicting_item_price_test.csv', index = False)
    
    print('\nwrite_csv_predicting_item_price_test POST:\n',data.head(5))
    
def write_csv_predicting_item_cnt_day_train():
    data = pd.read_csv('data/month_based_salesTrainData.csv')
    print('\nwrite_csv_predicting_item_price_test PRE:\n',data.head(5))
    
    data = add_item_category_id_by_item_id(data)
    data.to_csv('data/predicting_item_cnt_day_train.csv', index = False)
    
    print('\nwrite_csv_predicting_item_price_test POST:\n',data.head(5))
    
def write_csv_month_based_salesTrainData():
    
    data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
    print('\nwrite_csv_month_based_salesTrainData PRE:\n', data.head(5))
    
    data = data[data['item_cnt_day'] > 0]
    new_data = pd.DataFrame()
    uniques_month_nums = data['date_block_num'].unique()
    for month_num in uniques_month_nums:
        month = data[data['date_block_num'] == month_num]
        agg_overlappeds = month.groupby(['item_id','shop_id'], as_index = False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})
        agg_overlappeds['date_block_num'] = month_num
        new_data = pd.concat([new_data,agg_overlappeds])
    new_data.to_csv('data/month_based_salesTrainData.csv',index=False)
    
    print('\nwrite_csv_month_based_salesTrainData POST:\n', new_data.head(5))

def prepare_datas():
    write_csv_month_based_salesTrainData()
    write_csv_predicting_item_price_train()
    write_csv_predicting_item_price_test()
    write_csv_predicting_item_cnt_day_train()
    print('All the datas are prepared')

create_folders()
prepare_datas()
class DTR:
    def __init__(self):
        pass
    def get_accuracy_for_predicting_item_price(self):
        data = pd.read_csv('data/predicting_item_price_train.csv')
        data = data.astype({'item_price':str})
        x = data.drop(['item_price'], axis=1).to_numpy()
        y = data['item_price'].to_numpy()


        reg_decT = DecisionTreeRegressor(random_state=0)

        x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=10, test_size=0.2)

        reg_decT.fit(x_train, y_train)
        y_pred = reg_decT.predict(x_test)
        y_pred = y_pred.tolist()
        r2_score = metrics.r2_score(y_test, y_pred)
        print('Predicting item_price r_2 score:', r2_score)

    def get_accuracy_for_predicting_item_cnt_day(self):
        data = pd.read_csv('data/predicting_item_cnt_day_train.csv')
        
        data = data.astype({'item_cnt_day':str})
        x = data.drop(['item_cnt_day'], axis=1).to_numpy()
        y = data['item_cnt_day'].to_numpy()


        reg_decT = DecisionTreeRegressor(random_state=0)

        x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=10, test_size=0.2)
        reg_decT.fit(x_train, y_train)
        y_pred = reg_decT.predict(x_test)
        y_pred = y_pred.tolist()
        r2_score = metrics.r2_score(y_test, y_pred)
        print('Predicting item_cnt_day r_2 score:', r2_score)
        
    def predict_item_price_on_testData(self):
        data_train = pd.read_csv('data/predicting_item_price_train.csv')
        print('\npredict_item_price_on_testData PRE:\n', data_train.head(5))
        
        data_train = data_train.astype({'item_price':str})
        x_train = data_train.drop(['item_price'], axis=1).to_numpy()
        y_train = data_train['item_price'].to_numpy()

        data_test = pd.read_csv('data/predicting_item_price_test.csv')
        x_test = data_test.drop(['ID'], axis=1).to_numpy()

        reg_decT = DecisionTreeRegressor(random_state=0)

        reg_decT.fit(x_train, y_train)

        y_pred = reg_decT.predict(x_test)
        y_pred = y_pred.tolist()

        data_test['item_price'] = y_pred
        data_test.to_csv('data/testData_with_item_price.csv', index=False)
        print('testData_with_item_price.csv succesfully created.')
        
        print('\npredict_item_price_on_testData POST:\n', data_test.head(5))
dtr = DTR()
dtr.get_accuracy_for_predicting_item_price()
dtr.predict_item_price_on_testData()

dtr.get_accuracy_for_predicting_item_cnt_day()
class RFR:
    def __init__(self):
        pass
    
    def get_accuracy_for_predicting_item_cnt_day(self):
        data = pd.read_csv('data/predicting_item_cnt_day_train.csv')
        data = data.astype({'item_cnt_day':str})
        x = data.drop(['item_cnt_day'], axis=1).to_numpy()
        y = data['item_cnt_day'].to_numpy()


        rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

        x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=10, test_size=0.2)

        rf_reg.fit(x_train, y_train)
        y_pred = rf_reg.predict(x_test)
        y_pred = y_pred.tolist()
        r2_score = metrics.r2_score(y_test, y_pred)
        print('r_2 score:', r2_score)
        
    def predict_item_cnt_day_on_testData_with_item_price(self):
        data_train = pd.read_csv('data/predicting_item_cnt_day_train.csv')
        print('\npredict_item_cnt_day_on_testData_with_item_price PRE:\n', data_train.head(5))
        
        x_train = data_train.drop(['item_cnt_day'], axis=1).to_numpy()
        y_train = data_train['item_cnt_day'].to_numpy()

        data_test = pd.read_csv('data/testData_with_item_price.csv')
        x_test = data_test.drop(['ID'], axis=1).to_numpy()

        rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

        rf_reg.fit(x_train, y_train)

        y_pred = rf_reg.predict(x_test)
        y_pred = y_pred.tolist()

        data_test['item_cnt_month'] = y_pred
        data_test.to_csv('data/testData_with_item_price_and_item_cnt_day.csv', index=False)
        print('\npredict_item_cnt_day_on_testData_with_item_price POST:\n', data_test.head(5))
        
rfr = RFR()
rfr.get_accuracy_for_predicting_item_cnt_day()
rfr.predict_item_cnt_day_on_testData_with_item_price()
def submission_final():
    data = pd.read_csv('data/testData_with_item_price_and_item_cnt_day.csv')
    data = data.loc[:,['ID', 'item_cnt_month']]
    data['item_cnt_month'] = data['item_cnt_month'].clip(0,20)
    data.to_csv('submission.csv', index=False)
    print('submission.csv succesfully created.')
    print('submission data summary:\n', data.iloc[:,1:].head(15))
submission_final()