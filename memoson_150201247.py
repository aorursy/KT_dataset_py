import csv
def clearfile(file):
    open(file,"w",newline = "")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def linReg(sample_df):
    x = sample_df[["month"]][:-1]
    y = sample_df[["item_cnt_month"]][:-1]
    last_row = sample_df.iloc[-1]
    # split train/test
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.20,random_state = 99)
    # creating/fitting/testing model
    model = LinearRegression().fit(train_x,train_y)
    predict_test = model.predict(test_x)
    # test error
    test_error = mean_squared_error(test_y,predict_test)
    # print("test hatası : ",test_error*100)
    amount_November_2015 = model.predict(last_row.values.reshape(-1,1))
    return amount_November_2015[0], test_error
def update(dataFrame):
    updated = dataFrame.groupby(["date_block_num"]).agg("sum")
    updated = pandas.DataFrame(updated.reset_index("date_block_num"))
    updated.columns = ["month","item_cnt_month"]
    updated = extend_df(updated)
    return updated
def extend_df(df):
    must_be = list(range(0,35))
    current = df["month"].tolist()
    difference = list(set(must_be) - set(current))
    for i in difference:
        df = df.append({"month" : i,"item_cnt_month" : 0},ignore_index = True)
    df = df.sort_values(by = ["month"]).reset_index().drop(columns = ["index"])
    return df
def predict_future_sales(shop_id,item_id,train_df):
    train_crop = train_df.loc[(train_df['shop_id'] == shop_id) & (train_df['item_id'] == item_id)]
    train_crop = train_crop.drop(columns = ["shop_id","item_id","date","item_price"])
    updated_train_set = update(train_crop)
    november_2015_amount, error = linReg(updated_train_set)
    return november_2015_amount, error
import os
import pandas
# read train file into pandas dataframe
abs_path = "/kaggle/input/competitive-data-science-predict-future-sales"
train_file_path = os.path.join(abs_path,"sales_train.csv")
train_set = pandas.read_csv(train_file_path)
# read test file
test_file_path = os.path.join(abs_path,"test.csv")
test_set = pandas.read_csv(test_file_path)
import csv,os
# predict and write output file
output_file_path = os.path.join("../input/output/","/kaggle/working/submission.csv")
clearfile(output_file_path)
with open(output_file_path,"a",newline = "") as file:
    thewriter = csv.DictWriter(file,delimiter = ",",lineterminator='\n',fieldnames = ["ID","item_cnt_month"])
    thewriter.writeheader()
    sum_MSE = 0
    for i in range(len(test_set.index)):
        id = test_set.loc[i][0]
        shop_id = test_set.loc[i][1]
        item_id = test_set.loc[i][2]
        predicted_amount, MSE = predict_future_sales(shop_id,item_id,train_set)
        sum_MSE += MSE
        thewriter.writerow({"ID" : id,
                            "item_cnt_month" : "%.1f" % predicted_amount})
        print(id,"->","%.1f" % predicted_amount," MSE : ",MSE)
    print("sum MSE : ",sum_MSE," mean MSE : ",sum_MSE/len(test_set.index))
