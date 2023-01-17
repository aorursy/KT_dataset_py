# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filePaths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        print(path)

        filePaths.append(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

def viewInfo(table, name="Table"):

    col = table.index.tolist()

    row = table.columns.tolist()

    colLenght = len(col)

    rowLenght = len(row)

    print('[------{0}------]\n[*] Row : {1} | Col {2}'.format(name, rowLenght, colLenght))

    print(row, '\n')
import pandas as pd

customer = pd.read_csv(filePaths[0])

uselog_months = pd.read_csv(filePaths[2])

viewInfo(customer, "Customer")

viewInfo(uselog_months, "Uselog Months")
year_months = list(uselog_months["年月"].unique())

print("[+] Userlog months :", len(uselog_months["年月"].unique()), "\n", year_months, "\n")

uselog = pd.DataFrame()

for i in range(1, len(year_months)):

    tmp = uselog_months.loc[uselog_months["年月"]==year_months[i]]

    tmp.rename(columns={"count":"count_0"}, inplace=True)

    tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-1]]

    del tmp_before["年月"]

    tmp_before.rename(columns={"count":"count_1"}, inplace=True)

    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")

    print(i, tmp.shape)

    uselog = pd.concat([uselog, tmp], ignore_index=True)

print("\n")

viewInfo(uselog, "Use Log")
from dateutil.relativedelta import relativedelta

exit_customer = pd.DataFrame()

print("[0] All:", customer.shape)

exit_customer = customer.loc[customer["is_deleted"]==1]

print("[1] is_deleted:", exit_customer.shape)

exit_customer["exit_date"] = None

print("[2] Add col exit_date:", exit_customer.shape)

print("[3] to_datetime:", exit_customer["end_date"].iloc[0], "-->", end=" ")

exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

print(exit_customer["end_date"].iloc[0])

print("[4] Exit date = End_Date - 1 months\n###", exit_customer["exit_date"].iloc[0], '-->', end=' ')

for i in range(len(exit_customer)):

    exit_customer["exit_date"].iloc[i] = exit_customer["end_date"].iloc[i] - relativedelta(months=1)

print(exit_customer["exit_date"].iloc[0])

exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")

uselog["年月"] = uselog["年月"].astype(str)

viewInfo(uselog, "Uselog")

viewInfo(exit_customer, "Exit cus")

exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "年月"], how="left")

print("[Merge]")

viewInfo(exit_uselog, "Exit Uselog")

exit_uselog.head()
print("[0] Exit_uselog :", exit_uselog.shape)

exit_uselog = exit_uselog.dropna(subset=["name"])

print("[Drop] ", exit_uselog.shape)

print(len(exit_uselog["customer_id"].unique()))

exit_uselog.head()
conti_customer = pd.DataFrame()

conti_customer = customer.loc[customer["is_deleted"]==0]

conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")

conti_uselog = conti_uselog.dropna(subset=["name"])

print("[Info] ", conti_uselog.shape)

conti_uselog.head()
# Xáo chộn các hàng, reset index

conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)

print(conti_uselog.head(2))

# drop duplicates trong cột 'customer_id'

print("[Conti] ", conti_uselog.shape)

conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")

print("[Drop] ", conti_uselog.shape)

conti_uselog.head()
print("[+] Conti :", conti_uselog.shape)

print("[+] Exit :", exit_uselog.shape)

predict_data = pd.concat([conti_uselog, exit_uselog],ignore_index=True)

print("[*] Predict :", predict_data.shape)

predict_data.head()
# Thêm cột period

print("[+] Predict :", predict_data.shape)

predict_data["period"] = 0

# Thêm cột now_date

predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")

predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

for i in range(len(predict_data)):

    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])

    predict_data["period"][i] = int(delta.years*12 + delta.months)

print("[+] Add two rows 'now_date' and 'period'")

print("[+] Predict :", predict_data.shape)

predict_data.head()
predict_data.isna().sum()
predict_data = predict_data.dropna(subset=["count_1"])

predict_data.isna().sum()
target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]

predict_data_1 = predict_data[target_col]

predict_data_1.head()
predict_data_2 = pd.get_dummies(predict_data_1)

predict_data_2.head()

# one hot row
_model_demo = predict_data_2

viewInfo(_model_demo, "Data")
# Training model
from sklearn.tree import DecisionTreeClassifier

import sklearn.model_selection



exit = _model_demo.loc[_model_demo["is_deleted"]==1]

conti = _model_demo.loc[_model_demo["is_deleted"]==0].sample(len(exit))

X = pd.DataFrame()

y = pd.DataFrame()

X = pd.concat([exit, conti], ignore_index=True)

y = X["is_deleted"]

del X["is_deleted"]

# Chia data

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)



# Khởi tạo model

model = DecisionTreeClassifier(random_state=0)

# Train model

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

print("[+] Predict output :", str(len(y_test_pred)))
results_test = pd.DataFrame({"y_test":y_test ,"y_pred":y_test_pred })

results_test.head()
correct = len(results_test.loc[results_test["y_test"]==results_test["y_pred"]])

print("[+] Correct :", str(correct))

data_count = len(results_test)

print("[+] All :", str(data_count))

score_test = correct / data_count

print("[Score] ", str(score_test))
print(model.score(X_test, y_test))

print(model.score(X_train, y_train))
X = pd.concat([exit, conti], ignore_index=True)

y = X["is_deleted"]

del X["is_deleted"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)



model = DecisionTreeClassifier(random_state=0, max_depth=5)

model.fit(X_train, y_train)

print("[+] Test :", str(model.score(X_test, y_test)))

print("[+] Train :", str(model.score(X_train, y_train)))
importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})

importance
count_1 = 3

routing_flg = 1

period = 10

campaign_name = "入会費無料"

class_name = "オールタイム"

gender = "M"
if campaign_name == "入会費半額":

    campaign_name_list = [1, 0, 0]

elif campaign_name == "入会費無料":

    campaign_name_list = [0, 1, 0]

elif campaign_name == "通常":

    campaign_name_list = [0, 0, 1]

if class_name == "オールタイム":

    class_name_list = [1, 0, 0]

elif campaign_name == "デイタイム":

    campaign_name_list = [0, 1, 0]

elif campaign_name == "ナイト":

    campaign_name_list = [0, 0, 1]

if gender == "F":

    gender_list = [1, 0]

elif gender == "M":

    gender_list = [0, 1]

input_data = [count_1, routing_flg, period]

input_data.extend(campaign_name_list)

input_data.extend(class_name_list)

input_data.extend(gender_list)
print("[+] Dự đoán :", str(model.predict([input_data])))

# Fully connect

print("[+] Fully connect :", str(model.predict_proba([input_data])))