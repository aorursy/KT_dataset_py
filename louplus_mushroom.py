import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1321/mushrooms.csv")

df.tail()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import joblib



X = pd.get_dummies(df.iloc[:, 1:])  # 读取特征并独热编码

y = df['class']  # 目标值



model = RandomForestClassifier()  # 随机森林

print(cross_val_score(model, X, y, cv=5).mean())  # 交叉验证结果



model.fit(X, y)  # 训练模型

joblib.dump(model, "mushrooms.pkl")  # 保存模型

print("model saved.")
%%writefile predict.py

# 将此单元格代码写入 predict.py 文件方便后面执行

import joblib

import pandas as pd

from flask import Flask, request, jsonify



app = Flask(__name__)



@app.route("/", methods=["POST"])  # 请求方法为 POST

def inference():

    query_df = pd.DataFrame(request.json)  # 将 JSON 变为 DataFrame

    

    df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1321/mushrooms.csv")  # 读取数据

    X = pd.get_dummies(df.iloc[:, 1:])  # 读取特征并独热编码

    query = pd.get_dummies(query_df).reindex(columns=X.columns, fill_value=0)  # 将请求数据 DataFrame 处理成独热编码样式

    

    clf = joblib.load('mushrooms.pkl')  # 加载模型

    prediction = clf.predict(query)  # 模型推理

    return jsonify({"prediction": list(prediction)})  # 返回推理结果
import time

import subprocess as sp



# 启动子进程执行 Flask app

server = sp.Popen("FLASK_APP=predict.py flask run", shell=True)

time.sleep(5)  # 等待 5 秒保证 Flask 启动成功

server
import json



# 从测试数据中取 1 条用于测试推理

df_test = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1321/mushrooms_test.csv")

sample_data = df.sample(1).to_json(orient='records')

sample_json = json.loads(sample_data)

sample_json
import json

import requests



requests.post(url="http://localhost:5000", json=sample_json).content  # 建立 POST 请求，并发送数据请求
server.terminate()  # 结束子进程，关闭端口占用