!git clone https://github.com/huhuhang/mobilenet_v2-imagenet-api.git --depth=1 && rm -rf mobilenet_v2-imagenet-api/.git
import subprocess as sp



# 启动子进程执行 Flask app

server = sp.Popen("FLASK_APP=mobilenet_v2-imagenet-api/run.py flask run", shell=True)

server
import time

time.sleep(5) # 等待 5 秒保证子进程已完全启动
!curl -X POST -F image=@mobilenet_v2-imagenet-api/test.jpg 'http://127.0.0.1:5000/predict'