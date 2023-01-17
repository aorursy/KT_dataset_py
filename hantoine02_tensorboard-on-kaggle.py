# Clear any logs from previous runs

!rm -rf ./logs/ 

!mkdir ./logs/
# From Github Gist: https://gist.github.com/hantoine/4e7c5bc6748861968e61e60bab89e9b0

from urllib.request import urlopen

from io import BytesIO

from zipfile import ZipFile

from subprocess import Popen

from os import chmod

from os.path import isfile

import json

import time

import psutil



def launch_tensorboard():

    tb_process, ngrok_process = None, None

    

    # Launch TensorBoard

    if not is_process_running('tensorboard'):

        tb_command = 'tensorboard --logdir ./logs/ --host 0.0.0.0 --port 6006'

        tb_process = run_cmd_async_unsafe(tb_command)

    

    # Install ngrok

    if not isfile('./ngrok'):

        ngrok_url = 'https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip'

        download_and_unzip(ngrok_url)

        chmod('./ngrok', 0o755)



    # Create ngrok tunnel and print its public URL

    if not is_process_running('ngrok'):

        ngrok_process = run_cmd_async_unsafe('./ngrok http 6006')

        time.sleep(1) # Waiting for ngrok to start the tunnel

    ngrok_api_res = urlopen('http://127.0.0.1:4040/api/tunnels', timeout=10)

    ngrok_api_res = json.load(ngrok_api_res)

    assert len(ngrok_api_res['tunnels']) > 0, 'ngrok tunnel not found'

    tb_public_url = ngrok_api_res['tunnels'][0]['public_url']

    print(f'TensorBoard URL: {tb_public_url}')



    return tb_process, ngrok_process





def download_and_unzip(url, extract_to='.'):

    http_response = urlopen(url)

    zipfile = ZipFile(BytesIO(http_response.read()))

    zipfile.extractall(path=extract_to)





def run_cmd_async_unsafe(cmd):

    return Popen(cmd, shell=True)





def is_process_running(process_name):

    running_process_names = (proc.name() for proc in psutil.process_iter())

    return process_name in running_process_names





tb_process, ngrok_process = launch_tensorboard()
import tensorflow as tf



mnist = tf.keras.datasets.mnist



(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0



def create_model():

    return tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28, 28)),

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(10, activation='softmax')

    ])
import datetime

model = create_model()

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



model.fit(x=x_train, 

          y=y_train, 

          epochs=10, 

          validation_data=(x_test, y_test), 

          callbacks=[tensorboard_callback])