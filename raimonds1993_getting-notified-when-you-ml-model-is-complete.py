from IPython.display import Image

import os
display(Image("../input/mybotchat.jpg"))
display(Image("../input/botsearch.jpg"))
display(Image("../input/botfatherchat.jpg"))
!pip install knockknock
# Put your token as a string (be careful to not share you Token!)

your_token = "<your-token>"
import requests

import json



# Let's get your chat id! Be sure to have sent a message to your bot.



url = 'https://api.telegram.org/bot'+str(your_token)+'/getUpdates'

response = requests.get(url)

myinfo = response.json()

if response.status_code == 401:

  raise NameError('Check if your token is correct.')



try:  

  CHAT_ID: int = myinfo['result'][1]['message']['chat']['id']



  print('This is your Chat ID:', CHAT_ID)

  

except:

  print("Have you sent a message to your bot? Telegram's bots are quite shy 🤣.")
'''

from knockknock import telegram_sender



@telegram_sender(token=your_token, chat_id=CHAT_ID)

def train_your_nicest_model(your_nicest_parameters=''):

    # Here you can include your own model function and pass your parameters



    import time

    time.sleep(100)

    model_val_score = 'This is your model score'

    return {"Return something": model_val_score}

'''
# train_your_nicest_model()