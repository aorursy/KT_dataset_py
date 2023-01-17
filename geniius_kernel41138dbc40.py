# -- coding: utf-8 --

from flask import Flask, request

import files.config



#####

# Бот



import urllib3

import telepot

from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

# import telebot

# from telebot import types



############################

# Proxy

# You can leave this bit out if you're using a paid PythonAnywhere account

proxy_url = "http://proxy.server:3128"



telepot.api._pools = {

    'default': urllib3.ProxyManager(proxy_url=proxy_url, num_pools=3, maxsize=100, retries=False, timeout=60),

}

telepot.api._onetime_pool_spec = (urllib3.ProxyManager, dict(proxy_url=proxy_url, num_pools=3, maxsize=100, retries=False, timeout=60))

# end of the stuff that's only needed for free accounts



###################

# для запуска Flask

from flask_sslify import SSLify



app = Flask(__name__)

sslify = SSLify(app)



####################



# TOKEN = files.config.token



# URL = 'https://api.telegram.org/bot{}/'.format(TOKEN)



fin_info = 'инфа1'

com_info = 'Инфа2'



secret = "A_SECRET_NUMBER"

bot = telepot.Bot(files.config.token)

bot.setWebhook("мой вебсайт".format(secret), max_connections=500)



button_phone = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="Нажмите СЮДА на провереку возможности использования сервиса", resize_keyboard=True, request_contact=True)]])





@app.route('/{}'.format(secret), methods=["POST"])

def telegram_webhook():

    update = request.get_json()

####Рабочий код проверки chat_id и ИНН в файле

    if "message" in update:

        chat_id = update["message"]["chat"]["id"]

        if 'contact' in update["message"]:

            phone = update["message"]['contact']['phone_number']

            # bot.sendMessage(chat_id, text=phone) #Есть телефон phone

            p = open('ccылка', "r")

            textPhone = p.read()

            bp = open('ccылка1', "r")

            Bossesphones = bp.read()

            if str(phone) in textPhone:

                bot.sendMessage(chat_id, text='Вам разрешен доступ к данному сервису. Для работы просто введите ИНН организации', reply_markup=None)

                with open('ccылка3', 'a', encoding='utf-8') as aChats:

                    aChats.write(str(chat_id) + '\n')

            elif str(phone) in Bossesphones:

                bot.sendMessage(chat_id, text='Вам разрешен доступ к данному сервису, Вы относитесь к группе Руководителей и можете получать информацию о счетах', reply_markup=None)

                with open('ccылка4', 'a', encoding='utf-8') as aBosses:

                    aBosses.write(str(chat_id) + '\n')

            else:

                bot.sendMessage(chat_id, text='Вам не разрешен доступ в данную группу. Если вы знаете того, кто может вас добавить - попросите его и проверьте ещё раз', reply_markup=button_phone)

        elif "text" in update["message"]:

            text = update["message"]["text"]

            fname = update["message"]["chat"]["first_name"]

            lname = update["message"]["chat"]["last_name"]

            a = open('/home/GeniusMan1/bot/files/aChats.txt', "r")

            savedaChats = a.read()

            abc = open('/home/GeniusMan1/bot/files/aBosses.txt')

            aBossesChats = abc.read()

            if str(chat_id) in savedaChats:

                if len(text) in (10, 12):

                    f = open("/home/GeniusMan1/bot/files/inn.txt", "r")

                    inn = f.read()

                    if text in inn:

                        bot.sendMessage(chat_id, "{} {}, если компания коммерческая - счёт открывать нельзя. \nЕсли некоммерческая - заводите".format(fname, lname))

                    else:

                        bot.sendMessage(chat_id, "{} {}, можно открывать счёт".format(fname, lname))

                else:

                    bot.sendMessage(chat_id, '{} {}, перепроверьте вводимые цифры. \nИНН должен быть строго 10 или 12 цифр'.format(fname, lname))



            if str(chat_id) in aBossesChats:

                if text == '/command2' or text == '/info' or text == '/com':

                    if text == '/command2' or text == '/info':

                        bot.sendMessage(chat_id, text=fin_info, reply_markup=None)

                    elif text == '/com':

                        bot.sendMessage(chat_id, text=com_info, reply_markup=None)

                elif len(text) in (10, 12):

                    f = open("ccылка5", "r")

                    inn = f.read()

                    if text in inn:

                        bot.sendMessage(chat_id, "{} {}, если компания коммерческая - счёт открывать нельзя. \nЕсли некоммерческая - заводите".format(fname, lname))

                    else:

                        bot.sendMessage(chat_id, "{} {}, можно открывать счёт".format(fname, lname))

                else:

                    bot.sendMessage(chat_id, '{} {}, перепроверьте вводимые цифры. \nИНН должен быть строго 10 или 12 цифр\nТолько для Вас доступна команда запроса информации /info или /com'.format(fname, lname))

            else:

                pass

                # bot.sendMessage(chat_id, text='Вам не разрешен доступ в данную группу', reply_markup=button_phone) #Чат id

        else:

            bot.sendMessage(chat_id, text='Не допустимый формат для ввода', reply_markup=button_phone) #Чат id

            bot.sendMessage(311926937, text=update, reply_markup=button_phone) #Чат id

    return "OK"

if __name__ == '__main__':

    app.run()