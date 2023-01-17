import time, os

import numpy as np

import smtplib

import mimetypes

import requests, bs4, smtplib

from email.header import Header

from email.mime.multipart import MIMEMultipart

from email import encoders

from email.message import Message

from email.mime.audio import MIMEAudio

from email.mime.base import MIMEBase

from email.mime.image import MIMEImage

from email.mime.text import MIMEText

from email import encoders



#  Here, I just use TEST to finish running the notebook. Remove TEST to use it.

TEST = True
# Modified from https://github.com/jorgemgr94/python-smtplib-attachments



def sendattachmail(sender_address, sender_pass, receiver_mails, content, file_dir, num):

    # The mail addresses and password, in this example i'm using Gmail

    sender_address = sender_address

    sender_pass = sender_pass

    smtp_server = 'smtp.gmail.com'

    smtp_port = 587



    # Setup mail content

    mail_content = content



    # Setup global the MIME

    message = MIMEMultipart()

    message['Subject'] = 'Report ' + str(num) 



    # Body and the attachments for the mail

    message.attach(MIMEText(mail_content, 'plain'))

    attach_file_name = file_dir + f'/submission_{content}.csv'

    attach_file = open(attach_file_name, 'rb') # Open the file as binary mode

    payload = MIMEBase('application', 'octate-stream')

    payload.set_payload((attach_file).read())

    encoders.encode_base64(payload) # Encode the attachment

    payload.add_header('Content-Disposition', 'attachment', filename=f'submission_{content}.csv')

    message.attach(payload)



    #Create SMTP session 

    session = smtplib.SMTP(smtp_server, smtp_port)

    session.starttls() #enable security

    session.login(sender_address, sender_pass) #login

    text = message.as_string()



    # mails = ['xxxx@gmail.com', 'xxxx@gmail.com', ...]

    mails = receiver_mails



    for receiver_address in mails:

        # Setup the MIME

        message['From'] = sender_address

        message['To'] = receiver_address

        session.sendmail(sender_address, receiver_address, text)

        print('Mail Sent: '+receiver_address)

    

    session.quit()
if not TEST:

    

    # Your submission file path

    DIR = 'C:/Users/yirun_zhang008/Project/Santa/submission'



    # Check the initial best score

    score = []

    name = os.listdir(DIR)

    for item in name:

        score.append(float(item.split('_')[1].split('.c')[0]))

    best = np.min(score)

    print('Initial Best Score: ', best)



    i = 0

    while True:

        score = []

        name = os.listdir(DIR)

        for item in name:

            score.append(float(item.split('_')[1].split('.c')[0]))

        if np.min(score) < best:

            i += 1

            best = np.min(score)

            print(f'New Score: {best}')

            print('-'*40)

            sendattachmail(sender_address='xxxx@gmail.com', 

                           sender_pass='12345678', 

                           receiver_mails=['xxxx@gmail.com', 'xxxx@gmail.com'], 

                           content=str(best), 

                           file_dir=DIR, 

                           num=i)

        time.sleep(300) # check every 5 minutes