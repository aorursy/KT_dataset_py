!pip install talon
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import all dependencies.

import email

import talon

from talon import quotations

import re
emails_df = pd.read_csv("/kaggle/input/enron-email-dataset/emails.csv")

len(emails_df)
emails_df.head()
def remove_forwarded_by(text):

    condition = "[- ]*Forwarded by[\S\s]*Subject:[\S\t ]*"

    return re.sub(condition, "", text).strip()
# Get only emails with the tag -----Original Message-----.

emails_df["message"] = emails_df.message.map(remove_forwarded_by)

emails_df = emails_df[emails_df["message"].str.contains("-----Original Message-----")]

emails_df.head()
len(emails_df)
emails = list(map(email.parser.Parser().parsestr, emails_df["message"]))
emails[6].keys()
emails[6].values()
# Get email body.

print(emails[6].get_payload())
talon.init()
def strip_body(text):

    condition = "[- ]*Original Message*[ -]*[\S\t ]*"

    text = re.sub(condition, "", text, 1).strip()

    text = text.replace(">","")

    return text.strip()





def deal_with_replies(body, last_reply):

    body_temp = body.replace(last_reply, "")

    body_temp = strip_body(body_temp)

    body_temp = email.message_from_string(body_temp)

    if body_temp.get_content_type() != 'text/plain':

        return

    body_temp = body_temp.get_payload() # remove From, Sent, To, Subject

    if body_temp.count("-----Original Message-----") > 0:

        original = quotations.extract_from_plain(body_temp).strip()

        if original:

            yield [original.strip(), last_reply.strip()]

        if original and body_temp:

            yield from deal_with_replies(body_temp, original)

    else:

        yield [body_temp.strip(), last_reply.strip()]
seq2seq_dataset = []



for i, e in enumerate(emails):

    text =  e.get_payload()

    last_reply = quotations.extract_from_plain(text.replace(">", ""))

    if last_reply: # ignore empty replies

        for pair in deal_with_replies(text, last_reply):

            seq2seq_dataset.append(pair)
seq2seq_df = pd.DataFrame(seq2seq_dataset, columns=["input_text", "target_text"])

seq2seq_df.head()
#seq2seq_df.to_csv("input-target.csv")