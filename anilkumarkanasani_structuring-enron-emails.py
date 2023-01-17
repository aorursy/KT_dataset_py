# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import clear_output  # To clear the output of jupter notebook cell
! pip install mail-parser
import mailparser  # Wrapper package around basic python package (email)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_original = pd.read_csv('/kaggle/input/enron-email-dataset/emails.csv')
df_original.head()
df_original['person'] = df_original['file'].apply(lambda x : x.split('/')[0])
df_original['Mail_Category'] = df_original['file'].apply(lambda x : x.split('/')[1:-1])
df_original['num'] = df_original['file'].apply(lambda x : int(x.split('/')[-1].replace('.' , '')))
df_original.drop(columns=['file'] , inplace = True)
df_original.head()
df_temp = df_original.drop(columns=['message'])
df_temp.to_pickle("/kaggle/working/File_column.pkl")
df_temp.head()
intermediate_stage = list((np.arange(25000,len(df_original),25000)))
intermediate_stage.append (len(df_original))
print(intermediate_stage)
# Saving the same original data frame into another data frame (df) 
df = df_original

# Creating an empty data frame to save the output
df_extracted = pd.DataFrame()

record = 0

# iterating through each message in the data frame
for mail_string in df['message']:    
    record = record + 1
    
    # temp_dict is to save the extracted information in each iteration and to append to output dataframe
    temp_dict = {}
    
    # mailparser.parse_from_string is taking a string as input and yield mail parser object as output
    mail = mailparser.parse_from_string(mail_string)
    # Example output is : <mailparser.mailparser.MailParser at 0x7fa98c0ad690>
    
    # mail parser object.headers will yield all the header information of the mail as dictonary
    temp_dict = mail.headers
    
    # Example mail.header is as follows
    '''{'Message-ID': '<18782981.1075855378110.JavaMail.evans@thyme>',
         'Date': 'Mon, 14 May 2001 16:39:00 -0700 (PDT)',
         'From': 'phillip.allen@enron.com',
         'To': 'tim.belden@enron.com',
         'Subject': '',
         'Mime-Version': '1.0',
         'Content-Type': 'text/plain; charset=us-ascii',
         'Content-Transfer-Encoding': '7bit',
         'X-From': 'Phillip K Allen',
         'X-To': 'Tim Belden <Tim Belden/Enron@EnronXGate>',
         'X-cc': '',
         'X-bcc': '',
         'X-Folder': "\\Phillip_Allen_Jan2002_1\\Allen, Phillip K.\\'Sent Mail",
         'X-Origin': 'Allen-P',
         'X-FileName': 'pallen (Non-Privileged).pst'}'''
    
    # mail.body will yield the content of the email as string
    temp_dict ['Mail_Body'] = mail.body
    
    
    # Example mail.body is as follows
    '''"Traveling to have a business meeting takes the fun out of the trip.  Especially if you have to prepare a presentation.  
    I would suggest holding the business plan meetings here then take a trip without any formal business meetings.  
    I would even try and get some honest opinions on whether a trip is even desired or necessary.
    \n\nAs far as the business meetings, I think it would be more productive to try and stimulate discussions across the different groups about what is working and what is not.  
    Too often the presenter speaks and the others are quiet just waiting for their turn.   The meetings might be better if held in a round table discussion format.  
    \n\nMy suggestion for where to go is Austin.  Play golf and rent a ski boat and jet ski's.  Flying somewhere takes too much time."
    '''
    
    # preparing proper dictonary to append to output data frame
    temp = {'record_' + str(record): list(temp_dict.values()) }
    
    # if it is first record, the temp dictonary will be convert to output_df
    # if not, temp dictonary will be append to output_df
    if record != 1:
        df_extracted = df_extracted.append (pd.DataFrame.from_dict(temp, orient='index', columns = list(temp_dict.keys())))
    else:
        df_extracted = pd.DataFrame.from_dict(temp, orient='index', columns = list(temp_dict.keys()))
        
    # clearing output and printing iteration number is only to see the present iteration number    
    #clear_output()
    print(record)
    
    # For every 25000 records one pickle will be yielded and saved to temp folder
    if record in intermediate_stage:
        df_extracted.to_pickle("/kaggle/working/Messages_till_"+str(record)+".pkl")
        df_extracted = pd.DataFrame()
df_output = pd.DataFrame()

for record in intermediate_stage:
    pic_file = "/kaggle/working/Messages_till_"+str(record)+".pkl"
    df_temp = pd.read_pickle(pic_file)
    df_output = df_output.append(df_temp)
    os.remove (pic_file)
    print('File removed --- ' + str(pic_file) )

df_output.to_pickle("/kaggle/working/All_Messages.pkl")