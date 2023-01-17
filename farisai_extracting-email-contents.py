# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
emailsdf = pd.read_csv("/kaggle/input/enron-email-dataset/emails.csv")

print(type(emailsdf))
emailsdf.head()
emailsdf['message'][1]
# This is what each email has the first 14 lines are metadata then the email body in the following lies

[s.strip() for s in emailsdf['message'][1].splitlines()]
' '.join([s.strip() for s in emailsdf['message'][1].splitlines()][15:])
def cleanemail(email):

    return ' '.join([s.strip() for s in email.splitlines()][15:])



emailsdf['email_body'] = emailsdf['message'].apply(cleanemail)



emailsdf.head()
# I need to export them for personal purpose

#emailsdf.to_csv('clean emails.csv')
#export sample of first 500 rows

emailsdf.iloc[:500].to_csv('sample clean emails.csv')