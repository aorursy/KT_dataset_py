# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sun Feb 24 09:55:57 2019

@author: pandit

"""



class PasswordValidator:

    def __init__(self,passwd=None):

        self.passwd=passwd

    def pass_check(self):

        Specialsymbol=['$','#','@']

        return_val=True

        

        if len(passwd) < 6:

            print("The password should have atleast 6 letters")

            return_val=False

        if len(passwd)>12:

            print("The password should not be greater than 12 letters")

            return_val=False

        if not any(char.isdigit() for char in passwd):

            print("The password should have atleast one digit")

            return_val=False

        if not any(char.isupper() for char in passwd):

            print("The password should have atleast one uppercase letter")

            return_val=False

        if not any(char.islower() for char in passwd):

            print("The password should have atleast one lowercase letter")

            return_val=False

        if not any(char in Specialsymbol for char in passwd):

            print("The password should have atleast one special symbol")

            return_val=False

        if return_val:

            print("You entered password is valid")

            return return_val

        else:

            return False

        

if __name__ == "__main__":

    passwd=input("Enter your valid password:")

    obj=PasswordValidator(passwd)

    

    while not obj.pass_check():

        passwd=input("Enter your valid password: ")

        obj.pass_check()