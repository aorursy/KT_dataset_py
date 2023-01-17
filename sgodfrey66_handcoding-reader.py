!pip install pygsheets
# Python core libraries

import os

import json



# Google API

import pygsheets

from google.oauth2 import service_account

import google.auth



# Data analysis

import pandas as pd

# import numpy as np



# Progress bars

from tqdm.notebook import tqdm



# Kaggle_secrets

from kaggle_secrets import UserSecretsClient

# Path to the Sheets credential files

# cred_path = '/Users/stephengodfreywork/Documents/Workbench/GoogleTools/pygsheets'

cred_path='../input'

cred_file = 'GoogleSheets-7f6318bca1f0.json'



# Path to the output data files

data_path=''



# Kaggle COVID-19: hand-coding for medical care task 

# hand_coding_file='1t2e3CHGxHJBiFgHeW0dfwtvCG4x0CDCzcTFX7yz9Z2E'



# Copy of Kaggle COVID-19: hand-coding for medical care task 

hand_coding_file_key='13hYI7pnz-PvWB4tds71a9GyDdCfLr9mheycm-zUvS60'



# Create a credential file from Kaggle secrets

# Not working

# user_secrets = UserSecretsClient()

# cred_keys=["type","project_id","private_key_id","private_key", "client_email", "client_id",

#            "auth_uri", "token_uri", "auth_provider_x509_cert_url","client_x509_cert_url"]

# custom_creds={key:user_secrets.get_secret(key) for key in cred_keys}

custom_creds=''

# Read input from Google Sheet    

class SheetWrapper:

    ''' 

    Using the pygsheets library, read from a Google Sheet and return a dataframe.

        

    Attributes   

    -------------------

    config_path:

        Path containing the authorization credentials    

    config_file:

        File containing authorization credentials

    custom_creds:

        A dictionary of service-file credentials

    api_key_file:

        A file with API key information (only needed for OAuth)

    pygs_obj:

        A pygsheets authorization object, to be set in the create_spreadsheet_object

    auth_method:

        Authorization method   

    sheet_key:

        A Sheet key (e.g. 13hYI7pnz-PvWB4tds71a9GyDdCfLr9mheycm-zUvS60) used to identify the sheet to get data

    sheet_updated:

        Last update time for sheet    

    sheet_wks:

        A list of all worksheet names in the sheet

    dfs_wks:

        Names of the worksheets that have been used to populate the dfs attribute; This can be set 

        at instantiation and can be a single worksheet name, a list of names or None; If it is set to 

        None, the dfs attribute will be populated with all worksheets after running read_from_sheet()

    dfs:

        A dataframe or list of dataframes from data in the worksheets   

        

    Methods

    -------------------

    create_spreadsheet_object(self):

        Create a pygsheets object

    def read_from_sheet(self):        

        Read data on worksheets and store in the dfs (dataframes) attribute

        

    '''

 

    # Path containing the authorization credentials

    config_path=''

    # File containing authorization credentials

    config_file=''

    # Dictionary of service-file credentials

    custom_creds=''

    # File containing API key information

    api_key_file=''

    # A pygsheets authorization object

    pygs_obj = None

    # Authorization method

    auth_method='service'

    # A Sheet key

    sheet_key=''

    # Last update time for sheet

    sheet_updated=''

    # A list of all worksheet names in the sheet      

    sheet_wks=None

    # Names of the worksheets that have been used to populate the dfs attribute 

    dfs_wks=None

    # A dataframe or list of dataframes from data in the worksheets

    dfs=None

    

    # Initialization

    def __init__(self,

                 config_path='',

                 config_file='',

                 custom_creds='',

                 api_key_file='',

                 auth_method='',

                 sheet_key='',

                 dfs_wks=None):

    

        # Update the object's parameters

        if config_path!='':

            self.config_path=config_path

        if config_file!='':

            self.config_file=config_file

        if custom_creds!='':

            # If dictionarty convert to JSON

            if isinstance(custom_creds,dict):

                self.custom_creds=json.dumps(custom_creds)               

            # If not, check if valid JSON and if so save JSON; if not error

            else:

                try:

                    self.custom_creds=json.dumps(json.loads(custom_creds))

                except:

                    raise TypeError(custom_creds+' can not form valid JSON')

        if api_key_file!='':

            self.api_key_file=api_key_file

        if auth_method!='':

            self.auth_method=auth_method            

        if sheet_key!='':

            self.sheet_key=sheet_key

        if dfs_wks!=None:

            self.dfs_wks=dfs_wks             

            

    

    # Create a pygsheets object

    def create_spreadsheet_object(self):

        # If the authentication method is OAuth; read API file and create object

        if self.auth_method.upper()=='OAUTH':

            # Read the API file contents

            api_json=json.loads(open(os.path.join(self.config_path,self.api_key_file)).read())



            # extract the api key

            self.api_key=api_json['ReconTools']['api_key']



            # Authorize pygsheets API

            self.pygs_obj=pygsheets.authorize(client_secret=os.path.join(self.config_path, self.config_file), 

                             kwargs={'key':self.api_key})



        # if the authentication method is a service file

        elif self.auth_method.upper()=='SERVICE':



            #Google Sheets API authorization

            self.pygs_obj=pygsheets.authorize(service_file = os.path.join(self.config_path, self.config_file))



        # if the authentication method is custom

        elif self.auth_method.upper()=='CUSTOM':

 

            # Google Sheets API authorization

            os.environ['GDRIVE_API_CREDENTIALS'] = self.custom_creds



            # test_pygs_obj=pygsheets.authorize(custom_credentials=custom_creds_json)

            self.pygs_obj=pygsheets.authorize(service_account_env_var='GDRIVE_API_CREDENTIALS')



        else:

            pass



    # Read data on worksheets and store in the dfs (dataframes) attribute   

    def read_from_sheet(self):

        

        # Authorization

        self.create_spreadsheet_object()

        

        

        # Open the Google spreadsheet (where sheet_name is the name of my sheet)

        try:

            sh = self.pygs_obj.open_by_key(self.sheet_key)

            self.sheet_updated=sh.updated

            # Get a list of all worksheets

            self.sheet_wks = [item.title for item in sh.worksheets()]

        except:

            raise RuntimeError("Could not open " + self.sheet_key)



        # If wks_name is a single string value, try to return a dataframe of that worksheet

        if isinstance(self.dfs_wks, str) and self.dfs_wks!='':

            try:

                wks = sh.worksheet_by_title(self.wks_name)

                self.dfs=wks.get_as_df(has_header=True, index_colum=None, start=None, 

                               end=None, numerize=True, empty_value='', 

                               value_render='FORMATTED_VALUE')

                return

            except:

                raise RuntimeError('Could not read from '+wks_name+' check that it is a valid worksheet')



        # If wks_name is None, set it equal to the full list of worksheets in this sheet

        if self.dfs_wks==None:

            self.dfs_wks=self.sheet_wks



        # Try to return a list of dataframes for each worksheet in the wks_name list    

        out_dfs=[]

        try:

            for w in tqdm(self.dfs_wks):

                wks = sh.worksheet_by_title(w)

                # Try to use a pygsheets method to create a dataframe; this can error when worksheet has 

                # titles but no data in the associated column; pulling data as dictionary, then converting

                # can be successful

                try:

                    out_dfs.append(wks.get_as_df(has_header=True, index_colum=None, start=None, 

                                                 end=None, numerize=True, empty_value='', 

                                                 value_render='FORMATTED_VALUE'))

                except:

                    out_dfs.append(pd.DataFrame(wks.get_all_records(empty_value='', head=1, majdim='ROWS', 

                                                                    numericise_data=True)))



            self.dfs=out_dfs

            return

        except:

            raise RuntimeError('Could not read from '+w+' check that it is a valid worksheet')

#  Using the SheetWrapper class, read the data in the hand-coding sheet

hand_coded=SheetWrapper(config_path=cred_path,

                 config_file=cred_file,

                 custom_creds=custom_creds,

                 api_key_file='',

                 auth_method='service',

                 sheet_key=hand_coding_file_key,

                 dfs_wks=None)



hand_coded.read_from_sheet()
# Output each dataframe to a separate csv file

for worksheet, df in zip(hand_coded.dfs_wks,hand_coded.dfs):

    worksheet=worksheet.replace('.','_')

    sheet_time=hand_coded.sheet_updated[:19].replace(':','')

    out_file_name=f'{worksheet}_{sheet_time}.csv'

    print(out_file_name)

    df.to_csv(path_or_buf=os.path.join(data_path,out_file_name),

              index=False)       