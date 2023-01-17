# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
d = [

     {

       "Country": "Thailand",

       "Date": "08-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "09-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "10-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "11-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "12-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "13-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "14-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "15-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "16-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "17-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "18-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "19-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "20-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "21-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "22-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "23-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "24-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "25-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "26-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "27-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "28-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "29-Feb-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "01-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "02-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "03-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "04-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "05-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "06-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "07-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "08-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "09-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "10-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "11-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "12-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "13-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "14-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "15-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "16-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "17-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "18-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "19-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "20-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "21-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "22-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "23-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "24-Mar-20",

       "HCW Cases": 5,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "25-Mar-20",

       "HCW Cases": 7,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "26-Mar-20",

       "HCW Cases": 10,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "27-Mar-20",

       "HCW Cases": 10,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "28-Mar-20",

       "HCW Cases": 12,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "29-Mar-20",

       "HCW Cases": 20,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "30-Mar-20",

       "HCW Cases": 22,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "31-Mar-20",

       "HCW Cases": 25,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "01-Apr-20",

       "HCW Cases": 26,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "02-Apr-20",

       "HCW Cases": 28,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "03-Apr-20",

       "HCW Cases": 33,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "04-Apr-20",

       "HCW Cases": 36,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "05-Apr-20",

       "HCW Cases": 38,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "06-Apr-20",

       "HCW Cases": 51,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "07-Apr-20",

       "HCW Cases": 54,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "08-Apr-20",

       "HCW Cases": 56,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "09-Apr-20",

       "HCW Cases": 60,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "10-Apr-20",

       "HCW Cases": 64,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "11-Apr-20",

       "HCW Cases": 66,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "12-Apr-20",

       "HCW Cases": 73,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "13-Apr-20",

       "HCW Cases": 76,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "14-Apr-20",

       "HCW Cases": 76,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "15-Apr-20",

       "HCW Cases": 76,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "16-Apr-20",

       "HCW Cases": 76,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "17-Apr-20",

       "HCW Cases": 76,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "18-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "19-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "20-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "21-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "22-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "23-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "24-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "25-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "26-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Thailand",

       "Date": "27-Apr-20",

       "HCW Cases": 77,

       "Total Physicians": 56032,

       "Physicians per 1000": 0.81,

       "ICU Beds per 1000": 2.1,

       "Population": 69430000

     },

     {

       "Country": "Mainland China",

       "Date": "11-Feb-20",

       "HCW Cases": 1716,

       "Total Physicians": 2412210,

       "Physicians per 1000": 1.7855,

       "ICU Beds per 1000": 4.2,

       "Population": 1393000000

     },

     {

       "Country": "Mainland China",

       "Date": "20-Feb-20",

       "HCW Cases": 2055,

       "Total Physicians": 2412210,

       "Physicians per 1000": 1.7855,

       "ICU Beds per 1000": 4.2,

       "Population": 1393000000

     },

     {

       "Country": "Mainland China",

       "Date": "25-Feb-20",

       "HCW Cases": 3387,

       "Total Physicians": 2412210,

       "Physicians per 1000": 1.7855,

       "ICU Beds per 1000": 4.2,

       "Population": 1393000000

     },

     {

       "Country": "Mainland China",

       "Date": "04-Mar-20",

       "HCW Cases": 3400,

       "Total Physicians": 2412210,

       "Physicians per 1000": 1.7855,

       "ICU Beds per 1000": 4.2,

       "Population": 1393000000

     },

     {

       "Country": "Italy",

       "Date": "15-Mar-20",

       "HCW Cases": 2026,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "16-Mar-20",

       "HCW Cases": 1700,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "17-Mar-20",

       "HCW Cases": 2629,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "18-Mar-20",

       "HCW Cases": 2898,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "19-Mar-20",

       "HCW Cases": 3559,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "20-Mar-20",

       "HCW Cases": 3654,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "21-Mar-20",

       "HCW Cases": 4268,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "22-Mar-20",

       "HCW Cases": 4824,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "23-Mar-20",

       "HCW Cases": 5211,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "24-Mar-20",

       "HCW Cases": 5760,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "25-Mar-20",

       "HCW Cases": 6205,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "26-Mar-20",

       "HCW Cases": 6414,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "27-Mar-20",

       "HCW Cases": 7145,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "28-Mar-20",

       "HCW Cases": 7763,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "29-Mar-20",

       "HCW Cases": 8358,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "30-Mar-20",

       "HCW Cases": 8956,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "31-Mar-20",

       "HCW Cases": 9512,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "02-Apr-20",

       "HCW Cases": 10657,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "03-Apr-20",

       "HCW Cases": 11252,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "04-Apr-20",

       "HCW Cases": 12052,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "05-Apr-20",

       "HCW Cases": 12252,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "06-Apr-20",

       "HCW Cases": 12681,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "07-Apr-20",

       "HCW Cases": 13121,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "08-Apr-20",

       "HCW Cases": 13522,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "09-Apr-20",

       "HCW Cases": 14066,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "10-Apr-20",

       "HCW Cases": 15314,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "11-Apr-20",

       "HCW Cases": 15724,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "12-Apr-20",

       "HCW Cases": 15891,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "13-Apr-20",

       "HCW Cases": 16050,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "15-Apr-20",

       "HCW Cases": 16050,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "17-Apr-20",

       "HCW Cases": 17306,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "20-Apr-20",

       "HCW Cases": 17997,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "22-Apr-20",

       "HCW Cases": 18553,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Italy",

       "Date": "24-Apr-20",

       "HCW Cases": 19942,

       "Total Physicians": 241136,

       "Physicians per 1000": 4.093,

       "ICU Beds per 1000": 3.18,

       "Population": 60360000

     },

     {

       "Country": "Spain",

       "Date": "24-Mar-20",

       "HCW Cases": 5600,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "02-Apr-20",

       "HCW Cases": 15000,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "25-Mar-20",

       "HCW Cases": 6500,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "06-Apr-20",

       "HCW Cases": 19000,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "09-Apr-20",

       "HCW Cases": 19400,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "11-Apr-20",

       "HCW Cases": 25000,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "Spain",

       "Date": "19-Apr-20",

       "HCW Cases": 30000,

       "Total Physicians": 180633,

       "Physicians per 1000": 3.88,

       "ICU Beds per 1000": 3,

       "Population": 46940000

     },

     {

       "Country": "South Korea",

       "Date": "20-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 121778,

       "Physicians per 1000": 2.366,

       "ICU Beds per 1000": 12.27,

       "Population": 51640000

     },

     {

       "Country": "South Korea",

       "Date": "03-Apr-20",

       "HCW Cases": 241,

       "Total Physicians": 121778,

       "Physicians per 1000": 2.366,

       "ICU Beds per 1000": 12.27,

       "Population": 51640000

     },

     {

       "Country": "Romania",

       "Date": "24-Mar-20",

       "HCW Cases": 103,

       "Total Physicians": 58600,

       "Physicians per 1000": 2.669,

       "ICU Beds per 1000": 3,

       "Population": 19410000

     },

     {

       "Country": "Romania",

       "Date": "02-Apr-20",

       "HCW Cases": 300,

       "Total Physicians": 58600,

       "Physicians per 1000": 2.669,

       "ICU Beds per 1000": 3,

       "Population": 19410000

     },

     {

       "Country": "Romania",

       "Date": "06-Apr-20",

       "HCW Cases": 152,

       "Total Physicians": 58600,

       "Physicians per 1000": 2.669,

       "ICU Beds per 1000": 3,

       "Population": 19410000

     },

     {

       "Country": "Romania",

       "Date": "16-Apr-20",

       "HCW Cases": 981,

       "Total Physicians": 58600,

       "Physicians per 1000": 2.669,

       "ICU Beds per 1000": 3,

       "Population": 19410000

     },

     {

       "Country": "Singapore",

       "Date": "15-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "16-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "17-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "18-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "19-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "20-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "21-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "22-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "23-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "24-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "25-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "26-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "27-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "28-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "29-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "01-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "02-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "03-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "04-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "05-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "06-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "07-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "08-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "09-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "10-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "11-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "12-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "13-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "14-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "15-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "16-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "17-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "18-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "19-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "20-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "21-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "22-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "23-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "24-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "25-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "26-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "27-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "28-Mar-20",

       "HCW Cases": 0,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "29-Mar-20",

       "HCW Cases": 1,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "30-Mar-20",

       "HCW Cases": 2,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "31-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "01-Apr-20",

       "HCW Cases": 6,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "02-Apr-20",

       "HCW Cases": 8,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "03-Apr-20",

       "HCW Cases": 8,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "04-Apr-20",

       "HCW Cases": 12,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "05-Apr-20",

       "HCW Cases": 14,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "06-Apr-20",

       "HCW Cases": 15,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "07-Apr-20",

       "HCW Cases": 18,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "08-Apr-20",

       "HCW Cases": 19,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "09-Apr-20",

       "HCW Cases": 20,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "10-Apr-20",

       "HCW Cases": 23,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "11-Apr-20",

       "HCW Cases": 23,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "12-Apr-20",

       "HCW Cases": 23,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "13-Apr-20",

       "HCW Cases": 23,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "14-Apr-20",

       "HCW Cases": 25,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "15-Apr-20",

       "HCW Cases": 25,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "16-Apr-20",

       "HCW Cases": 26,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "17-Apr-20",

       "HCW Cases": 27,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "18-Apr-20",

       "HCW Cases": 28,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "19-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "20-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "21-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "22-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "23-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "24-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "25-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Singapore",

       "Date": "26-Apr-20",

       "HCW Cases": 29,

       "Total Physicians": 13308,

       "Physicians per 1000": 2.36,

       "ICU Beds per 1000": 2.4,

       "Population": 5639000

     },

     {

       "Country": "Taiwan",

       "Date": "15-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "16-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "17-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "18-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "19-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "20-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "21-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "22-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "23-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "24-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "25-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "26-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "27-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "28-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "29-Feb-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "01-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "02-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "03-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "04-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "05-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "06-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "07-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "08-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "09-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "10-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "11-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "12-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "13-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "14-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "15-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "16-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "17-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "18-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "19-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "20-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "21-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "22-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "23-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "24-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "25-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "26-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "27-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "28-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "29-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "30-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "31-Mar-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "01-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "02-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "03-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "04-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "05-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "06-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "07-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "08-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "09-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "10-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "11-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "12-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "13-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "14-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "15-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "16-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "17-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "18-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "19-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "20-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "21-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "22-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "23-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "24-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "25-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "26-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Taiwan",

       "Date": "27-Apr-20",

       "HCW Cases": 3,

       "Total Physicians": 71340,

       "Physicians per 1000": 3,

       "ICU Beds per 1000": 5.9,

       "Population": 23780000

     },

     {

       "Country": "Hong Kong",

       "Date": "11-Feb-20",

       "HCW Cases": 0,

       "Total Physicians": 9827,

       "Physicians per 1000": 1.319,

       "ICU Beds per 1000": 4.89,

       "Population": 7451000

     },

     {

       "Country": "Hungary",

       "Date": "04-Apr-20",

       "HCW Cases": 85,

       "Total Physicians": 31576,

       "Physicians per 1000": 3.231,

       "ICU Beds per 1000": 7,

       "Population": 9773000

     },

     {

       "Country": "Germany",

       "Date": "02-Apr-20",

       "HCW Cases": 2300,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "09-Apr-20",

       "HCW Cases": 4700,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "10-Apr-20",

       "HCW Cases": 5100,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "11-Apr-20",

       "HCW Cases": 5300,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "12-Apr-20",

       "HCW Cases": 5500,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "13-Apr-20",

       "HCW Cases": 5713,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "14-Apr-20",

       "HCW Cases": 5846,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "15-Apr-20",

       "HCW Cases": 6058,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "16-Apr-20",

       "HCW Cases": 6395,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "17-Apr-20",

       "HCW Cases": 6711,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "18-Apr-20",

       "HCW Cases": 7043,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "19-Apr-20",

       "HCW Cases": 7293,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "20-Apr-20",

       "HCW Cases": 7413,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "21-Apr-20",

       "HCW Cases": 7575,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "22-Apr-20",

       "HCW Cases": 7862,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "23-Apr-20",

       "HCW Cases": 8102,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "24-Apr-20",

       "HCW Cases": 8326,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "25-Apr-20",

       "HCW Cases": 8539,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     },

     {

       "Country": "Germany",

       "Date": "26-Apr-20",

       "HCW Cases": 8744,

       "Total Physicians": 351195,

       "Physicians per 1000": 4.26,

       "ICU Beds per 1000": 8,

       "Population": 83020000

     }

    ]

med = pd.DataFrame(data=d)

med['Date'] = med['Date'].apply(pd.Timestamp)

med['Date'] = med['Date'].apply(lambda x: x.date())

med['Date'] = pd.to_datetime(med['Date'])

countries = list(med['Country'].unique())

med = med.groupby(['Country', 'Date']).last().sort_values(['Country', 'Date'])

med = med.rename(columns={'Total cases among medical staff' : 'Physician cases'})

pd.set_option('display.max_rows', med.shape[0]+1)



med.head()
cov = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0, names=['SNo',

                                                                                                         'Date',

                                                                                                         'Province',

                                                                                                         'Country',

                                                                                                         'Last Update',

                                                                                                         'Total Cases',

                                                                                                         'Total Deaths',

                                                                                                         'Total Recovered'])



cov['Date'] = cov['Date'].apply(pd.Timestamp)

cov['Date'] = cov['Date'].apply(lambda x: x.date())

cov['Date'] = pd.to_datetime(cov['Date'])

del cov['SNo']

del cov['Province']

del cov['Last Update']

cov = cov.loc[cov['Country'].isin(countries)]

cov = cov.groupby(['Country', 'Date']).sum()

cov['New Cases'] = cov.groupby('Country')['Total Cases'].diff()

cov['Day'] = cov.groupby('Country').cumcount()



pd.set_option('display.max_rows', cov.shape[0])



cov.head()



df = pd.concat([cov, med], axis=1, join='inner').sort_values(['Country', 'Date'])

df['CFR'] = 100 * df['Total Deaths'] / df['Total Cases']

df['Death Rate'] = 100 * df['Total Deaths'] / df['Population']

df['Ratio of total cases to total population'] = 100 * df['Total Cases'] / df['Population'] 

df['Ratio of HCW cases to total HCW'] = 100 * df['HCW Cases'] / df['Total Physicians']

df['Ratio of HCW cases to total cases'] = 100 * df['HCW Cases'] / df['Total Cases']





df.to_csv(r'Inner_join_by_date.csv')

pd.set_option('display.max_rows',50)



df.head()



dfo = pd.concat([cov, med], axis=1).sort_values(['Country', 'Date'])

dfo['CFR'] = 100 * dfo['Total Deaths'] / dfo['Total Cases']

dfo['Death Rate'] = 100 * dfo['Total Deaths'] / dfo['Population']

dfo['Ratio of total cases to total population'] = 100 * dfo['Total Cases'] / dfo['Population'] 

dfo['Ratio of HCW cases to total HCW'] = 100 * dfo['HCW Cases'] / dfo['Total Physicians']

dfo['Ratio of HCW cases to total cases'] = 100 * dfo['HCW Cases'] / dfo['Total Cases']

df.to_csv(r'Outer_join_by_date.csv')

# pd.set_option('display.max_rows', 10)



dfo.head()

N = 10

dfn = pd.concat([cov, med], axis=1).sort_values(['Country', 'Date'])

dfn['CFR'] = 100 * dfn['Total Deaths'] / dfn['Total Cases']

dfn['Death Rate'] = 100 * dfn['Total Deaths'] / dfn['Population']

dfn['Ratio of total cases to total population'] = 100 * dfn['Total Cases'] / dfn['Population'] 

dfn['Ratio of HCW cases to total HCW'] = 100 * dfn['HCW Cases'] / dfn['Total Physicians']

dfn['Ratio of HCW cases to total cases'] = 100 * dfn['HCW Cases'] / dfn['Total Cases']



dfn = dfn.loc[dfn['Total Cases'] > N]



dfn['Day'] = dfn.groupby('Country').cumcount()

df.to_csv(r'Outer_join_from_N.csv')



# pd.set_option('display.max_rows', 10)

dfn.head()
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)', 'rgb(86, 100, 18)', 'rgb(140, 214, 18)']

# Fig 4: Physician cases of total cases by country

c_last = df.groupby('Country').last().reset_index().sort_values('Ratio of HCW cases to total cases', ascending=False)

fig = px.bar(c_last, y='Country', x='Ratio of HCW cases to total cases', color='Country', orientation='h', text='Ratio of HCW cases to total cases')

fig.update_traces(texttemplate='%{text:.2s}%', textposition='outside')



fig.update_layout(showlegend=False,

                  title_text="Ratio of HCW cases to total cases by country")

fig.show()
# Fig 2: Physician cases of total cases vs. Days

fig = go.Figure()



for color, country in zip(colors, countries):

    c = dfn.reset_index().loc[(dfn.reset_index()['HCW Cases'] > 0) & (dfn.reset_index()['Country'] == country)]

    y_arg = 'Ratio of HCW cases to total cases'

    x_arg = 'Day'

    fig.add_trace(

        go.Scatter(x=c[x_arg], y=c[y_arg], name=country, mode='markers+lines', connectgaps=True, line=dict(color=color, width=2)))

    fig.update_xaxes(title_text='Days since the %dth case'%N)

    fig.update_yaxes(title_text=y_arg)



fig.update_layout(legend=dict(

                              x=0.8,

                              y=1.0,

                              bgcolor='rgba(255, 255, 255, 0)',

                              bordercolor='rgba(255, 255, 255, 0)'),

                              title_text="Ratio of HCW cases to total cases vs. Days")



fig.show()



fig = go.Figure()

for color, country in zip(colors, countries):

    c = dfn.reset_index().loc[(dfn.reset_index()['HCW Cases'] > 0) & (dfn.reset_index()['Country'] == country)]

    y_arg = 'Ratio of HCW cases to total cases'

    x_arg = 'Country'

    fig.add_trace(

        go.Box(x=c[x_arg], y=c[y_arg], showlegend=False, marker_color=color, quartilemethod="linear"))

    fig.update_xaxes(title_text=x_arg)

    fig.update_yaxes(title_text=y_arg)



fig.update_layout(legend=dict(

                              x=0.7,

                              y=1.0,

                              bgcolor='rgba(255, 255, 255, 0)',

                              bordercolor='rgba(255, 255, 255, 0)'),

                              title_text="Ratio of HCW cases to total cases vs. Days - Box plot")

fig.show()
# Fig 3: Total physician cases vs. Total cases worldwide

c = dfn.copy(deep=True).reset_index()

c = c.loc[~c['Country'].isin(['Hong Kong', 'Hungary'])]

c = c.groupby('Country').apply(lambda group: group.interpolate(method='polynomial', order=1))

c = c.loc[c['HCW Cases'].notna()]





counter = 1

count = pd.DataFrame(c.groupby('Day').count()['HCW Cases']).rename(columns={'HCW Cases' : 'Count'})

d = pd.concat([c.groupby('Day').sum(), count], axis=1).reset_index()

# d['Date'] = d['Date'].apply(lambda x: str(x.date()))

# pd.set_option('display.max_rows', d.shape[0]+1)

d['Ratio of total cases to total population'] = 100 * d['Total Cases'] / d['Population']



d['Ratio of HCW cases to total HCW'] = 100 * d['HCW Cases'] / d['Total Physicians']





fig = px.scatter(d.loc[d['Count'] >= counter], x='Ratio of total cases to total population', y='Ratio of HCW cases to total HCW', trendline='ols', size='Count', text='Count', title='Ratio of HCW cases to total HCW vs. Ratio of total cases to total population for countries with available data')

fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=5))

fig.show()



fig = go.Figure()

# countries=['Singapore']

# colors = ['rgb(227, 119, 194)']



for color, country in zip(colors, countries):

    c = dfn.reset_index().loc[(dfn.reset_index()['HCW Cases'] > 0) & (dfn.reset_index()['Country'] == country)]

    y_arg = 'Ratio of HCW cases to total HCW'

    x_arg = 'Ratio of total cases to total population'

    fig.add_trace(

        go.Scatter(x=c[x_arg], y=c[y_arg], name=country, mode='markers+lines', connectgaps=True, line=dict(color=color, width=3)))

    fig.update_xaxes(title_text=x_arg, type='log')

    fig.update_yaxes(title_text=y_arg, type='log')



fig.update_layout(showlegend=True, legend=dict(

                              x=0,

                              y=1.0,

                              bgcolor='rgba(255, 255, 255, 0)',

                              bordercolor='rgba(255, 255, 255, 0)'),

                              title_text='Ratio of HCW cases to total HCW vs. Ratio of total cases to total population for countries with available data')

fig.show()


fig = go.Figure()



x_arg = 'Country'

y1_arg = 'Ratio of HCW cases to total cases'

y2_arg = 'CFR'





c = df.groupby('Country').last().reset_index().sort_values(y2_arg, ascending=False)







fig.add_trace(

    go.Bar(x=c[x_arg], y=c[y1_arg], name=y1_arg, text=c[y1_arg], texttemplate='%{text:.2s}%', textposition='auto', marker_color='green'))

    

fig.add_trace(

    go.Scatter(x=c[x_arg], y=c[y2_arg], name=y2_arg, mode='markers+lines+text', marker=dict(size=6), line=dict(color='black', width=2), text=c[y2_arg], texttemplate='%{text:.2s}%', textposition='top right', textfont=dict(

                                                                                                                                                                                                                

                                                                                                                                                                                                                size=18,

                                                                                                                                                                                                                color="black"

                                                                                                                                                                                                            )))



fig.update_yaxes(title_text='Ratio (%)')

# fig.update_traces()

fig.update_layout(barmode='group',

    legend=dict(

                x=0.6,

                y=1.0,

                bgcolor='rgba(255, 255, 255, 0)',

                bordercolor='rgba(255, 255, 255, 0)'),

                title_text="Case fatality rate and ratio of HCW cases to total cases by country")

fig.show()
dfn.columns
c = df.groupby('Country').mean().reset_index()



fig = px.scatter(c.loc[c['HCW Cases'] > 0], x='Ratio of HCW cases to total HCW', y='Death Rate',trendline='ols', text='Country', title="Death rate vs. 'Ratio of HCW cases to total HCW' (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
c = df.groupby('Country').mean().reset_index()



fig = px.scatter(c.loc[c['HCW Cases'] > 0], x='Ratio of HCW cases to total HCW', y='Ratio of total cases to total population',trendline='ols', text='Country', title="Total cases from the general population vs. Physician cases of total physicians (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
c = df.groupby('Country').mean().reset_index()



fig = px.scatter(c.loc[c['HCW Cases'] > 0], x='Ratio of HCW cases to total HCW', y='Death Rate',trendline='ols', text='Country', title="Death rate vs. Ratio of HCW cases to total cases (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()



c = df.groupby('Country').last().reset_index()



fig = px.scatter(c.loc[c['HCW Cases'] > 0], x='Total Cases', y='HCW Cases',trendline='ols', text='Country', title="HCW cases vs. Total cases (last update)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()
c = df.groupby('Country').mean().reset_index()



fig = px.scatter(c.loc[c['HCW Cases'] > 0], x='Ratio of HCW cases to total cases', y='Death Rate',trendline='ols', text='Country', title="Death rate vs. Ratio of HCW cases to total cases (mean)")



fig.update_traces(textposition='top center', textfont=dict(size=12), marker=dict(size=15))

fig.show()