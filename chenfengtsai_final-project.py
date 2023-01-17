# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.express as px
counties = pd.read_csv("abridged_couties.csv")
counties
confirmed = pd.read_csv("time_series_covid19_confirmed_US.csv")
confirmed
counties["MaleAge<10"] = counties["PopMale<52010"] + counties["PopMale5-92010"]
counties["MaleAge10-19"] = counties["PopMale10-142010"] + counties["PopMale15-192010"]
counties["MaleAge20-34"] = counties["PopMale20-242010"] + counties["PopMale25-292010"] + counties["PopMale30-342010"] 
counties["MaleAge35-54"] = counties["PopMale35-442010"] + counties["PopMale45-542010"] 
counties["MaleAge55-74"] = counties["PopMale55-592010"] + counties["PopMale60-642010"] + counties["PopMale65-742010"]
counties["MaleAge>74"] = counties["PopMale75-842010"] + counties["PopMale>842010"]

counties["FmleAge<10"] = counties["PopFmle<52010"] + counties["PopFmle5-92010"]
counties["FmleAge10-19"] = counties["PopFmle10-142010"] + counties["PopFmle15-192010"]
counties["FmleAge20-34"] = counties["PopFmle20-242010"] + counties["PopFmle25-292010"] + counties["PopFmle30-342010"] 
counties["FmleAge35-54"] = counties["PopFmle35-442010"] + counties["PopFmle45-542010"] 
counties["FmleAge55-74"] = counties["PopFmle55-592010"] + counties["PopFmle60-642010"] + counties["PopFmle65-742010"]
counties["FmleAge>74"] = counties["PopFmle75-842010"] + counties["PopFmle>842010"]
counties = counties.loc[:, ["CountyName", "State", "PopTotalMale2017", "PopTotalFemale2017", "PopulationEstimate2018", "DiabetesPercentage", 
                 "HeartDiseaseMortality", "StrokeMortality", "Smokers_Percentage", "RespMortalityRate2014", "MaleAge<10", "MaleAge10-19",
                "MaleAge20-34", "MaleAge35-54", "MaleAge55-74", "MaleAge>74", "FmleAge<10", "FmleAge10-19", "FmleAge20-34", 
                 "FmleAge35-54","FmleAge55-74", "FmleAge>74"]]
counties
confirmed = confirmed.loc[:, ["Admin2", "Province_State", "2/1/20", "2/11/20", "2/21/20", "3/1/20", "3/11/20", "3/21/20", "4/1/20", "4/11/20", "4/18/20"]]
confirmed["CountyName"] = confirmed["Admin2"]
confirmed["State"] = confirmed["Province_State"]
confirmed = confirmed.drop("Admin2", axis = 1).drop("Province_State", axis = 1)
pd.merge(counties, confirmed, how = "inner", on = ["CountyName", "State"])